import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


GPUS = [0, 1, 2, 3, 4, 5]
CHUNK_SEC = 30
SAMPLE_RATE = 16000
MODEL_ID = "/data/disk1/guohaoran/model/whisper-large-v3"
DEFAULT_HF_MODEL_ID = "openai/whisper-large-v3"


def resolve_model_id() -> str:
    """解析可用模型来源：环境变量 > 常见本地路径 > Hugging Face 仓库。"""
    candidates = []
    env_model = os.getenv("WHISPER_MODEL_ID") or os.getenv("MODEL_ID")
    if env_model:
        candidates.append(env_model)
    candidates.extend([
        MODEL_ID,
        "D:/Development/models/whisper-large-v3",
    ])

    for candidate in candidates:
        # 本地目录存在则优先使用（跨平台兼容路径写法）
        if candidate and Path(candidate).exists():
            return candidate

    return DEFAULT_HF_MODEL_ID


def resolve_devices() -> list[str]:
    """自动选择可用设备：优先 GPU，无 CUDA 时回退到 CPU。"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        enabled_gpus = [gpu_id for gpu_id in GPUS if 0 <= gpu_id < gpu_count]
        if not enabled_gpus:
            enabled_gpus = list(range(gpu_count))
        return [f"cuda:{gpu_id}" for gpu_id in enabled_gpus]
    return ["cpu"]


def load_model_on_device(device: str, model_source: str):
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_source, dtype=dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_source)
    return model, processor, device, dtype


def transcribe_chunk(args):
    """转录单个 chunk，返回 (chunk_idx, segments)"""
    chunk_idx, audio_chunk, model, processor, device, dtype = args
    inputs = processor(audio_chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    input_features = inputs.input_features.to(device, dtype=dtype)

    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            language="chinese",
            task="transcribe",
            return_timestamps=True,
        )

    token_ids = predicted_ids[0].tolist()
    text = processor.tokenizer.decode(token_ids, skip_special_tokens=False, decode_with_timestamps=True)
    segments = parse_timestamp_text(text, chunk_idx * CHUNK_SEC)
    return chunk_idx, segments


def parse_timestamp_text(text: str, time_offset: float):
    """解析 Whisper 带时间戳的输出，如 <|0.00|>你好<|0.50|>"""
    import re
    segments = []
    for m in re.finditer(r"<\|(\d+\.\d+)\|>(.*?)<\|(\d+\.\d+)\|>", text):
        start = float(m.group(1)) + time_offset
        content = m.group(2).strip()
        end = float(m.group(3)) + time_offset
        if content:
            segments.append({
                "text": content,
                "start": round(start, 3),
                "end": round(end, 3),
            })
    return segments


def audio_transcribe(task_id: str):
    task_dir = Path("tasks") / task_id
    input_audio = task_dir / "audio" / "audio.mp3"
    output_dir = task_dir / "automatic_speech_recognition"
    output_file = output_dir / "output.json"

    if not input_audio.exists():
        raise FileNotFoundError(f"音频文件不存在: {input_audio}")

    output_dir.mkdir(parents=True, exist_ok=True)
    model_source = resolve_model_id()
    devices = resolve_devices()

    # 加载音频并切分
    print(f"正在加载音频: {input_audio}")
    audio, _ = librosa.load(str(input_audio), sr=SAMPLE_RATE)
    chunk_samples = CHUNK_SEC * SAMPLE_RATE

    audio_chunks = []
    for i in range(0, len(audio), chunk_samples):
        audio_chunks.append(audio[i:i + chunk_samples])
    print(f"音频切分为 {len(audio_chunks)} 个 {CHUNK_SEC}s chunk，使用设备 {devices}")

    # 每个设备加载一个模型
    print("正在加载模型到各设备...")
    resources = []
    for device in devices:
        model, processor, device, dtype = load_model_on_device(device, model_source)
        resources.append((model, processor, device, dtype))
        print(f"  设备 {device} 就绪")

    # 轮询分配 chunk 到设备
    tasks = []
    for i, chunk in enumerate(audio_chunks):
        res = resources[i % len(resources)]
        tasks.append((i, chunk, *res))

    # 多线程并行转录
    print("正在转录...")
    all_segments = [None] * len(audio_chunks)
    with ThreadPoolExecutor(max_workers=len(resources)) as executor:
        futures = {executor.submit(transcribe_chunk, t): t[0] for t in tasks}
        done = 0
        for future in as_completed(futures):
            chunk_idx, segs = future.result()
            all_segments[chunk_idx] = segs
            done += 1
            if done % 5 == 0 or done == len(tasks):
                print(f"  进度: {done}/{len(tasks)}")

    # 合并所有 chunk 结果
    segments = []
    for segs in all_segments:
        if segs:
            segments.extend(segs)
    segments.sort(key=lambda x: x["start"])
    print(f"Whisper 共输出 {len(segments)} 个片段")

    output_data = {
        "task_id": task_id,
        "source_audio": str(input_audio),
        "model": model_source,
        "segments": segments,
        "segment_count": len(segments),
    }

    output_file.write_text(json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"转录结果已导出: {output_file}，共 {len(segments)} 个片段")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: uv run src/c_automatic_speech_recognition.py <task_id>")
        sys.exit(1)

    audio_transcribe(sys.argv[1])
