import json
import sys
from pathlib import Path
from moviepy import VideoFileClip

# 提取间隔（秒）
INTERVAL_SECONDS = 2


def extract_frames(task_id: str):
    task_dir = Path("tasks") / task_id
    input_video = task_dir / "input" / "video.mp4"

    if not input_video.exists():
        raise FileNotFoundError(f"输入视频不存在: {input_video}")

    output_dir = task_dir / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    video = VideoFileClip(str(input_video))
    duration = video.duration
    t = 0.0
    idx = 0

    print(f"正在提取帧: {input_video}，间隔 {INTERVAL_SECONDS}s，总时长 {duration:.1f}s")
    while t < duration:
        frame = video.get_frame(t)
        from PIL import Image
        img = Image.fromarray(frame)
        img.save(str(output_dir / f"image_{idx}.jpg"))
        idx += 1
        t += INTERVAL_SECONDS

    video.close()

    config = {"interval": INTERVAL_SECONDS}
    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"已提取 {idx} 帧到 {output_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: uv run .\\src\\e_extract_frames.py <task_id>")
        sys.exit(1)

    extract_frames(sys.argv[1])
