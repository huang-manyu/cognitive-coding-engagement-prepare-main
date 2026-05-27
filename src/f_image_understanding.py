import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from gpt_client import call_gpt_vision

# 超参数
MAX_WORKERS = 4
MODEL = "gpt-5"
SYSTEM_PROMPT = "你是一个图像理解专家。请仔细观察图像，描述图像中的场景、人物、动作、物体和文字等关键信息。"
USER_PROMPT = "请描述这张图片中的内容。"


def process_image(image_path: Path, output_dir: Path) -> str:
    stem = image_path.stem  # image_0
    output_path = output_dir / f"{stem}.json"

    if output_path.exists():
        return f"跳过（已存在）: {stem}"

    response = call_gpt_vision(
        image_path=image_path,
        user_prompt=USER_PROMPT,
        system_prompt=SYSTEM_PROMPT,
        model=MODEL,
    )

    result = {
        "image": image_path.name,
        "model": MODEL,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": USER_PROMPT,
        "response": response,
    }
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return f"完成: {stem}"


def image_sort_key(p: Path) -> int:
    m = re.search(r"(\d+)", p.stem)
    return int(m.group(1)) if m else 0


def image_understanding(task_id: str):
    task_dir = Path("tasks") / task_id
    frames_dir = task_dir / "frames"

    if not frames_dir.exists():
        raise FileNotFoundError(f"帧目录不存在: {frames_dir}")

    images = sorted(frames_dir.glob("image_*.jpg"), key=image_sort_key)
    if not images:
        raise FileNotFoundError(f"未找到帧图片: {frames_dir}")

    output_dir = task_dir / "image_understanding"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"共 {len(images)} 张图片，{MAX_WORKERS} 线程并发，模型 {MODEL}")

    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_image, img, output_dir): img for img in images}
        for future in as_completed(futures):
            done += 1
            msg = future.result()
            print(f"[{done}/{len(images)}] {msg}")

    print(f"全部完成，结果输出到 {output_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: uv run .\\src\\f_image_understanding.py <task_id>")
        sys.exit(1)

    image_understanding(sys.argv[1])
