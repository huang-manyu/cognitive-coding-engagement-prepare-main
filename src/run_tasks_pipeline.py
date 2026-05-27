import argparse
import subprocess
import sys
from pathlib import Path


def get_task_dirs(tasks_root: Path) -> list[Path]:
    if not tasks_root.exists():
        raise FileNotFoundError(f"tasks 目录不存在: {tasks_root}")
    return sorted([p for p in tasks_root.iterdir() if p.is_dir()], key=lambda p: p.name)


def run_step(script_path: Path, task_id: str, cwd: Path) -> None:
    cmd = [sys.executable, str(script_path), task_id]
    print(f"  -> 执行: {script_path.name} {task_id}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def run_pipeline_for_task(task_id: str, project_root: Path) -> tuple[bool, str]:
    scripts = [
        project_root / "src" / "a_video_to_audio.py",
        project_root / "src" / "b_speaker_diarization.py",
        project_root / "src" / "c_automatic_speech_recognition.py",
        project_root / "src" / "d_crowd_detect.py",
        project_root / "src" / "e_merge_results.py",
    ]

    for script in scripts:
        if not script.exists():
            return False, f"脚本不存在: {script}"

    print(f"\n===== 开始任务: {task_id} =====")
    try:
        for script in scripts:
            run_step(script, task_id, project_root)
    except subprocess.CalledProcessError as e:
        return False, f"{e.cmd} 执行失败，退出码 {e.returncode}"
    except Exception as e:
        return False, str(e)

    return True, "完成"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="遍历 tasks 下任务并按 a/b/c/d/e_merge 顺序依次执行。"
    )
    parser.add_argument(
        "--tasks-root",
        default="tasks",
        help="任务目录路径（默认: tasks）",
    )
    parser.add_argument(
        "--task-id",
        default=None,
        help="只运行指定 task_id（可选）",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    tasks_root = (project_root / args.tasks_root).resolve()

    if args.task_id:
        task_ids = [args.task_id]
    else:
        task_dirs = get_task_dirs(tasks_root)
        task_ids = [p.name for p in task_dirs]

    if not task_ids:
        print(f"未找到任务目录: {tasks_root}")
        return 0

    print(f"项目目录: {project_root}")
    print(f"任务目录: {tasks_root}")
    print(f"待处理任务数: {len(task_ids)}")

    results: list[tuple[str, bool, str]] = []
    for task_id in task_ids:
        ok, msg = run_pipeline_for_task(task_id, project_root)
        results.append((task_id, ok, msg))
        state = "成功" if ok else "失败"
        print(f"===== 任务结束: {task_id} [{state}] {msg} =====")

    success_count = sum(1 for _, ok, _ in results if ok)
    fail_count = len(results) - success_count

    print("\n===== 汇总 =====")
    for task_id, ok, msg in results:
        state = "成功" if ok else "失败"
        print(f"- {task_id}: {state} | {msg}")
    print(f"总计: {len(results)}，成功: {success_count}，失败: {fail_count}")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
