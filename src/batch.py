from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
from importlib import import_module
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter, sleep


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

STAGE_FUNCTIONS = {
    "a": ("a_video_to_audio", "video_to_audio"),
    "b": ("b_speaker_diarization", "audio_to_info"),
    "c": ("c_automatic_speech_recognition", "audio_transcribe"),
    "d": ("d_crowd_detect", "crowd_detect"),
    "e": ("e_extract_frames", "extract_frames"),
    "f": ("f_image_understanding", "image_understanding"),
    "g": ("g_merge_results", "merge_results"),
}
PARALLEL_STAGES = ("b", "c", "d", "e")
PARALLEL_STAGE_POLL_INTERVAL_SEC = 0.2
PRINT_LOCK = threading.Lock()


@dataclass(slots=True)
class StageResult:
    stage: str
    success: bool
    duration_sec: float
    error: str | None = None


@dataclass(slots=True)
class TaskResult:
    task_id: str
    success: bool
    duration_sec: float
    stage_results: list[StageResult]


def log(message: str, prefix: str | None = None) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] {message}" if prefix is None else f"[{timestamp}][{prefix}] {message}"
    with PRINT_LOCK:
        print(line, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run task pipelines in batch mode. Each task runs "
            "a -> (b, c, d, e in parallel) -> f -> g."
        )
    )
    parser.add_argument(
        "task_ids",
        nargs="+",
        help="One or more task IDs. Comma-separated values are also accepted.",
    )
    parser.add_argument(
        "--task-workers",
        type=int,
        default=None,
        help="Maximum number of task pipelines to run at the same time. Defaults to all tasks.",
    )
    return parser.parse_args()


def normalize_task_ids(raw_task_ids: list[str]) -> list[str]:
    task_ids: list[str] = []
    for raw in raw_task_ids:
        for part in raw.split(","):
            task_id = part.strip()
            if task_id:
                task_ids.append(task_id)
    return task_ids


def dedupe_task_ids(task_ids: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    duplicates: list[str] = []

    for task_id in task_ids:
        if task_id in seen:
            duplicates.append(task_id)
            continue
        seen.add(task_id)
        unique.append(task_id)

    if duplicates:
        log(f"Duplicate task IDs ignored: {', '.join(duplicates)}", prefix="batch")

    return unique


def get_stage_callable(stage: str):
    module_name, func_name = STAGE_FUNCTIONS[stage]
    module = import_module(module_name)
    return getattr(module, func_name)


def get_stage_script(stage: str) -> Path:
    module_name, _ = STAGE_FUNCTIONS[stage]
    script_path = SRC_DIR / f"{module_name}.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Stage script not found: {script_path}")
    return script_path


def run_stage(task_id: str, stage: str) -> StageResult:
    prefix = f"{task_id}/{stage}"
    module_name, func_name = STAGE_FUNCTIONS[stage]

    log(f"Starting {module_name}.{func_name}()", prefix=prefix)
    start_time = perf_counter()

    try:
        stage_func = get_stage_callable(stage)
        stage_func(task_id)
    except Exception as exc:
        duration_sec = perf_counter() - start_time
        log(f"Failed with error: {exc}", prefix=prefix)
        return StageResult(stage=stage, success=False, duration_sec=duration_sec, error=str(exc))

    duration_sec = perf_counter() - start_time
    log(f"Finished successfully in {duration_sec:.1f}s", prefix=prefix)
    return StageResult(stage=stage, success=True, duration_sec=duration_sec)


def start_parallel_stage(task_id: str, stage: str) -> tuple[subprocess.Popen[bytes], float]:
    prefix = f"{task_id}/{stage}"
    module_name, func_name = STAGE_FUNCTIONS[stage]
    script_path = get_stage_script(stage)

    log(f"Starting {module_name}.{func_name}()", prefix=prefix)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    process = subprocess.Popen(
        [sys.executable, str(script_path), task_id],
        cwd=ROOT_DIR,
        env=env,
    )
    return process, perf_counter()


def run_parallel_stages(task_id: str) -> list[StageResult]:
    results_by_stage: dict[str, StageResult] = {}
    running_processes: dict[str, tuple[subprocess.Popen[bytes], float]] = {}

    for stage in PARALLEL_STAGES:
        try:
            running_processes[stage] = start_parallel_stage(task_id, stage)
        except Exception as exc:
            log(f"Failed with error: {exc}", prefix=f"{task_id}/{stage}")
            results_by_stage[stage] = StageResult(
                stage=stage,
                success=False,
                duration_sec=0.0,
                error=str(exc),
            )

    while running_processes:
        for stage, (process, start_time) in list(running_processes.items()):
            returncode = process.poll()
            if returncode is None:
                continue

            duration_sec = perf_counter() - start_time
            prefix = f"{task_id}/{stage}"
            if returncode == 0:
                log(f"Finished successfully in {duration_sec:.1f}s", prefix=prefix)
                results_by_stage[stage] = StageResult(
                    stage=stage,
                    success=True,
                    duration_sec=duration_sec,
                )
            else:
                error = f"subprocess exited with code {returncode}"
                log(f"Failed with error: {error}", prefix=prefix)
                results_by_stage[stage] = StageResult(
                    stage=stage,
                    success=False,
                    duration_sec=duration_sec,
                    error=error,
                )
            del running_processes[stage]

        if running_processes:
            sleep(PARALLEL_STAGE_POLL_INTERVAL_SEC)

    return [results_by_stage[stage] for stage in PARALLEL_STAGES]


def run_task_pipeline(task_id: str) -> TaskResult:
    log("Pipeline started", prefix=task_id)
    pipeline_start = perf_counter()
    stage_results: list[StageResult] = []

    a_result = run_stage(task_id, "a")
    stage_results.append(a_result)
    if not a_result.success:
        duration_sec = perf_counter() - pipeline_start
        log("Pipeline stopped after stage a", prefix=task_id)
        return TaskResult(task_id=task_id, success=False, duration_sec=duration_sec, stage_results=stage_results)

    parallel_results = run_parallel_stages(task_id)
    stage_results.extend(parallel_results)
    failed_parallel = [result.stage for result in parallel_results if not result.success]
    if failed_parallel:
        duration_sec = perf_counter() - pipeline_start
        log(
            f"Pipeline stopped after parallel stages failed: {', '.join(failed_parallel)}",
            prefix=task_id,
        )
        return TaskResult(task_id=task_id, success=False, duration_sec=duration_sec, stage_results=stage_results)

    f_result = run_stage(task_id, "f")
    stage_results.append(f_result)
    if not f_result.success:
        duration_sec = perf_counter() - pipeline_start
        log(f"Pipeline failed in stage f after {duration_sec:.1f}s", prefix=task_id)
        return TaskResult(task_id=task_id, success=False, duration_sec=duration_sec, stage_results=stage_results)

    g_result = run_stage(task_id, "g")
    stage_results.append(g_result)
    duration_sec = perf_counter() - pipeline_start
    success = g_result.success

    if success:
        log(f"Pipeline finished successfully in {duration_sec:.1f}s", prefix=task_id)
    else:
        log(f"Pipeline failed in stage g after {duration_sec:.1f}s", prefix=task_id)

    return TaskResult(task_id=task_id, success=success, duration_sec=duration_sec, stage_results=stage_results)


def format_stage_summary(result: StageResult) -> str:
    status = "ok" if result.success else "failed"
    detail = ""
    if result.error:
        detail = f", error={result.error}"
    return f"{result.stage}:{status} ({result.duration_sec:.1f}s{detail})"


def main() -> int:
    args = parse_args()
    task_ids = dedupe_task_ids(normalize_task_ids(args.task_ids))
    if not task_ids:
        raise SystemExit("No valid task IDs were provided.")

    if args.task_workers is not None and args.task_workers <= 0:
        raise SystemExit("--task-workers must be greater than 0.")

    os.chdir(ROOT_DIR)
    task_workers = args.task_workers or len(task_ids)
    task_workers = min(task_workers, len(task_ids))

    log(
        f"Starting batch for {len(task_ids)} task(s) with {task_workers} task worker(s)",
        prefix="batch",
    )

    results_by_task: dict[str, TaskResult] = {}
    with ThreadPoolExecutor(max_workers=task_workers) as executor:
        future_to_task = {
            executor.submit(run_task_pipeline, task_id): task_id
            for task_id in task_ids
        }
        for future in as_completed(future_to_task):
            task_id = future_to_task[future]
            try:
                results_by_task[task_id] = future.result()
            except Exception as exc:
                log(f"Unhandled pipeline error: {exc}", prefix=task_id)
                results_by_task[task_id] = TaskResult(
                    task_id=task_id,
                    success=False,
                    duration_sec=0.0,
                    stage_results=[],
                )

    failures = 0
    for task_id in task_ids:
        result = results_by_task[task_id]
        stage_summary = ", ".join(format_stage_summary(stage) for stage in result.stage_results) or "no stages run"
        status = "SUCCESS" if result.success else "FAILED"
        log(
            f"{status} total={result.duration_sec:.1f}s | {stage_summary}",
            prefix=f"summary/{task_id}",
        )
        if not result.success:
            failures += 1

    if failures:
        log(f"Batch finished with {failures} failed task(s)", prefix="batch")
        return 1

    log("Batch finished successfully", prefix="batch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
