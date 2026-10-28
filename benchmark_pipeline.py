#!/usr/bin/env python3
"""Run repeated KITTI perception playback benchmarks and collect telemetry.

Run with ``pixi run benchmark-pipeline`` for the default three repetitions.
Each run gets its own ROS log and CPU/GPU resource samples. The controller
builds once, then asks the launch file to shut down when bag playback ends.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import platform
import psutil
import re
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from compute_pipeline_telemetry import PATTERNS, parse_log, summarize

ROOT = Path(__file__).resolve().parent
GPU_SAMPLE_FIELDS = (
    "timestamp",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "power.draw",
)
RESOURCE_FIELDS = (
    "timestamp_utc",
    "elapsed_s",
    "system_cpu_pct",
    "pipeline_cpu_pct_of_host",
    "pipeline_cpu_cores",
    "pipeline_rss_mb",
    "system_ram_pct",
    "gpu_util_pct",
    "gpu_memory_util_pct",
    "gpu_memory_used_mb",
    "gpu_memory_total_mb",
    "gpu_power_w",
)
PLAYBACK_START_MARKER = "BENCHMARK_BAG_PLAYBACK_STARTED"
BAG_EXIT_PATTERN = re.compile(r"BENCHMARK_BAG_EXIT_CODE=(-?\d+)")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def optional_float(value: str) -> float | None:
    value = value.strip()
    if not value or value.upper() in {"N/A", "[N/A]", "NOT SUPPORTED"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def query_gpu(fields: tuple[str, ...]) -> list[str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    rows = list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
    if not rows:
        raise RuntimeError("nvidia-smi returned no GPU rows")
    return [value.strip() for value in rows[0]]


def cpu_model() -> str | None:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    return platform.processor() or None


def git_metadata() -> dict[str, Any]:
    if not shutil.which("git"):
        return {"commit": None, "worktree_dirty": None}
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "worktree_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def bag_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def process_tree(root: psutil.Process, known: dict[int, psutil.Process]) -> list[psutil.Process]:
    try:
        current = [root, *root.children(recursive=True)]
    except psutil.Error:
        current = [root]

    for process in current:
        if process.pid not in known:
            known[process.pid] = process
            try:
                process.cpu_percent(interval=None)
            except psutil.Error:
                pass
    return current


def sample_resources(
    root: psutil.Process,
    known_processes: dict[int, psutil.Process],
    started: float,
) -> tuple[dict[str, Any], str | None]:
    current_processes = process_tree(root, known_processes)
    process_cpu_raw = 0.0
    process_rss = 0
    for process in current_processes:
        try:
            process_cpu_raw += process.cpu_percent(interval=None)
            process_rss += process.memory_info().rss
        except psutil.Error:
            continue

    row: dict[str, Any] = {
        "timestamp_utc": utc_now(),
        "elapsed_s": round(time.monotonic() - started, 3),
        "system_cpu_pct": psutil.cpu_percent(interval=None),
        "pipeline_cpu_pct_of_host": process_cpu_raw / max(psutil.cpu_count() or 1, 1),
        "pipeline_cpu_cores": process_cpu_raw / 100.0,
        "pipeline_rss_mb": process_rss / (1024.0 * 1024.0),
        "system_ram_pct": psutil.virtual_memory().percent,
    }
    error = None
    try:
        gpu = query_gpu(GPU_SAMPLE_FIELDS)
        row.update(
            {
                "gpu_util_pct": optional_float(gpu[1]),
                "gpu_memory_util_pct": optional_float(gpu[2]),
                "gpu_memory_used_mb": optional_float(gpu[3]),
                "gpu_memory_total_mb": optional_float(gpu[4]),
                "gpu_power_w": optional_float(gpu[5]),
            }
        )
    except (OSError, subprocess.SubprocessError, RuntimeError, IndexError) as exc:
        error = str(exc)
        row.update({field: None for field in RESOURCE_FIELDS[7:]})
    return row, error


def terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    for sig, timeout in ((signal.SIGINT, 25), (signal.SIGTERM, 8), (signal.SIGKILL, 3)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            continue


def build_launch_command(bag_path: Path, play_rate: float, playback_delay: float) -> list[str]:
    launch_args = [
        "ros2",
        "launch",
        "kitti_r1_perception",
        "myperception.launch.py",
        f"bag_path:={bag_path}",
        f"play_rate:={play_rate}",
        f"playback_delay:={playback_delay}",
        "shutdown_on_bag_exit:=true",
    ]
    setup_script = ROOT / "install" / "setup.bash"
    command = f"source {shlex.quote(str(setup_script))} && exec {shlex.join(launch_args)}"
    return ["bash", "-c", command]


def run_one(
    run_index: int,
    run_dir: Path,
    bag_path: Path,
    play_rate: float,
    playback_delay: float,
    sample_interval: float,
    run_timeout: float,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=False)
    pipeline_log = run_dir / "pipeline.log"
    resource_log = run_dir / "resource_usage.csv"
    command = build_launch_command(bag_path, play_rate, playback_delay)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    rows: list[dict[str, Any]] = []
    monitor_errors: list[str] = []
    status = "running"
    interrupted = False
    launch_started_utc = utc_now()
    launch_started = time.monotonic()
    playback_started_utc: str | None = None
    playback_started: float | None = None
    playback_finished: float | None = None
    log_offset = 0
    log_tail = b""
    process: subprocess.Popen[str] | None = None
    psutil.cpu_percent(interval=None)

    with pipeline_log.open("w", encoding="utf-8", errors="replace") as log_stream:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            root_process = psutil.Process(process.pid)
            known_processes: dict[int, psutil.Process] = {root_process.pid: root_process}
            last_sample: float | None = None
            while process.poll() is None:
                if run_timeout > 0 and time.monotonic() - launch_started >= run_timeout:
                    status = "timed_out"
                    terminate_process_group(process)
                    break
                try:
                    time.sleep(0.2)
                except KeyboardInterrupt:
                    status = "interrupted"
                    interrupted = True
                    terminate_process_group(process)
                    break

                with pipeline_log.open("rb") as reader:
                    reader.seek(log_offset)
                    new_bytes = reader.read()
                    log_offset = reader.tell()
                combined_bytes = log_tail + new_bytes
                log_tail = combined_bytes[-len(PLAYBACK_START_MARKER):]
                now = time.monotonic()

                if playback_started is None and PLAYBACK_START_MARKER.encode() in combined_bytes:
                    playback_started = now
                    playback_started_utc = utc_now()
                    psutil.cpu_percent(interval=None)
                    for monitored_process in process_tree(root_process, known_processes):
                        try:
                            monitored_process.cpu_percent(interval=None)
                        except psutil.Error:
                            continue
                    last_sample = now

                exit_match = BAG_EXIT_PATTERN.search(combined_bytes.decode("utf-8", errors="replace"))
                if exit_match and playback_started is not None and playback_finished is None:
                    playback_finished = now

                if process.poll() is not None:
                    break
                if (
                    playback_started is not None
                    and playback_finished is None
                    and last_sample is not None
                    and now - last_sample >= sample_interval
                ):
                    row, monitor_error = sample_resources(root_process, known_processes, playback_started)
                    rows.append(row)
                    last_sample = now
                    if monitor_error and monitor_error not in monitor_errors:
                        monitor_errors.append(monitor_error)
        except psutil.Error as exc:
            monitor_errors.append(f"Process monitoring failed: {exc}")
            while process.poll() is None:
                try:
                    time.sleep(min(sample_interval, 1.0))
                except KeyboardInterrupt:
                    status = "interrupted"
                    interrupted = True
                    terminate_process_group(process)
                    break

    if process is None:
        raise RuntimeError("Pipeline process did not start")
    if process.poll() is None:
        process.wait()
    exit_code = process.returncode
    launch_duration_s = time.monotonic() - launch_started
    log_text = pipeline_log.read_text(errors="replace")
    bag_exit_match = BAG_EXIT_PATTERN.search(log_text)
    bag_exit_code = int(bag_exit_match.group(1)) if bag_exit_match else None
    playback_started_in_log = PLAYBACK_START_MARKER in log_text
    if playback_started is not None and playback_finished is None and bag_exit_match:
        playback_finished = time.monotonic()
    playback_duration_s = (
        playback_finished - playback_started
        if playback_started is not None and playback_finished is not None
        else None
    )

    if status == "running":
        if exit_code != 0:
            status = "failed"
        elif not playback_started_in_log:
            status = "missing_playback_start_marker"
        elif bag_exit_code is None:
            status = "missing_bag_exit_marker"
        elif bag_exit_code != 0:
            status = "bag_playback_failed"
        elif not rows:
            status = "no_resource_samples"
        else:
            status = "completed"
        if status == "completed" and monitor_errors:
            status = "completed_with_monitor_errors"

    with resource_log.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=RESOURCE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    raw_telemetry = parse_log(pipeline_log)
    timing_summary = {name: summarize(values) for name, values in raw_telemetry.items()}
    resource_summary = {}
    for field in RESOURCE_FIELDS[2:]:
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        resource_summary[field] = summarize(values)

    return {
        "run": run_index,
        "status": status,
        "launch_started_utc": launch_started_utc,
        "playback_started_utc": playback_started_utc,
        "launch_duration_s": launch_duration_s,
        "playback_duration_s": playback_duration_s,
        "exit_code": exit_code,
        "bag_exit_code": bag_exit_code,
        "command": command,
        "pipeline_log": str(pipeline_log.relative_to(ROOT)),
        "resource_log": str(resource_log.relative_to(ROOT)),
        "resource_samples": len(rows),
        "monitor_errors": monitor_errors,
        "timing": timing_summary,
        "resources": resource_summary,
        "interrupted": interrupted,
    }


def aggregate_run_means(runs: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for metric in PATTERNS:
        means = [
            run["timing"][metric]["mean"]
            for run in runs
            if run["timing"].get(metric) is not None
        ]
        aggregate[metric] = summarize(means)
    return aggregate


def write_outputs(output_dir: Path, metadata: dict[str, Any], runs: list[dict[str, Any]]) -> None:
    completed_runs = [run for run in runs if run["status"] == "completed"]
    summary = {
        "metadata": metadata,
        "runs": runs,
        "median_of_run_means": aggregate_run_means(completed_runs),
    }
    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    metric_columns = {
        "lidar_callback_mean_ms": "LiDAR Total Callback Cost",
        "camera_callback_mean_ms": "Camera Total Callback Cost",
        "lidar_pre_callback_age_mean_ms": "LiDAR Pre-Callback Age",
        "camera_pre_callback_age_mean_ms": "Camera Pre-Callback Age",
        "camera_inference_mean_ms": "Pure ONNX Inference",
        "fusion_compute_mean_ms": "Fusion Compute Interval",
        "fusion_callback_mean_ms": "Fusion Total Callback",
        "bev_render_mean_ms": "BEV Total Callback Cost",
    }
    csv_fields = [
        "run",
        "status",
        "playback_duration_s",
        "resource_samples",
        "system_cpu_pct_mean",
        "pipeline_cpu_pct_of_host_mean",
        "pipeline_cpu_cores_mean",
        "pipeline_rss_mb_peak",
        "system_ram_pct_mean",
        "gpu_util_pct_mean",
        "gpu_util_pct_peak",
        "gpu_memory_used_mb_peak",
        *metric_columns,
    ]
    with (output_dir / "benchmark_runs.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
        writer.writeheader()
        for run in runs:
            row: dict[str, Any] = {
                "run": run["run"],
                "status": run["status"],
                "playback_duration_s": round(run["playback_duration_s"], 3) if run["playback_duration_s"] is not None else "",
                "resource_samples": run["resource_samples"],
            }
            for field in RESOURCE_FIELDS[2:]:
                stats = run["resources"].get(field)
                if field in {"pipeline_rss_mb", "gpu_memory_used_mb"}:
                    suffix = "peak"
                else:
                    suffix = "mean"
                column = f"{field}_{suffix}"
                if column in csv_fields:
                    row[column] = round(stats["max" if suffix == "peak" else "mean"], 3) if stats else ""
            gpu_util_stats = run["resources"].get("gpu_util_pct")
            if gpu_util_stats:
                row["gpu_util_pct_peak"] = round(gpu_util_stats["max"], 3)
            for column, metric in metric_columns.items():
                stats = run["timing"].get(metric)
                row[column] = round(stats["mean"], 3) if stats else ""
            writer.writerow(row)

    report_lines = [
        "# KITTI Perception Benchmark",
        "",
        f"Created: {metadata['created_utc']}",
        f"Bag: `{metadata['bag_path']}` ({metadata['bag_size_bytes']:,} bytes)",
        f"Playback rate: {metadata['play_rate']}x; startup delay: {metadata['playback_delay_s']} s",
        f"Requested repetitions: {metadata['requested_runs']}",
        f"Git revision: `{metadata['git']['commit']}`; worktree dirty: {metadata['git']['worktree_dirty']}",
        f"Hardware: {metadata['system']['cpu_model']}; {metadata['system']['gpu_name']}",
        "",
        "Resource percentages are sampled once per configured interval. Pipeline CPU is normalized to total logical CPU capacity; GPU utilization is device-wide.",
        "",
        "| Run | Status | Playback (s) | System CPU (%) | Pipeline CPU (%) | GPU mean / peak (%) | VRAM peak (MiB) | LiDAR callback (ms) | Camera inference (ms) | Fusion compute (ms) | Fusion callback (ms) | BEV render (ms) |",
        "| ---: | :--- | ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in runs:
        resources = run["resources"]
        timing = run["timing"]
        sys_cpu = resources.get("system_cpu_pct")
        pipeline_cpu = resources.get("pipeline_cpu_pct_of_host")
        gpu = resources.get("gpu_util_pct")
        vram = resources.get("gpu_memory_used_mb")
        fusion_compute = timing.get("Fusion Compute Interval")
        fusion_callback = timing.get("Fusion Total Callback")
        lidar_callback = timing.get("LiDAR Total Callback Cost")
        camera_inference = timing.get("Pure ONNX Inference")
        bev_render = timing.get("BEV Total Callback Cost")
        report_lines.append(
            "| {run} | {status} | {duration:.2f} | {sys_cpu} | {pipeline_cpu} | {gpu} | {vram} | {lidar_callback} | {camera_inference} | {fusion_compute} | {fusion_callback} | {bev_render} |".format(
                run=run["run"],
                status=run["status"],
                duration=run["playback_duration_s"] if run["playback_duration_s"] is not None else 0.0,
                sys_cpu=f"{sys_cpu['mean']:.1f}" if sys_cpu else "n/a",
                pipeline_cpu=f"{pipeline_cpu['mean']:.1f}" if pipeline_cpu else "n/a",
                gpu=f"{gpu['mean']:.1f} / {gpu['max']:.1f}" if gpu else "n/a",
                vram=f"{vram['max']:.0f}" if vram else "n/a",
                lidar_callback=f"{lidar_callback['mean']:.2f}" if lidar_callback else "n/a",
                camera_inference=f"{camera_inference['mean']:.2f}" if camera_inference else "n/a",
                fusion_compute=f"{fusion_compute['mean']:.2f}" if fusion_compute else "n/a",
                fusion_callback=f"{fusion_callback['mean']:.2f}" if fusion_callback else "n/a",
                bev_render=f"{bev_render['mean']:.2f}" if bev_render else "n/a",
            )
        )
    report_lines.extend(
        [
            "",
            "## Cross-run timing summary",
            f"Median of each completed run's mean across {len(completed_runs)} runs; this is an observational summary, not a confidence interval.",
            "",
            "| Metric | Runs with samples | Median run mean (ms) | Min run mean (ms) | Max run mean (ms) |",
            "| :--- | ---: | ---: | ---: | ---: |",
        ]
    )
    aggregate = summary["median_of_run_means"]
    for metric, stats in aggregate.items():
        if stats:
            report_lines.append(
                f"| {metric} | {stats['count']} | {stats['median']:.2f} | {stats['min']:.2f} | {stats['max']:.2f} |"
            )
    report_lines.extend(
        [
            "",
            "## Artifacts",
            "Each run directory contains `pipeline.log` and `resource_usage.csv`; `build.log` records the one-time build.",
            "",
        ]
    )
    (output_dir / "benchmark_summary.md").write_text("\n".join(report_lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build once, replay the KITTI pipeline repeatedly, and collect timing plus CPU/GPU utilization."
    )
    parser.add_argument("--runs", type=int, default=3, help="number of independent bag-playback runs (default: 3)")
    parser.add_argument("--bag-path", type=Path, default=ROOT / "2011_09_29_drive_0071_sync", help="ROS 2 bag directory")
    parser.add_argument("--play-rate", type=float, default=1.0, help="ros2 bag playback rate")
    parser.add_argument("--playback-delay", type=float, default=8.0, help="seconds before bag playback starts")
    parser.add_argument("--sample-interval", type=float, default=1.0, help="CPU/GPU resource sampling interval")
    parser.add_argument("--run-timeout", type=float, default=900.0, help="per-run wall timeout in seconds; 0 disables it")
    parser.add_argument("--output-dir", type=Path, help="new result directory (default: benchmark_results/<timestamp>)")
    parser.add_argument("--skip-build", action="store_true", help="reuse the current install instead of building once")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.sample_interval <= 0:
        parser.error("--sample-interval must be greater than 0")
    if args.play_rate <= 0:
        parser.error("--play-rate must be greater than 0")
    return args


def main() -> int:
    args = parse_args()
    bag_path = args.bag_path.expanduser().resolve()
    if not bag_path.exists():
        print(f"Bag path not found: {bag_path}", file=sys.stderr)
        return 2
    if not shutil.which("nvidia-smi"):
        print("nvidia-smi was not found; GPU utilization cannot be recorded.", file=sys.stderr)
        return 2
    if not (ROOT / "install" / "setup.bash").is_file():
        print("install/setup.bash is missing; run the workspace build first.", file=sys.stderr)
        return 2

    created = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = args.output_dir or (ROOT / "benchmark_results" / created)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    gpu_info = query_gpu(("name", "driver_version", "memory.total"))
    metadata = {
        "created_utc": utc_now(),
        "requested_runs": args.runs,
        "bag_path": str(bag_path),
        "bag_size_bytes": bag_size_bytes(bag_path),
        "play_rate": args.play_rate,
        "playback_delay_s": args.playback_delay,
        "sample_interval_s": args.sample_interval,
        "run_timeout_s": args.run_timeout,
        "build_performed": not args.skip_build,
        "system": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "ros_distro": os.environ.get("ROS_DISTRO"),
            "cpu_model": cpu_model(),
            "logical_cpu_count": psutil.cpu_count(),
            "memory_total_bytes": psutil.virtual_memory().total,
            "gpu_name": gpu_info[0],
            "gpu_driver": gpu_info[1],
            "gpu_memory_total_mb": optional_float(gpu_info[2]),
        },
        "git": git_metadata(),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    if not args.skip_build:
        print("Building workspace once before measured runs...")
        with (output_dir / "build.log").open("w", encoding="utf-8") as build_log:
            build = subprocess.run(
                ["colcon", "build", "--symlink-install"],
                cwd=ROOT,
                stdout=build_log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if build.returncode != 0:
            print(f"Build failed with exit code {build.returncode}; see {output_dir / 'build.log'}", file=sys.stderr)
            return build.returncode or 1

    runs: list[dict[str, Any]] = []
    interrupted = False
    for run_index in range(1, args.runs + 1):
        print(f"Starting benchmark run {run_index}/{args.runs}...")
        run_dir = output_dir / f"run_{run_index:02d}"
        try:
            result = run_one(
                run_index,
                run_dir,
                bag_path,
                args.play_rate,
                args.playback_delay,
                args.sample_interval,
                args.run_timeout,
            )
        except KeyboardInterrupt:
            interrupted = True
            break
        runs.append(result)
        write_outputs(output_dir, metadata, runs)
        playback_duration = result["playback_duration_s"]
        playback_label = f"{playback_duration:.1f}s playback" if playback_duration is not None else "playback not started"
        print(f"Run {run_index}: {result['status']} ({playback_label}, {result['launch_duration_s']:.1f}s launch total)")
        if result["status"] != "completed":
            interrupted = result["interrupted"]
            break

    write_outputs(output_dir, metadata, runs)
    print(f"Benchmark artifacts: {output_dir}")
    if interrupted:
        return 130
    return 0 if len(runs) == args.runs and all(run["status"] == "completed" for run in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
