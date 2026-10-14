#!/usr/bin/env python3
"""Compute telemetry statistics from a ROS pipeline session log.

Usage:
    python3 compute_pipeline_telemetry.py /path/to/pipeline_session.log
    python3 compute_pipeline_telemetry.py
"""

import re
import sys
import math
from pathlib import Path


PATTERNS = {
    "LiDAR Pre-Callback Age": r"^\[lidar_cluster_node-\d+\].*Pre-Callback Age\s*:\s*([+-]?[0-9.]+)ms",
    "Camera Pre-Callback Age": r"^\[camera_detector_node-\d+\].*Pre-Callback Age\s*:\s*([+-]?[0-9.]+)ms",
    "Pre-Callback Age": r"Pre-Callback Age\s*:\s*([+-]?[0-9.]+)ms",
    "Numpy Byte Slice": r"Numpy Byte Slice\s*:\s*([0-9.]+)ms",
    "NumPy ROI Filter": r"NumPy ROI Filter\s*:\s*([0-9.]+)ms",
    "GPU VRAM Upload Cost": r"GPU VRAM Upload Cost\s*:\s*([0-9.]+)ms",
    "GPU CUDA RANSAC Cost": r"GPU CUDA RANSAC Cost\s*:\s*([0-9.]+)ms",
    "GPU CUDA DBSCAN Cost": r"GPU CUDA DBSCAN Cost\s*:\s*([0-9.]+)ms",
    "ROS Message Loop": r"ROS Message Loop\s*:\s*([0-9.]+)ms",
    "LiDAR Total Callback Cost": r"^\[lidar_cluster_node-\d+\].*Total Callback Cost\s*:\s*([0-9.]+)ms",
    "Camera Total Callback Cost": r"^\[camera_detector_node-\d+\].*Total Callback Cost\s*:\s*([0-9.]+)ms",
    "Total Callback Cost": r"Total Callback Cost\s*:\s*([0-9.]+)ms",
    "Pure ONNX Inference": r"Pure ONNX Inference\s*:\s*([0-9.]+)ms",
    "GPU Processing Time": r"GPU Processing Time\s*:\s*([0-9.]+)ms",
    "Fusion Compute Interval": r"Compute interval\s*:\s*([0-9.]+)\s*ms",
    "Fusion Total Callback": r"Total callback\s*:\s*([0-9.]+)\s*ms",
    "Fusion Timestamp Difference": r"Timestamp difference\s*:\s*([0-9.]+)\s*ms",
    "Asynchronous BEV Render Complete. Cost": r"Asynchronous BEV Render Complete\. Cost\s*:\s*([0-9.]+)ms",
}


def parse_log(log_path: Path):
    metrics = {name: [] for name in PATTERNS}
    text = log_path.read_text(errors="replace")

    for line in text.splitlines():
        for name, pattern in PATTERNS.items():
            match = re.search(pattern, line)
            if match:
                metrics[name].append(float(match.group(1)))
                break

    return metrics


def summarize(values):
    if not values:
        return None
    ordered = sorted(values)

    def percentile(fraction):
        rank = (len(ordered) - 1) * fraction
        lower = math.floor(rank)
        upper = math.ceil(rank)
        if lower == upper:
            return ordered[lower]
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)

    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "median": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "min": min(values),
        "max": max(values),
    }


def main():
    log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("pipeline_session.log")

    if not log_path.exists():
        print(f"Log file not found: {log_path}", file=sys.stderr)
        print("Usage: python3 compute_pipeline_telemetry.py /path/to/pipeline_session.log", file=sys.stderr)
        raise SystemExit(1)

    metrics = parse_log(log_path)

    print(f"Telemetry summary for: {log_path}")
    print("=" * 80)
    print(f"{'Metric':<36} {'Count':>8} {'Mean':>9} {'Median':>9} {'P95':>9} {'P99':>9} {'Min':>9} {'Max':>9}")
    print("-" * 108)

    for name in PATTERNS:
        stats = summarize(metrics[name])
        if stats is None:
            continue
        print(
            f"{name:<36} "
            f"{stats['count']:>8} "
            f"{stats['mean']:>12.2f} "
            f"{stats['median']:>9.2f} "
            f"{stats['p95']:>9.2f} "
            f"{stats['p99']:>9.2f} "
            f"{stats['min']:>12.2f} "
            f"{stats['max']:>12.2f}"
        )


if __name__ == "__main__":
    main()
