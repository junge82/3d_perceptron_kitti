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

# Pre-compiled regex minták a maximális feldolgozási sebesség és pontosság érdekében
PATTERNS = {
    "LiDAR Pre-Callback Age": re.compile(r"^\[lidar_cluster_node-\d+\].*Pre-Callback Age\s*:\s*([+-]?[0-9.]+)ms"),
    "Camera Pre-Callback Age": re.compile(r"^\[camera_detector_node-\d+\].*Pre-Callback Age\s*:\s*([+-]?[0-9.]+)ms"),
    "Numpy Byte Slice": re.compile(r"Numpy Byte Slice\s*:\s*([0-9.]+)ms"),
    "NumPy ROI Filter": re.compile(r"NumPy ROI Filter\s*:\s*([0-9.]+)ms"),
    "GPU VRAM Upload Cost": re.compile(r"GPU VRAM Upload Cost\s*:\s*([0-9.]+)ms"),
    "GPU CUDA RANSAC Cost": re.compile(r"GPU CUDA RANSAC Cost\s*:\s*([0-9.]+)ms"),
    "GPU CUDA DBSCAN Cost": re.compile(r"GPU CUDA DBSCAN Cost\s*:\s*([0-9.]+)ms"),
    "ROS Message Loop": re.compile(r"ROS Message Loop\s*:\s*([0-9.]+)ms"),
    "LiDAR Total Callback Cost": re.compile(r"^\[lidar_cluster_node-\d+\].*Total Callback Cost\s*:\s*([0-9.]+)ms"),
    "Camera Total Callback Cost": re.compile(r"^\[camera_detector_node-\d+\].*Total Callback Cost\s*:\s*([0-9.]+)ms"),
    "Pure ONNX Inference": re.compile(r"Pure ONNX Inference\s*:\s*([0-9.]+)ms"),
    "GPU Processing Time": re.compile(r"GPU Processing Time\s*:\s*([0-9.]+)ms"),
    "Fusion Compute Interval": re.compile(r"Compute interval\s*:\s*([0-9.]+)\s*ms"),
    "Fusion Total Callback": re.compile(r"Total callback\s*:\s*([0-9.]+)\s*ms"),
    "Fusion Timestamp Difference": re.compile(r"Timestamp difference\s*:\s*([0-9.]+)\s*ms"),
    # JAVÍTVA: Az új, moduláris BEV logolási struktúrához igazított kétcsatornás minták
    "BEV Pure Render Cost": re.compile(r"Asynchronous BEV Render Complete\. Render\s*:\s*([0-9.]+)ms"),
    "BEV Total Callback Cost": re.compile(r"Total Pipeline Callback\s*:\s*([0-9.]+)ms"),
}


def parse_log(log_path: Path):
    metrics = {name: [] for name in PATTERNS}
    
    # Biztonságos, memóriahatékony soronkénti beolvasás
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            for name, pattern in PATTERNS.items():
                match = pattern.search(line)
                if match:
                    metrics[name].append(float(match.group(1)))
                    # ⚡ JAVÍTVA: Kivettük a break-et! Így ha egy sor illeszkedik a generikus 
                    # és a specifikus (nódus-névvel ellátott) mintára is, mindkét helyre bekerül az adat.

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

    print(f"\nTelemetry summary for: {log_path}")
    print("=" * 116)
    print(f"{'Metric Name':<38} {'Count':>6} {'Mean':>10} {'Median':>9} {'P95':>9} {'P99':>9} {'Min':>10} {'Max':>10}")
    print("-" * 116)

    for name in PATTERNS:
        stats = summarize(metrics[name])
        if stats is None:
            continue
        print(
            f"{name:<38} "
            f"{stats['count']:>6} "
            f"{stats['mean']:>10.2f} "
            f"{stats['median']:>9.2f} "
            f"{stats['p95']:>9.2f} "
            f"{stats['p99']:>9.2f} "
            f"{stats['min']:>10.2f} "
            f"{stats['max']:>10.2f}"
        )
    print("=" * 116)


if __name__ == "__main__":
    main()
