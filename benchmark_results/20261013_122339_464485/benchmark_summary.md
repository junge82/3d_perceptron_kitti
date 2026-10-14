# KITTI Perception Benchmark

Created: 2026-10-13T10:23:39+00:00
Bag: `/home/gergo/work/3d_perceptron_kitti/2011_09_29_drive_0071_sync` (5,012,821,546 bytes)
Playback rate: 1.0x; startup delay: 8.0 s
Requested repetitions: 3
Git revision: `c5d53b6`; worktree dirty: True
Hardware: 12th Gen Intel(R) Core(TM) i5-12400; NVIDIA GeForce RTX 4060

Resource percentages are sampled once per configured interval. Pipeline CPU is normalized to total logical CPU capacity; GPU utilization is device-wide.

| Run | Status | Playback (s) | System CPU (%) | Pipeline CPU (%) | GPU mean / peak (%) | VRAM peak (MiB) | LiDAR callback (ms) | Camera inference (ms) | Fusion compute (ms) | Fusion callback (ms) | BEV render (ms) |
| ---: | :--- | ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | completed | 110.90 | 99.3 | 0.1 | 9.9 / 17.0 | 1568 | 33.06 | 5.07 | 2.74 | 3.51 | 2.59 |
| 2 | completed | 110.74 | 99.2 | 0.1 | 9.9 / 13.0 | 1540 | 33.40 | 5.01 | 2.67 | 3.46 | 2.58 |
| 3 | completed | 110.95 | 99.3 | 0.1 | 8.9 / 14.0 | 1577 | 32.98 | 5.00 | 2.77 | 3.54 | 2.85 |

## Cross-run timing summary
Median of each completed run's mean across 3 runs; this is an observational summary, not a confidence interval.

| Metric | Runs with samples | Median run mean (ms) | Min run mean (ms) | Max run mean (ms) |
| :--- | ---: | ---: | ---: | ---: |
| LiDAR Pre-Callback Age | 3 | 6.87 | 5.97 | 6.90 |
| Camera Pre-Callback Age | 3 | -5.28 | -6.35 | -5.25 |
| Numpy Byte Slice | 3 | 0.02 | 0.02 | 0.02 |
| NumPy ROI Filter | 3 | 1.64 | 1.63 | 1.64 |
| GPU VRAM Upload Cost | 3 | 0.35 | 0.34 | 0.36 |
| GPU CUDA RANSAC Cost | 3 | 3.23 | 3.19 | 3.30 |
| GPU CUDA DBSCAN Cost | 3 | 21.12 | 21.00 | 21.42 |
| ROS Message Loop | 3 | 3.73 | 3.65 | 3.77 |
| LiDAR Total Callback Cost | 3 | 33.06 | 32.98 | 33.40 |
| Camera Total Callback Cost | 3 | 12.76 | 12.43 | 12.94 |
| Pure ONNX Inference | 3 | 5.01 | 5.00 | 5.07 |
| Fusion Compute Interval | 3 | 2.74 | 2.67 | 2.77 |
| Fusion Total Callback | 3 | 3.51 | 3.46 | 3.54 |
| Fusion Timestamp Difference | 3 | 1.15 | 1.15 | 1.15 |
| Asynchronous BEV Render Complete. Cost | 3 | 2.59 | 2.58 | 2.85 |

## Artifacts
Each run directory contains `pipeline.log` and `resource_usage.csv`; `build.log` records the one-time build.
