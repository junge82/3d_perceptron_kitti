# KITTI Perception Benchmark

Created: 2026-10-12T16:56:31+00:00
Bag: `/home/gergo/work/3d_perceptron_kitti/2011_09_29_drive_0071_sync` (5,012,821,546 bytes)
Playback rate: 0.5x; startup delay: 8.0 s
Requested repetitions: 3
Git revision: `c5d53b6`; worktree dirty: True
Hardware: 12th Gen Intel(R) Core(TM) i5-12400; NVIDIA GeForce RTX 4060

Resource percentages are sampled once per configured interval. Pipeline CPU is normalized to total logical CPU capacity; GPU utilization is device-wide.

| Run | Status | Playback (s) | System CPU (%) | Pipeline CPU (%) | GPU mean / peak (%) | VRAM peak (MiB) | LiDAR callback (ms) | Camera inference (ms) | Fusion compute (ms) | Fusion callback (ms) | BEV render (ms) |
| ---: | :--- | ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | completed | 220.61 | 98.2 | 0.1 | 6.4 / 11.0 | 1348 | 30.89 | 5.17 | 2.35 | 3.16 | 2.89 |
| 2 | completed | 220.70 | 98.9 | 0.1 | 6.0 / 11.0 | 1489 | 31.70 | 5.29 | 4.28 | 5.06 | 2.91 |
| 3 | completed | 220.67 | 98.8 | 0.1 | 5.0 / 11.0 | 1517 | 31.67 | 5.29 | 2.31 | 3.10 | 2.61 |

## Cross-run timing summary
Median of each completed run's mean across 3 runs; this is an observational summary, not a confidence interval.

| Metric | Runs with samples | Median run mean (ms) | Min run mean (ms) | Max run mean (ms) |
| :--- | ---: | ---: | ---: | ---: |
| LiDAR Pre-Callback Age | 3 | 2.76 | 2.68 | 3.69 |
| Camera Pre-Callback Age | 3 | -2.88 | -2.99 | -2.87 |
| Numpy Byte Slice | 3 | 0.02 | 0.02 | 0.02 |
| NumPy ROI Filter | 3 | 1.58 | 1.58 | 1.59 |
| GPU VRAM Upload Cost | 3 | 0.31 | 0.30 | 0.33 |
| GPU CUDA RANSAC Cost | 3 | 2.33 | 2.33 | 2.42 |
| GPU CUDA DBSCAN Cost | 3 | 20.07 | 19.46 | 20.08 |
| ROS Message Loop | 3 | 3.74 | 3.61 | 3.77 |
| LiDAR Total Callback Cost | 3 | 31.67 | 30.89 | 31.70 |
| Camera Total Callback Cost | 3 | 11.76 | 11.54 | 11.94 |
| Pure ONNX Inference | 3 | 5.29 | 5.17 | 5.29 |
| Fusion Compute Interval | 3 | 2.35 | 2.31 | 4.28 |
| Fusion Total Callback | 3 | 3.16 | 3.10 | 5.06 |
| Fusion Timestamp Difference | 3 | 1.14 | 1.14 | 1.16 |
| Asynchronous BEV Render Complete. Cost | 3 | 2.89 | 2.61 | 2.91 |

## Artifacts
Each run directory contains `pipeline.log` and `resource_usage.csv`; `build.log` records the one-time build.
