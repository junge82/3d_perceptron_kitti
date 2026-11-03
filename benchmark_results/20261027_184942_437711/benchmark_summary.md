# KITTI Perception Benchmark

Created: 2026-10-27T17:49:42+00:00
Bag: `/home/gergo/work/3d_perceptron_kitti/2011_09_29_drive_0071_sync` (5,012,821,546 bytes)
Playback rate: 1.0x; startup delay: 8.0 s
Requested repetitions: 3
Git revision: `5ead6e7`; worktree dirty: True
Hardware: 12th Gen Intel(R) Core(TM) i5-12400; NVIDIA GeForce RTX 4060

Resource percentages are sampled once per configured interval. Pipeline CPU is normalized to total logical CPU capacity; GPU utilization is device-wide.

| Run | Status | Playback (s) | System CPU (%) | Pipeline CPU (%) | GPU mean / peak (%) | VRAM peak (MiB) | LiDAR callback (ms) | Camera inference (ms) | Fusion compute (ms) | Fusion callback (ms) | BEV render (ms) |
| ---: | :--- | ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | completed | 110.87 | 99.0 | 0.1 | 10.1 / 15.0 | 1422 | 31.45 | 4.92 | 2.49 | 3.21 | 2.57 |
| 2 | completed | 110.81 | 99.0 | 0.1 | 9.8 / 14.0 | 1454 | 32.97 | 4.89 | 2.69 | 3.46 | 2.85 |
| 3 | completed | 110.87 | 99.3 | 0.1 | 10.0 / 13.0 | 1435 | 31.14 | 4.79 | 2.67 | 3.46 | 2.78 |

## Cross-run timing summary
Median of each completed run's mean across 3 runs; this is an observational summary, not a confidence interval.

| Metric | Runs with samples | Median run mean (ms) | Min run mean (ms) | Max run mean (ms) |
| :--- | ---: | ---: | ---: | ---: |
| LiDAR Pre-Callback Age | 3 | 6.23 | 5.03 | 6.87 |
| Camera Pre-Callback Age | 3 | -5.79 | -5.87 | -5.76 |
| Numpy Byte Slice | 3 | 0.02 | 0.02 | 0.02 |
| NumPy ROI Filter | 3 | 1.61 | 1.58 | 1.62 |
| GPU VRAM Upload Cost | 3 | 0.32 | 0.32 | 0.33 |
| GPU CUDA RANSAC Cost | 3 | 2.62 | 2.49 | 2.69 |
| GPU CUDA DBSCAN Cost | 3 | 20.25 | 20.03 | 21.68 |
| ROS Message Loop | 3 | 3.70 | 3.60 | 3.74 |
| LiDAR Total Callback Cost | 3 | 31.45 | 31.14 | 32.97 |
| Camera Total Callback Cost | 3 | 11.31 | 11.16 | 11.37 |
| Pure ONNX Inference | 3 | 4.89 | 4.79 | 4.92 |
| Fusion Compute Interval | 3 | 2.67 | 2.49 | 2.69 |
| Fusion Total Callback | 3 | 3.46 | 3.21 | 3.46 |
| Fusion Timestamp Difference | 3 | 1.16 | 1.15 | 1.16 |
| BEV Pure Render Cost | 3 | 1.72 | 1.68 | 1.76 |
| BEV Total Callback Cost | 3 | 2.78 | 2.57 | 2.85 |

## Artifacts
Each run directory contains `pipeline.log` and `resource_usage.csv`; `build.log` records the one-time build.
