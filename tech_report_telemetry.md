# Telemetry Summary from Recorded Pipeline Session

This summary was derived by computing the arithmetic mean of each repeated timing entry recorded in `pipeline_session.log`. The values below describe the observed runtime behavior of the live ROS 2 pipeline during the recorded session and should be interpreted as sample means, not as a formal benchmark protocol.

## 1. Mean Latency by Pipeline Stage

| Metric | Sample Count | Mean | Minimum | Maximum |
| --- | ---: | ---: | ---: | ---: |
| Pre-Callback Age | 1969 | -0.07 ms | -14.02 ms | 294.79 ms |
| Numpy Byte Slice | 1059 | 0.02 ms | 0.01 ms | 0.14 ms |
| NumPy ROI Filter | 1059 | 1.64 ms | 1.02 ms | 7.71 ms |
| GPU VRAM Upload Cost | 1059 | 0.33 ms | 0.22 ms | 3.75 ms |
| GPU CUDA RANSAC Cost | 1059 | 2.50 ms | 1.12 ms | 14.40 ms |
| GPU CUDA DBSCAN Cost | 1059 | 22.07 ms | 9.61 ms | 163.14 ms |
| ROS Message Loop | 1059 | 4.33 ms | 1.23 ms | 26.59 ms |
| Total Callback Cost | 1969 | 25.22 ms | 7.21 ms | 175.80 ms |
| Pure ONNX Inference | 910 | 5.77 ms | 3.89 ms | 25.25 ms |
| Asynchronous BEV Render Complete. Cost | 106 | 2.93 ms | 1.26 ms | 14.83 ms |

## 2. Interpretation

The mean LiDAR callback cost in the recorded run was approximately 25.22 ms, with the dominant contributor being the GPU DBSCAN stage at 22.07 ms. The ROI filtering, RANSAC plane removal, and ROS message packaging stages were comparatively small, together contributing only a few milliseconds on average. This indicates that, for the recorded data and runtime configuration, clustering dominates the LiDAR callback time while the raw conversion and upload path remain lightweight.

The mean camera-side inference time was 5.77 ms, which is consistent with a small ONNX model running on a GPU-capable execution provider while the system alternates between inference and the rest of the ROS callback pipeline. The BEV renderer contributed a mean cost of 2.93 ms, indicating that the visualization stage remained lightweight relative to the LiDAR and sensor-fusion processing path.

## 3. Scope and Caveats

These means are sample averages from one recorded session and are not a substitute for a rigorous repeatable benchmark. They do not establish deterministic real-time guarantees, and they are sensitive to bag playback timing, runtime CPU/GPU load, ROS scheduling, and the exact Open3D/ONNX execution path used during the run. The figures should therefore be treated as observational telemetry rather than a production-grade performance certification.

## 4. Summary Statement

Across the recorded run, the LiDAR branch was the primary computational bottleneck, dominated by DBSCAN clustering, while camera inference and BEV rendering remained comparatively modest. The observed telemetry supports the conclusion that the system is operational and the timings are stable enough to observe a consistent pattern, but it does not replace a formal benchmark study with controlled repetitions and a fixed measurement protocol.
