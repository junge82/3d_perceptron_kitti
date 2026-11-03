# KITTI 3D Perception Pipeline

A production-ready ROS 2 Jazzy application that integrates camera and LiDAR data using 2D-projected geometric Frustum Fusion, low-latency deep learning 2D inference, and rigid calibration-matrix projections.

## 👁️ Pipeline Demonstration

![3D Perception Pipeline Processing](video/kimenet.gif)

---

## 🛠️ Tech Stack & Key Features

- **Robotics Framework:** `ROS 2 Jazzy` utilizing native node composition, simulation clock synchronization, and optimized execution loops.
- **Deep Learning Inference:** `ONNX Runtime` utilizing CUDA execution providers and explicit I/O binding for real-time 2D object detection on the left KITTI camera stream.
- **Point Cloud Processing:** `Open3D` and `NumPy` for ground plane removal (RANSAC), Spatial ROI filtering, and GPU-accelerated DBSCAN clustering.
- **Sensor Fusion Architecture:** Vectorized parallel matrix broadcasting using `PyTorch` CUDA tensors to match 2D camera detections with projected 3D LiDAR cluster centroids via factory-calibrated rigid matrices.
- **Determinism & Reproducibility:** 100% reproducible environments powered by `Pixi` (RoboStack conda ecosystem) ensuring zero-system-dependency deployment.

---

## 📐 System Architecture

```mermaid
graph TD
    Bag([KITTI ROS 2 Bag]) -->|/kitti/camera_left/image_raw| CamNode[Camera Inference Node: ONNX]
    Bag -->|/kitti/velodyne/pointcloud| LidarNode[LiDAR Processing Node: Open3D]
    
    CamNode -->|/detection/camera_2d| FusionNode[Projected Frustum Fusion Node]
    LidarNode -->|/detection/lidar_clusters| FusionNode
    
    FusionNode -->|/detection/fused_3d| RViz[RViz2 Visualization]
    FusionNode -->|/visualization/bev_image| BEV[BEV Stream]
```

The data flow spans across four decoupled nodes. Camera and LiDAR detections converge at the fusion layer; the Bird's-Eye-View (BEV) visualizer visualizes the latest processed outputs asynchronously.

---

## 📊 Performance & Benchmarking

The end-to-end pipeline has been rigorously profiled and automated across multiple consecutive evaluation runs to guarantee deterministic execution metrics.

### Hardware Environment
- **CPU:** 12th Gen Intel(R) Core(TM) i5-12400 (6 Cores / 12 Threads)
- **GPU:** NVIDIA GeForce RTX 4060
- **Dataset / Workload:** KITTI Sync Bag (`2011_09_29_drive_0071_sync`, ~5.0 GB)

### End-to-End Execution Profile
Empirical metrics recorded across 3 consecutive automated evaluation runs at standard 1.0x native playback frequency (10 Hz). 

*Note: Latency values represent the median of each completed run's mean execution time (not the p50 value of the raw frame-by-frame distribution).*

| Pipeline Component | Underlying Operations / Framework | Performance Metric (Median of Means) | Target Bound |
| :--- | :--- | :--- | :--- |
| **LiDAR Processing** | NumPy ROI Filter + **Open3D GPU CUDA RANSAC & DBSCAN** | **31.45 ms** (Total Callback) | < 100.0 ms (10Hz) |
| **Camera Inference** | **ONNX Runtime (CUDA Execution Provider)** | **4.89 ms** (Pure Inference) | < 33.3 ms (30Hz) |
| **Geometric Fusion** | Association & Extrinsic PyTorch Broadcasting | **2.67 ms** (Compute Interval) | < 10.0 ms |
| **BEV Visualization** | Asynchronous Bird's-Eye-View Canvas Render | **2.78 ms** (Total Callback) | < 20.0 ms |

### Key Bottleneck Analysis
- **GPU CUDA DBSCAN (3D Spatial Clustering):** **20.25 ms** (median run mean). Represents the primary processing loop footprint, showing an exceptionally tight variance across the entire evaluation track.
- **Total Camera Callback Overhead:** **11.31 ms**. While pure ONNX CUDA inference requires only 4.89 ms, the remaining overhead is bound to CPU-side image preprocessing, layout restructuring to NCHW, GPU-CPU memory transfer, and OpenCV Non-Maximum Suppression (NMS).
- **Total Fusion Callback:** **3.46 ms**. Maintained via an optimized `ApproximateTimeSynchronizer` queue layout.

### Resource Utilization Summary
- **Memory Footprint:** Peak VRAM consumption stabilized at **~1435 MiB**.
- **Hardware Load:** Device-wide GPU utilization maintained a lean **10.0% mean / 15.0% peak**, leaving substantial headroom for additional tracking and control modules.

---

## ⚙️ Requirements & Environment Setup

- [Pixi](https://pixi.sh) (Modern package management tool)
- Linux x86-64 with an NVIDIA GPU and compatible drivers (CUDA 12.9 workflow).
- A KITTI ROS 2 bag. The default launch expects `2011_09_29_drive_0071_sync/` in the repository root.

The Pixi environment fully wraps ROS 2 Jazzy and all required system/Python dependencies cleanly in user space, ensuring cross-machine reproducibility.

---

## 🚀 Execution & Run Guide

From the repository root, install dependencies and execute the launch system using a single toolchain:

```bash
# Install and cache dependencies in an isolated workspace
pixi install

# Run the complete end-to-end perception pipeline with RViz visualization
pixi run run-pipeline
```

The automation task verifies your local GPU architecture, builds the isolated ROS workspace, brings up all pipeline perception nodes alongside RViz2, and plays back the default data bag using the simulation clock. Logs are preserved in `pipeline_session.log`.

### Customizing Playback & Parameters

To build manually and run against custom bags or change runtime parameters:

```bash
# Explicit workspace compilation
pixi run colcon build --symlink-install

# Execution with custom ROS 2 parameters
pixi run bash -c 'source install/setup.bash && ros2 launch kitti_r1_perception myperception.launch.py bag_path:=/absolute/path/to/bag'
```

*Note: The main launch file accepts tuning parameters such as `play_rate`, `playback_delay`, and specific clustering thresholds. See [`myperception.launch.py`](ros2_ws/src/kitti_r1_perception/launch/myperception.launch.py) for the complete interface configuration.*

---

## 📡 ROS 2 Topic Interface

### Subscribed Topics (Inputs)
- `/kitti/camera_left/image_raw` (`sensor_msgs/msg/Image`)
- `/kitti/velodyne/pointcloud` (`sensor_msgs/msg/PointCloud2`)
*Note: The pipeline bypasses the ROS 2 TF framework entirely to eliminate CPU-bound transformation lookup latencies, utilizing parameterized rigid calibration matrices directly within GPU VRAM.*

### Published Topics (Outputs)
- `/detection/camera_2d` — 2D Bounding Boxes with class probabilities.
- `/detection/lidar_clusters` — Unassociated spatial clusters.
- `/lidar/points_no_ground` — Downsampled, filtered raw point cloud.
- `/detection/fused_3d` — **Fused 3D Bounding Boxes combining LiDAR spatial geometry with YOLO class probabilities.**
- `/visualization/bev_image` — Real-time Bird's-Eye-View projection canvas.

---

## 🔬 Limitations & Error Analysis (Edge Cases)

- **Spatial & Class Ambiguity (e.g., Minibus vs. Motorcycle):** Since 3D bounding box estimation relies on geometric frustum association and DBSCAN clustering, extreme spatial occlusion or tightly packed entities can cause cluster bleeding. For instance, a motorcycle adjacent to a larger vehicle might temporarily combine into a single spatial heuristic cluster.
- **Absence of Temporal Tracking & Heading:** The current fusion node operates purely on a frame-by-frame basis without temporal tracking (e.g., Kalman Filters). Consequently, object orientation (yaw headings) and velocity vectors are not yet estimated or accumulated across time.
- **Temporal Ghosting on BEV Cache:** Because the Fusion node conditionally omits publishing during empty cycles to reduce stream overhead, the asynchronous BEV visualizer retains the last valid detection pointer. In dynamic ego-motion scenarios, this can cause transient "ghost tracks" until the cache is explicitly overwritten by a new synchronized camera-LiDAR pair.

### Future Roadmap
- [ ] Integrating a multi-object temporal tracker (Extended Kalman Filter / SORT) to compute object heading, velocity vectors, and handle transient occlusions.
- [ ] Implementing dynamic Odometry Compensation using KITTI OXTS (GPS/IMU) telemetry data to establish stable global coordinate tracking.
- [ ] Transitioning the core Fusion node to native ROS 2 Lifecycle C++ Components to achieve zero-copy data transport.

---

## 📖 Extended Documentation

- [Technical Deep-Dive & Architecture Report](TECHNICAL_REPORT.md)
- [Developer & Contribution Guide](DEVELOPMENT.md)
- [Perception Package Implementation Details](ros2_ws/src/kitti_r1_perception/README.md)
- [High-Throughput KITTI-to-MCAP Converter](mcap_converter/README.md)
- [Sensor Frame Calibration & Mathematics Work](CompVisionKnowHow.md)

