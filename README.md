# KITTI 3D Perception Pipeline

A ROS 2 Jazzy application that combines KITTI camera and LiDAR data for object detection, 3D fusion, and bird's-eye-view (BEV) visualization.

## What It Does

- **LiDAR:** removes the ground plane and clusters point clouds into 3D detections.
- **Camera:** runs YOLO object detection on the left KITTI camera stream using ONNX Runtime.
- **Fusion:** associates camera detections with projected LiDAR cluster centers.
- **Visualization:** displays the robot and perception output in RViz and publishes a BEV image.

## Requirements

- [Pixi](https://pixi.sh/latest/)
- Linux x86-64 with an NVIDIA GPU and a compatible driver for the configured CUDA 12.9 workflow
- A KITTI ROS 2 bag. The default launch expects `2011_09_29_drive_0071_sync/` in the repository root.

The Pixi environment supplies ROS 2 Jazzy and the Python dependencies. Review the KITTI dataset terms before downloading or redistributing data.

## Run

From the repository root:

```bash
pixi install
pixi run run-pipeline
```

The task checks the GPU environment, builds the ROS workspace, starts the perception nodes and RViz, then plays the default bag with the ROS simulation clock. Logs are written to `pipeline_session.log`.

To play another ROS 2 bag, build and pass its directory to the main launch file:

```bash
pixi run colcon build --symlink-install
pixi run bash -c 'source install/setup.bash && ros2 launch kitti_r1_perception myperception.launch.py bag_path:=/absolute/path/to/bag'
```

The launch also accepts `play_rate`, `playback_delay`, and LiDAR/BEV tuning arguments. See [`myperception.launch.py`](ros2_ws/src/kitti_r1_perception/launch/myperception.launch.py) for defaults.

## Topics

The bag provides `/kitti/camera_left/image_raw`, `/kitti/velodyne/pointcloud`, camera calibration, and transforms. The perception nodes publish:

- `/detection/camera_2d`
- `/detection/lidar_clusters` and `/lidar/points_no_ground`
- `/detection/fused_3d`
- `/visualization/bev_image`

### 📊 Known Limitations & Error Analysis (Edge Cases)
- **Class Ambiguity (e.g., Minibus vs. Motorcycle):** Since the 3D bounding box estimation relies on geometric frustum fusion and DBSCAN clustering, spatial occlusion or tightly packed objects can cause cluster overlapping. If a motorcycle is adjacent to a larger vehicle, the spatial heuristics might misclassify the fused cluster.
- **Future Improvements:** Implementing temporal tracking (e.g., Extended Kalman Filter / SORT) to maintain class consistency across consecutive frames.


**Calibration note:** fusion currently uses hard-coded camera intrinsics and LiDAR-to-camera extrinsics; See the [technical report](TECHNICAL_REPORT.md) for architecture, node interfaces, and limitations.

## More Information

- [Developer guide](DEVELOPMENT.md)
- [Perception package guide](ros2_ws/src/kitti_r1_perception/README.md)
- [KITTI-to-MCAP converter](mcap_converter/README.md)
- [Sensor-frame alignment and calibration work](CompVisionKnowHow.md)
- [Benchmark protocol plan](benchmark_action_plan.md)
