# kitti_r1_perception

This package provides a modular 3D perception pipeline for the KITTI robot.

## Vision Messages Installation

To handle 2D and 3D computer vision bounding boxes, ensure `vision_msgs` is installed in your workspace:

```bash
git clone -b jazzy https://github.com/ros-perception/vision_msgs.git
```

## Nodes

### 1. `lidar_cluster_node`

Subscribes to raw point clouds, segments the ground plane, clusters the remaining foreground points, and generates 3D spatial clusters.

- **Subscribes**: 
    - `/kitti/velodyne/pointcloud` (`sensor_msgs/PointCloud2`)
- **Publishes**: 
    - `/detection/lidar_clusters` (`vision_msgs/Detection3DArray`) — Raw 3D spatial clusters.
    - `/lidar/points_no_ground` (`sensor_msgs/PointCloud2`) — Point cloud filtered from ground plane.
- **Parameters**:
    - `voxel_size` (0.15): Leaf size for voxel grid downsampling.
    - `ground_distance_threshold` (0.3): Distance threshold for RANSAC plane segmentation.
    - `ransac_n` (3): Number of points sampled to estimate RANSAC plane model parameters.
    - `ransac_iterations` (30): Maximum iterations for RANSAC plane estimation.
    - `cluster_tolerance` (0.65): Epsilon distance for Euclidean clustering.
    - `min_cluster_size` (20): Minimum points for a valid cluster.
    - `max_cluster_size` (1000): Maximum points for a cluster.
    - `bev_min_x` (1.5): Minimum X limit for filtering input cloud (meters).
    - `bev_max_x` (50.0): Maximum X limit for filtering input cloud (meters).
    - `bev_min_y` (-25.0): Minimum Y limit for filtering input cloud (meters).
    - `bev_max_y` (25.0): Maximum Y limit for filtering input cloud (meters).


### 2. `camera_detector_node`

Runs YOLOv8/11 object detection models on incoming left camera imagery to produce 2D semantic bounding boxes.

- **Subscribes**: 
    - `/kitti/camera_left/image_raw` (`sensor_msgs/Image`)
- **Publishes**: 
    - `/detection/camera_2d` (`vision_msgs/Detection2DArray`)
	- `/detection/camera_annotated` (`sensor_msgs.msg/Image`)
- **Parameters**:
    - `model_version` (default: `'yolov8n'`): Pre-trained model flavor to load ('yolov8n' or 'yolo11n').
    - `conf_threshold` (default: `0.25`): Confidence filtering threshold for model inference.
	- `iou_threshold` (default: `0.45`): 

### 3. `frustum_fusion_node`

Performs late fusion by projecting 3D LiDAR cluster centroids into the 2D image plane and matching them with camera-based semantic predictions.

- **Subscribes**: 
    - `/detection/lidar_clusters` (`vision_msgs/Detection3DArray`)
    - `/detection/camera_2d` (`vision_msgs/Detection2DArray`)
- **Publishes**: 
    - `/detection/fused_3d` (`vision_msgs/Detection3DArray`) — Synchronized, class-labeled 3D bounding boxes.
- **Parameters**:    
    - `match_radius` (default: `50.0`): Maximum search radius in pixels to pair a projected 3D cluster with a 2D bounding box center.

### 4. `bev_visualizer_node`

Generates a top-down Bird's-Eye-View (BEV) canvas mapping out filtered LiDAR points alongside fused, synchronized 3D bounding boxes.

- **Subscribes**: 
    - `/kitti/velodyne/pointcloud` (`sensor_msgs/PointCloud2`)
    - `/detection/fused_3d` (`vision_msgs/Detection3DArray`)
- **Publishes**: 
    - `/visualization/bev_image` (`sensor_msgs/Image`)
- **Parameters**:
    - `bev_min_x` (1.5): Visual footprint minimum limit in the forward X direction (meters).
    - `bev_max_x` (50.0): Visual footprint maximum limit in the forward X direction (meters). 
	- `bev_min_y` (-25.0): Visual footprint minimum limit in the lateral Y direction (meters).
    - `bev_max_y` (25.0): Visual footprint maximum limit in the lateral Y direction (meters).  
    - `range_y` (10.0): Visual footprint width in the lateral Y direction (meters).
    - `resolution` (default: `0.10`): Map granularity tracking meters per pixel.

## Execution

### ROS 2 Launch
To initiate the full pipeline along with the robot state publisher, run:
```bash
ros2 launch kitti_r1_perception myperception_launch.py
```

### Pixi Pipeline
Alternatively, you can orchestrate and run the complete task runner sequence through Pixi:
```bash
pixi run run-pipeline
```

> 💡 **RViz2 Visualization Tip:** For accurate tracking, clear visual synchronization, and zero semantic noise, make sure that your RViz `Detection3DArray` display topic is explicitly mapped to the fused output: `/detection/fused_3d` 

## Benchmark

- **Dataset Bag**: `2011_09_29_drive_0071_sync`
- **Execution Command**: `pixi run run-pipeline`
- **Telemetry Evaluation**: Triggered via `compute_pipeline_telemetry` during active pipeline sessions.
