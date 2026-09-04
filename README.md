

# KITTI 3D Perception Pipeline
This is the accompanying repository for the '𝗠𝗼𝘃𝗶𝗻𝗴 𝗳𝗿𝗼𝗺 𝟮𝗗 𝘁𝗼 𝟯𝗗: 𝗕𝘂𝗶𝗹𝗱𝗶𝗻𝗴 𝗮 𝗟𝗶𝗗𝗔𝗥-𝗖𝗮𝗺𝗲𝗿𝗮 𝗙𝘂𝘀𝗶𝗼𝗻 𝗣𝗶𝗽𝗲𝗹𝗶𝗻𝗲 𝗶𝗻 𝗥𝗢𝗦 𝟮' 

The workspace contains a ROS 2 Jazzy 3D perception pipeline developed for the KITTI robot.
The system processes Lidar point clouds and camera images to provide fused 3D object detections and a Bird's-Eye-View (BEV) visualization.

- '/kitti/velodyne/pointcloud',

## Prerequisites

- Ubuntu 22.04 
- ROS 2 Jazzy Jalisco
- Pixi (Package Manager)

## Installation

1. Clone the repository (if applicable) or navigate to the workspace root.
2. Initialize the environment using Pixi:
   ```bash
   pixi install
   ```

## Build

Build the workspace using `colcon`:
```bash
pixi run build
source install/setup.bash
```

## Running the Pipeline

To launch the full perception pipeline:
```bash
ros2 launch kitti_r1_perception perception_launch.py
```
pixi run run-pipeline

### With ROS Bag
Download bag here: https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0001/2011_09_26_drive_0001_sync.zip

Since this project is designed to run with recorded data, ensure you play your ROS bag with the `--clock` argument:
```bash
ros2 bag play <your_bag_file> --clock
```

## Packages

- **kitti_r1_description**: Contains the URDF and meshes 
- **kitti_r1_description_r1_perception**: Implements the perception nodes (Lidar clustering, Camera detection, Frustum fusion, BEV visualization).

- velodyne_viewer.py
- kitti_calib_projection.py

## Pixi Comands

pixi shell

pixi install / pixi update

pixi run

pixi run test-gpu

changelog: pixi upgrade to open3d support

# Kitti to ROS Bag converter [MCAP]

./kitti_2_rosbag_converter.py

parse_kiti_timestamps
create_fields
convert_bin_to_pointcloud
parse_oxts_to_messages
create_kitti_raw_bag

 - camera_left    -
 - camera_right   -
 - pointcloud     -  
 - oxts/fix       -
 - oxts/twist     -



# Title: Coordinate systems
https://mmdetection3d.readthedocs.io/en/v1.0.0rc1/tutorials/coord_sys_tutorial.html

# Kitti sources: 
- https://colab.research.google.com/github/itberrios/CV_tracking/blob/main/kitti_tracker/1_kitti_object_detection_lidar.ipynb#scrollTo=YgHXWJHwpU4U
- https://www.cvlibs.net/publications/Geiger2013IJRR.pdf



### 4.2. Comparative Profiling: Unoptimized Bottlenecks vs. Accelerated GPU Engine

The comparative profiling trace confirms that transitioning from unvectorized iteration blocks to direct memory maps completely eliminates implicit processing overhead and PCIe pipeline stalls:

```text
[🐌 UNOPTIMIZED BASELINE TOPOLOGY]
Raw Buffer ──► Generator Iteration [143ms] ──► Fragmented Upload [38ms] ──► Nested Viz Loops [191ms] ──► 224.70ms Latency

[⚡ ACCELERATED HARDWARE TOPOLOGY]
Raw Buffer ──────────────────────► Zero-Copy Strided View [0.02ms] ──────────────────────► Flat 1D Masking [1.13ms] ──► 22.50ms Latency
```

#### Final Real-Time System Telemetry Trace (Locked at 10 Hz Dataset Target)
```text
[camera_detector_node-2] ONNX Inference: 3.91ms | Zero-Copy IOBinding View | Total Callback Cost: 7.17ms 
[lidar_cluster_node-1]   Numpy Slice: 0.02ms | GPU VRAM Upload: 0.24ms | GPU CUDA DBSCAN Cluster: 13.19ms
[frustum_fusion_node-3]  Parallel Matrix Broadcasting | GPU Processing: 0.70 ms | Successfully Fused: 2
[bev_visualizer_node-4]  ⚡ Asynchronous BEV Render Complete. Pipeline Speed: 10.0 FPS | Canvas Drawing Cost: 1.13ms
```


### What I Improved

Custom CUDA kernel



### Benchmark


### Known Limitations & Edge Cases

 - A model bizonyos távolságokban hajlamos összekeverni a minibuszokat a motorbiciklikkel
 - 


3d pipeline

RANSAC + DBSCAN (LIDAR): 
   RANSAC: ground plane removal
   DBSCAN: clustering the objects

ONNX YOLO (CAMERA): Yolo model exported to ONNX format, highly optimizationed performance

Approximate Time Policy / Message Filter: LIDAR and camera signal synchronized in time


### 🛠️ Architecture & Pipeline
This project implements a highly efficient, hybrid 3D Sensor Fusion pipeline in ROS2:
1. **LiDAR Pipeline:** Ground plane removal using **RANSAC**, followed by 3D object clustering via **DBSCAN**.
2. **Camera Pipeline:** 2D object detection using an optimized **ONNX YOLO** model for low-latency inference.
3. **Synchronization:** Uses ROS2 **Message Filters (ApproximateTime Policy)** to seamlessly sync LiDAR point clouds and camera frames.
4. **Fusion:** Projects 2D ROI (Region of Interest) into 3D space to filter and fuse LiDAR clusters within the camera frustum.

### 📊 Known Limitations & Error Analysis (Edge Cases)
- **Class Ambiguity (e.g., Minibus vs. Motorcycle):** Since the 3D bounding box estimation relies on geometric frustum fusion and DBSCAN clustering, spatial occlusion or tightly packed objects can cause cluster overlapping. If a motorcycle is adjacent to a larger vehicle, the spatial heuristics might misclassify the fused cluster.
- **Future Improvements:** Implementing temporal tracking (e.g., Extended Kalman Filter / SORT) to maintain class consistency across consecutive frames.

