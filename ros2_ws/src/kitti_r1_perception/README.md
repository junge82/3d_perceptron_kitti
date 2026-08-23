# kitti_r1_perception

This package provides a modular 3D perception pipeline for the KITTI robot.

# Vision Messages 

git clone -b jazzy https://github.com/ros-perception/vision_msgs.git

## Nodes

### 1. `lidar_cluster_node`

Subscribes to point clouds, segments the ground plane, clusters the remaining points, and publishes 3D bounding boxes.

- **Topic**: `/kitti/velodyne/pointcloud` (Input: `sensor_msgs/PointCloud2`)
- **Publishes**: 
    - `/detection/lidar_clusters` (`vision_msgs/Detection3DArray`)
    - `/lidar/points_no_ground` (`sensor_msgs/PointCloud2`)
- **Parameters**:
    - `voxel_size` (0.1): Leaf size for voxel grid downsampling.
    - `ground_distance_threshold` (0.2): Distance threshold for RANSAC plane segmentation.
    - `cluster_tolerance` (0.5): Epsilon distance for Euclidean clustering.
    - `min_cluster_size` (50): Minimum points for a valid cluster.
    - `max_cluster_size` (10000): Maximum points for a cluster.

### 2. `camera_detector_node`

Runs YOLOv8/11 object detection on camera images.

- **Topic**: `/kitti/camera_left/image_raw` (Input: `sensor_msgs/Image`)
- **Publishes**: `/detection/camera_2d` (`vision_msgs/Detection2DArray`)
- **Parameters**:
    - `model_version` ('yolov8n'): Model to use ('yolov8n' or 'yolo11n').
    - `conf_threshold` (0.25): Confidence threshold for detections.

### 3. `frustum_fusion_node`

Fuses 3D lidar clusters with 2D camera detections by projecting 3D centroids onto the image plane.

- **Subscribes**: 
    - `/detection/lidar_clusters` (`vision_msgs/Detection3DArray`)
    - `/detection/camera_2d` (`vision_msgs/Detection2DArray`)
    - `/zed2/zed/rgb/camera_info` (`sensor_msgs/CameraInfo`)
- **Publishes**: `/detection/fused_3d` (`vision_msgs/Detection3DArray`)
- **Parameters**:
    - `lidar_frame` ('velodyne'): Source frame of the lidar data.
    - `camera_frame` ('zed2_left_camera_optical_frame'): Target camera optical frame.
    - `match_radius` (50.0): Pixel radius to match a projected 3D point to a 2D bounding box center.

### 4. `bev_visualizer_node`

Generates a top-down Bird's-Eye-View image visualizing the lidar points and fused detections.

- **Subscribes**: 
    - `/kitti/velodyne/pointcloud` (`sensor_msgs/PointCloud2`)
    - `/detection/fused_3d` (`vision_msgs/Detection3DArray`)
- **Publishes**: `/visualization/bev_image` (`sensor_msgs/Image`)
- **Parameters**:
    - `range_x` (10.0): View range in X direction (meters).
    - `range_y` (10.0): View range in Y direction (meters).
    - `resolution` (0.05): Meters per pixel.

## Launch

Use the provided launch file to start all nodes and the robot state publisher:

```bash
ros2 launch kitti_r1_perception perception_launch.py
```

pixi run run-pipeline

[Raw PointCloud Data] 
       │  (Published on /kitti/velodyne/pointcloud with frame_id: "velodyne")
       ▼
   [ RViz2 ] ◄─── (Queries the /tf topic for the relation between "velodyne" and "map")
       │
       ├───► [ Your tf_velodyne Node ] provides the 180° Yaw-flip matrix
       ▼
[Transformed Points] ──► Rendered accurately on the screen relative to the "map" grid


You need to add tf_camera because your RViz layout now includes an annotated camera image display (/detection/camera_annotated).

In the KITTI dataset, the 3D bounding boxes from your /detection/lidar_clusters topic are calculated using data from both the camera and the LiDAR.
For RViz to accurately project and match those 3D bounding boxes over the objects in the 2D image, it must know the exact physical distance and angle between the camera lens and the LiDAR sensor.

Without tf_camera, you will run into the following problems:

1. The Image Panel Will Throw Errors Your RViz Fixed Frame is set to map. 
   If the image topic sends data labeled with frame_id: camera,
   RViz will check the /tf topic to find out where the camera is located relative to the map.
   If it finds nothing, the Image panel will display a red error: "No transform from [camera] to [map]"

2.  3D Bounding Boxes Won't Align with the Image
If your detection node is projecting 3D clusters onto the camera image (or vice versa),
it requires a spatial link. 
Without tf_camera, the system cannot translate the coordinate points from the velodyne space to the camera space,
causing your 3D bounding boxes to float in empty space or misalign with the cars/pedestrians in the video.

How the Full Sensor Rig Links 

Together By adding both nodes, you create a complete coordinate tree inside ROS 2 Jazzy that looks like this:
          ┌─────────┐
          │   map   │  <-- (The absolute center of your RViz grid)
          └───┬─┬───┘
              │ │
    ┌─────────┘ └─────────┐
    │ tf_velodyne         │ tf_camera
    ▼                     ▼
┌──────────┐          ┌──────────┐
│ velodyne │          │  camera  │  <-- (Tells RViz where the lens is)
└──────────┘          └──────────┘



## Using quaternions
 Using quaternions instead of Euler angles is preferred in ROS 2 Jazzy and modern robotics for two main reasons: it avoids a severe command-line parsing bug in ROS 2 and it prevents a mathematical lockup known as Gimbal Lock.
 
 Here is a breakdown of why quaternions are the standard choice:
 1. The ROS 2 Command-Line Parsing Bug (The Immediate Issue)
   As you experienced firsthand with your console logs earlier, the ROS 2 Humble, Iron, and Jazzy implementations of static_transform_publisher contain an argument parsing quirk.
   
   When you pass --roll, --pitch, and --yaw parameters, the underlying C++ code relies on specific ordering or library interpretations that frequently drop the values entirely, falling back to an identity matrix (0 rotation).
   
   Quaternions (--qx, --qy, --qz, --qw) map directly to the internal math structures of the geometry_msgs/msg/TransformStamped message format. There is no conversion step required, making them highly robust and bug-free when parsed by the terminal action or launch system.
 
 2. Eliminating Gimbal Lock (The Mathematical Reason)Euler angles calculate orientation by applying three sequential rotations (e.g., first Roll, then Pitch, then Yaw).
   The Problem: If you pitch a sensor perfectly straight up or down (by 90 degrees), two of your rotation axes align and overlap.
   
   The Result: You lose an entire degree of freedom. If you try to rotate the sensor horizontally after that, the math breaks down because the system can no longer distinguish between a Roll change and a Yaw change. This mathematical lockup is called Gimbal Lock.
   
   The Quaternion Fix: Quaternions represent a 3D rotation as a single vector (axis) and a single angle of rotation around that axis using 4 numbers. Because they do not rely on a sequence of steps, Gimbal Lock is mathematically impossible.
   
 3. Computational Efficiency
   When a node like RViz2 or a localization pipeline calculates thousands of point cloud transforms per second, it has to convert coordinates constantly.
   
   To combine multiple Euler angles, a computer must compute expensive trigonometric functions (sines and cosines).
   
   Quaternions combine multiple rotations using simple matrix multiplication (pure addition and multiplication). This makes them significantly faster and lighter on computer memory.

 Summary                             Comparison
 Feature                Euler Angles (Roll/Pitch/Yaw)   Quaternions (qx/qy/qz/qw)
 Human Readability      Intuitive (e.g., 90° flip)      Hard to read mentally
 Gimbal Lock Risk       Yes (Math breaks at 90° pitch)  None (Perfect 3D tracking)
 ROS 2 Parser Stability Prone to argument-dropping bugs 100% stable and native
 Processing Speed       Slow (Trigonometry heavy)       Blazing fast (Matrix math)
 
 Do you need help calculating a specific custom rotation into quaternion format for your launch file, or are your sensor alignments perfectly locked in?