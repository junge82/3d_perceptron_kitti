# Sensor Frame Alignment in ROS 2

## Introduction

A multi-sensor perception system is only as reliable as the coordinate frames behind it. In this setup, the LiDAR point cloud and the camera image must be expressed in the same reference frame before they can be interpreted correctly by RViz or by the perception pipeline.

When RViz uses `map` as the fixed frame, every sensor frame must be connected to that world frame through the TF tree. If the transform chain is incomplete or inconsistent, the data will appear visually misaligned even when the underlying detection logic is valid.

---

## LiDAR-to-map alignment

The LiDAR point cloud must be transformed into the same coordinate frame used by RViz as its fixed reference.

```text
[Raw PointCloud Data]
       │  Published on /kitti/velodyne/pointcloud
       │  frame_id: "velodyne"
       ▼
[RViz2]
       │  Queries /tf to resolve the transform from "velodyne" to "map"
       ▼
[tf_velodyne node]
       │  Provides the LiDAR-to-map transform
       ▼
[Transformed Points]
       └── Rendered correctly in the "map" frame
```

A correct `tf_velodyne` transform is therefore required before the LiDAR cloud can be displayed meaningfully in RViz.

---

## Why `tf_camera` is required

Once the camera is included in the visualization, the camera frame must also be connected to the same global reference frame used by RViz.

The exact transform needed is the pose of the camera expressed in the `map` frame:

```text
T_map_camera = [x, y, z, qx, qy, qz, qw]
```

This transform has two components:

- Translation `(x, y, z)`: the camera's physical offset in the `map` frame.
- Rotation `(qx, qy, qz, qw)`: the camera orientation relative to `map`.

In ROS TF terms, this is the transform from `map` to `camera` (equivalently, the inverse from `camera` to `map` when required by RViz or other nodes).

Without `tf_camera`, two issues arise:

1. RViz cannot resolve the camera frame in the fixed frame.
   - If RViz is fixed to `map` and the image uses `frame_id: camera`, it must resolve a transform from `camera` to `map`.
   - If no transform exists, RViz reports:
     > No transform from [camera] to [map]

2. LiDAR and camera data cannot be related spatially.
   - Even when the LiDAR cloud is already aligned to `map`, the camera image remains disconnected unless the camera pose is known in the same frame.
   - Without `tf_camera`, the image, LiDAR cloud, and projected annotations cannot be interpreted as belonging to the same physical scene.

In the current launch file, this transform is created with `static_transform_publisher` as:

```text
--frame-id map --child-frame-id camera
--x 0 --y 0 --z 0
--qx 0 --qy 0 --qz 0 --qw 1
```

This is an identity pose: zero translation and no rotation relative to `map`. In a real calibrated sensor rig, the camera transform should reflect the actual sensor mounting offset and orientation.

---

## Full sensor rig linkage

With both transforms present, the system creates a complete coordinate tree in ROS 2 Jazzy:

```text
          ┌──────────────┐
          │     map      │
          │ fixed frame  │
          └──────┬───────┘
                 │
      ┌──────────┼──────────┐
      │                     │
      ▼                     ▼
  tf_velodyne           tf_camera
      │                     │
      ▼                     ▼
  velodyne                camera
```

This provides a complete transform chain:

- `map -> velodyne`
- `map -> camera`
- `velodyne -> camera` indirectly through the TF tree

The exact implementation is defined in [ros2_ws/src/kitti_r1_perception/launch/myperception.launch.py](ros2_ws/src/kitti_r1_perception/launch/myperception.launch.py#L223-L258), where the `map_to_velodyne_tf` and `map_to_camera_tf` publishers are created with `static_transform_publisher`.

---

## BEV fusion and TF requirements

For BEV visualization and frustum fusion, the camera and LiDAR streams are not independent views; they must be related in one spatial frame. There are two distinct but connected requirements:

1. The fusion algorithm uses a direct LiDAR-to-camera rigid transform for projection math.
   - In the node, this is implemented as a rotation matrix and translation vector that transform LiDAR points into the camera frame before projection into image coordinates.
   - This is a geometric calibration step used within the algorithm itself.

2. The ROS TF tree is still required for visualization and world alignment.
   - `tf_velodyne` tells RViz where the LiDAR is relative to `map`.
   - `tf_camera` tells RViz where the camera is relative to `map`.
   - Together they allow the camera image, LiDAR cloud, and projected results to be displayed in a coherent world frame.

In short, the hardcoded calibration matrix handles the projection step, while TF handles the spatial context for visualization. Both are important, but they serve different purposes.

---

## Why quaternions are preferred

In ROS 2, quaternions are generally preferred over roll/pitch/yaw for sensor transforms because they avoid parser instability and gimbal lock. They also produce more efficient and numerically stable transform math in repeated projection and fusion pipelines.

### 1. ROS 2 argument parsing

The `static_transform_publisher` tool can behave inconsistently with Euler values such as `--roll`, `--pitch`, and `--yaw`. Quaternions map directly to the internal transform representation and are therefore more robust.

### 2. Gimbal lock

Euler angles are applied sequentially. Near a 90° pitch, two axes align and the system loses one degree of freedom. This is known as gimbal lock. Quaternions do not suffer from this issue.

### 3. Computational efficiency

Quaternions are more efficient and numerically stable than repeatedly converting Euler angles through trigonometric functions.

---

## Quick comparison

| Feature | Euler angles | Quaternions |
|---|---|---|
| Human readability | Intuitive | Harder to read mentally |
| Gimbal lock risk | Yes | No |
| ROS 2 parser stability | Can be unreliable | Stable and native |
| Processing cost | Higher | Lower |
| Recommended in ROS 2 | Sometimes | Preferred |

---

## Summary

Proper spatial calibration is essential for both visualization and fusion. In practice:

- `tf_velodyne` aligns the LiDAR with the fixed world frame
- `tf_camera` aligns the camera with the same frame
- the fusion node uses a direct LiDAR-to-camera rigid transform for projection geometry
- the current setup uses placeholder transforms and therefore should be treated as a calibration baseline, not a final sensor pose

---

## Compact note

- RViz requires TF to render multiple sensor streams in one coherent scene.
- The current launch-file sensor poses are placeholder values and do not represent the real KITTI sensor geometry.
- The frustum fusion node uses a direct rigid transformation for projection math, separate from the ROS TF tree.

### Future work

- TODO: replace the placeholder static transforms with the real KITTI calibration-derived poses for `velodyne` and `camera`.
  - Action plan: read the KITTI calibration files, convert the extrinsics into ROS TF poses, and publish the correct `map -> velodyne` and `map -> camera` transforms in the launch file.

- TODO: eliminate the hardcoded rigid transform in the frustum fusion node by loading the LiDAR-to-camera calibration from TF or `camera_info`-based calibration data instead of using a fixed matrix in code.
  - Action plan: remove the hardcoded `T_lidar_to_cam_np` matrix from the node, query the correct transform from TF at runtime, convert it to a NumPy/PyTorch matrix, and reuse it for the projection step so the fusion remains consistent with the calibrated sensor geometry.