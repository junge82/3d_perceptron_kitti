# Math Behind the Converter

## Sensor Timestamps

KITTI timestamps are parsed as UTC. The converter avoids floating-point time conversion and builds integer nanoseconds:

```text
timestamp_ns = seconds_since_epoch * 1_000_000_000 + fractional_nanoseconds
```

The fractional part is padded or truncated to nine digits. Each stream's timestamp is used for its ROS message header and MCAP record.

## LiDAR Point Layout

Each binary KITTI point contains four consecutive `float32` values: `x`, `y`, `z`, and intensity.

```text
point_step = 4 values * 4 bytes = 16 bytes
row_step = point_count * point_step
```

The converter keeps the point order and copies the packed values into `PointCloud2.data`.

## Camera Calibration

For each rectified camera, `P_rect` is split into its intrinsic matrix `K` and final column `p4`. Camera translation is recovered by solving `K * t = p4`; the implementation uses a linear solve instead of computing `inverse(K)`.

The transform from Velodyne coordinates to a rectified camera frame is composed as:

```text
R_camera_from_velodyne = R_rect * R_velo
t_camera_from_velodyne = R_rect * t_velo + t_camera
```

The left-from-right transform is:

```text
T_left_from_right = T_left_from_velodyne * inverse(T_right_from_velodyne)
```

These are homogeneous rigid transforms: the upper-left 3x3 block is rotation and the final column is translation. `TransformStamped` stores the rotation as a quaternion. The parent frame is the destination frame and the child frame is the source frame.

## CameraInfo

The rectified intrinsic matrix `K` is the left 3x3 portion of `P_rect`. Rectified images use identity rectification and zero distortion in the emitted `CameraInfo`; `P` retains the full 3x4 projection matrix.
