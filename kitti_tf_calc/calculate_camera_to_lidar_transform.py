"""Print the inverse of the KITTI LiDAR-to-camera calibration transform."""

from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation


package_source = (
    Path(__file__).resolve().parents[1]
    / 'ros2_ws'
    / 'src'
    / 'kitti_r1_perception'
)
sys.path.insert(0, str(package_source))

from kitti_r1_perception.calibration import T_LIDAR_TO_CAM  # noqa: E402


def main():
    rotation_lidar_to_camera = T_LIDAR_TO_CAM[:3, :3]
    translation_lidar_to_camera = T_LIDAR_TO_CAM[:3, 3]

    rotation_camera_to_lidar = rotation_lidar_to_camera.T
    translation_camera_to_lidar = (
        -rotation_camera_to_lidar @ translation_lidar_to_camera
    )
    quaternion_camera_to_lidar = Rotation.from_matrix(
        rotation_camera_to_lidar
    ).as_quat()

    np.set_printoptions(precision=8, suppress=True)
    print("Inverse KITTI transform (camera -> LiDAR):")
    print(f"translation [m]: {translation_camera_to_lidar}")
    print(f"quaternion [x, y, z, w]: {quaternion_camera_to_lidar}")


if __name__ == '__main__':
    main()
