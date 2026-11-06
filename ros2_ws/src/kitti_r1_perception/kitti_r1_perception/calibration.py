"""KITTI calibration values shared by fusion and launch files."""

import numpy as np


K_KITTI = np.array([
    [721.5377, 0.0, 609.5593],
    [0.0, 721.5377, 172.854],
    [0.0, 0.0, 1.0],
], dtype=np.float32)

T_LIDAR_TO_CAM = np.array([
    [7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
    [1.480249e-02, 7.280733e-04, -9.998902e-01, -7.631618e-02],
    [9.998621e-01, 7.523790e-03, 1.480755e-02, -2.717806e-01],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float32)

# Existing map-to-velodyne launch translation.
MAP_TO_VELODYNE_TRANSLATION = np.array([
    0.27290344,
    -0.00196927,
    -0.07228590,
], dtype=np.float32)

MAP_TO_VELODYNE_QUATERNION = np.array([
    0.0,
    0.0,
    1.0,
    0.0,
], dtype=np.float32)
