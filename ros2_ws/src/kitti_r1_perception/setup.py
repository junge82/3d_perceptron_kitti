import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'kitti_r1_perception'
launch_files = [path for path in glob('launch/*.launch.py') if os.path.isfile(path)]

# 1. Start with your base data files
data_files_list = [
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    (os.path.join('share', package_name, 'launch'), launch_files),
]

# 3. Add all files in torch_home cleanly using a single list comprehension
data_files_list += [
    (os.path.join('share', package_name, path), [os.path.join(path, f)])
    for path, _, files in os.walk('torch_home') for f in files
]
setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=data_files_list,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Gergely Jung',
    maintainer_email='jung.gergely@outlook.hu',
    description='Modular 3D LiDAR-Camera Perception Pipeline for KITTI Dataset',
    license='MIT',
    entry_points={
        'console_scripts': [
            'lidar_cluster_node = kitti_r1_perception.lidar_cluster_node:main',
            'camera_detector_node = kitti_r1_perception.camera_detector_node:main',
            'frustum_fusion_node = kitti_r1_perception.frustum_fusion_node:main',
            'bev_visualizer_node = kitti_r1_perception.bev_visualizer_node:main',
            'kitti_image_to_bag = kitti_r1_perception.kitti_image_to_bag:main',
        ],
    },
)
