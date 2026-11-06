import os
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from kitti_r1_perception.calibration import (
    MAP_TO_VELODYNE_QUATERNION,
    MAP_TO_VELODYNE_TRANSLATION,
)

def generate_launch_description():
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true', 
        description='Use simulation clock topic if true'
    )

    sim_time_config = LaunchConfiguration('use_sim_time')

    log_level_arg = DeclareLaunchArgument(
        'log_level',
        default_value='INFO',
        description='Logging level (DEBUG, INFO, WARN, ERROR, FATAL)'
    )

    # Lidar cluster tuning parameters
    voxel_size_arg = DeclareLaunchArgument('voxel_size', default_value='0.15')
    ground_distance_threshold_arg = DeclareLaunchArgument('ground_distance_threshold', default_value='0.3')
    cluster_tolerance_arg = DeclareLaunchArgument('cluster_tolerance', default_value='0.6')
    min_cluster_size_arg = DeclareLaunchArgument('min_cluster_size', default_value='20')
    max_cluster_size_arg = DeclareLaunchArgument('max_cluster_size', default_value='1000')

    # BEV ROI parameters
    bev_min_x_arg = DeclareLaunchArgument('bev_min_x', default_value='1.5')
    bev_max_x_arg = DeclareLaunchArgument('bev_max_x', default_value='50.0')
    bev_min_y_arg = DeclareLaunchArgument('bev_min_y', default_value='-25.0')
    bev_max_y_arg = DeclareLaunchArgument('bev_max_y', default_value='25.0')
    bev_resolution_arg = DeclareLaunchArgument('bev_resolution', default_value='0.10')
    bev_min_z_arg = DeclareLaunchArgument('bev_min_z', default_value='-1.8')
    bev_max_z_arg = DeclareLaunchArgument('bev_max_z', default_value='2.0')

    # Lidar Cluster Node
    lidar_node = Node(
        package='kitti_r1_perception',
        executable='lidar_cluster_node',
        name='lidar_cluster_node',
        output='screen',
        emulate_tty=True,
        parameters=[
            {'use_sim_time': sim_time_config},
            {'voxel_size': LaunchConfiguration('voxel_size')},
            {'ground_distance_threshold': LaunchConfiguration('ground_distance_threshold')},
            {'cluster_tolerance': LaunchConfiguration('cluster_tolerance')},
            {'min_cluster_size': LaunchConfiguration('min_cluster_size')},
            {'max_cluster_size': LaunchConfiguration('max_cluster_size')},
            {'bev_min_x': LaunchConfiguration('bev_min_x')},
            {'bev_max_x': LaunchConfiguration('bev_max_x')},
            {'bev_min_y': LaunchConfiguration('bev_min_y')},
            {'bev_max_y': LaunchConfiguration('bev_max_y')},
            {'bev_min_z': LaunchConfiguration('bev_min_z')},
            {'bev_max_z': LaunchConfiguration('bev_max_z')}
        ],
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level'),
                   '--log-file-name', 'lidar_cluster_node.log'
        ]
    )

    # Camera Detector Node
    camera_node = Node(
        package='kitti_r1_perception',
        executable='camera_detector_node',
        name='camera_detector_node',
        output='screen',
        emulate_tty=True,
        additional_env={'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:128',
                        'TORCH_CUDA_ARCH_LIST': '8.9+PTX',
                        'TORCH_HOME': os.path.join(get_package_share_directory('kitti_r1_perception'), 'models'),
                        'PYTHONUNBUFFERED': '1'
                        },
        parameters=[{'use_sim_time': sim_time_config, 'model_version': 'yolov8n'},
                    {'bev_min_x': LaunchConfiguration('bev_min_x')},
                    {'bev_max_x': LaunchConfiguration('bev_max_x')},
                    {'bev_min_y': LaunchConfiguration('bev_min_y')},
                    {'bev_max_y': LaunchConfiguration('bev_max_y')}
                   ],
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level'),
                    '--log-file-name', 'camera_detector_node.log'
                ],
    )

    # Frustum Fusion Node
    fusion_node = Node(
        package='kitti_r1_perception',
        executable='frustum_fusion_node',
        name='frustum_fusion_node',
        output='screen',
        emulate_tty=True,
        parameters=[{'use_sim_time': sim_time_config}],
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level'),
                   '--log-file-name', 'frustum_fusion_node.log'
                   ]
    )

    # BEV Visualizer Node
    bev_node = Node(
        package='kitti_r1_perception',
        executable='bev_visualizer_node',
        name='bev_visualizer_node',
        output='screen',
        parameters=[{'use_sim_time': sim_time_config}],
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level'),
                   '--log-file-name', 'bev_visualizer_node.log'
                   ]
    )

    # RViz2 Node
    rviz_config_file = PathJoinSubstitution([
        FindPackageShare('kitti_r1_description'),
        'rviz',
        'mydisplay.rviz'
    ])

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',        
        parameters=[{'use_sim_time': sim_time_config}],
        arguments=['-d', rviz_config_file,
                   '--ros-args', '--log-level', LaunchConfiguration('log_level'),
                   '--log-file-name', 'rviz2.log'
        ])

    # TF Static Publishers
    map_to_camera_tf_publisher = Node(
        package='tf2_ros', executable='static_transform_publisher', name='map_to_camera_tf',
        parameters=[{'use_sim_time': sim_time_config}],
        arguments=['--x', '0.0', '--y', '0.0', '--z', '0.0', '--qx', '0.0', '--qy', '0.0', '--qz', '0.0', '--qw', '1.0', '--frame-id', 'map', '--child-frame-id', 'camera']
    )

    map_to_velodyne_tf_publisher = Node(
        package='tf2_ros', executable='static_transform_publisher', name='map_to_velodyne_tf',
        parameters=[{'use_sim_time': sim_time_config}],
        arguments=[
            '--x', f'{MAP_TO_VELODYNE_TRANSLATION[0]:.8f}',
            '--y', f'{MAP_TO_VELODYNE_TRANSLATION[1]:.8f}',
            '--z', f'{MAP_TO_VELODYNE_TRANSLATION[2]:.8f}',
            '--qx', f'{MAP_TO_VELODYNE_QUATERNION[0]:.8f}',
            '--qy', f'{MAP_TO_VELODYNE_QUATERNION[1]:.8f}',
            '--qz', f'{MAP_TO_VELODYNE_QUATERNION[2]:.8f}',
            '--qw', f'{MAP_TO_VELODYNE_QUATERNION[3]:.8f}',
            '--frame-id', 'map', '--child-frame-id', 'velodyne',
        ]
    )

    return LaunchDescription([
        use_sim_time_arg,  
        log_level_arg,
        voxel_size_arg,
        ground_distance_threshold_arg,
        cluster_tolerance_arg,
        min_cluster_size_arg,
        max_cluster_size_arg,
        bev_min_x_arg,
        bev_max_x_arg,
        bev_min_y_arg,
        bev_max_y_arg,
        bev_resolution_arg,
        bev_min_z_arg,
        bev_max_z_arg,
        lidar_node,
        camera_node,
        fusion_node,
        bev_node,
        map_to_camera_tf_publisher,
        map_to_velodyne_tf_publisher,
        rviz_node
    ])
