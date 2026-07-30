import os
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, RegisterEventHandler, OpaqueFunction, TimerAction
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare parameter mapping defaults
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true', 
        description='Use simulation clock topic if true'
    )

    sim_time_config = LaunchConfiguration('use_sim_time')

    package_share_dir = Path(get_package_share_directory('kitti_r1_perception'))
    print(f"Package share directory: {package_share_dir}")
    # 2. Step up past 'share', package name, 'install', to reach '3d_perceptron_kitti'
    workspace_root = package_share_dir.parents[3]
    print(f"Workspace root directory: {workspace_root}")
    # 3. Safely target your dataset folder relative to the workspace root core
    default_bag_path = (workspace_root / "2011_09_29_drive_0071_sync").resolve()

    # Declare the bag path argument using the calculated directory fallback
    bag_path_arg = DeclareLaunchArgument(
        'bag_path',
        default_value=str(default_bag_path),
        description='Path to the input ROS 2 bag folder'
    )

    log_level_arg = DeclareLaunchArgument(
        'log_level',
        default_value='INFO',
        description='Logging level (DEBUG, INFO, WARN, ERROR, FATAL)'
    )
    
    # NEW: Control playback speed
    play_rate_arg = DeclareLaunchArgument(
        'play_rate',
        default_value='1.0',
        description='Playback rate speed multiplier for ros2 bag play'
    )

    # NEW: Control the initial startup delay in seconds
    playback_delay_arg = DeclareLaunchArgument(
        'playback_delay',
        default_value='0.0',
        description='Seconds to wait after RViz starts before playing the bag'
    )

    # Lidar Cluster Node
    lidar_node = Node(
        package='kitti_r1_perception',
        executable='lidar_cluster_node',
        name='lidar_cluster_node',
        output='screen',
        emulate_tty=True,
        parameters=[{'use_sim_time': sim_time_config}],
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
        parameters=[{'use_sim_time': sim_time_config, 'model_version': 'yolov8n'}],
        arguments=['--ros-args',
                   '--log-level',
                    LaunchConfiguration('log_level'),
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
        arguments=['--ros-args',
                   '--log-level',
                   LaunchConfiguration('log_level'),
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
            arguments=['--ros-args',
                       '--log-level',
                       LaunchConfiguration('log_level'),
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

    # Static transform publishers
    map_to_velodyne_tf_publisher = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_velodyne_tf',
        parameters=[{'use_sim_time': sim_time_config}],
        arguments=[
            '--x', '0', '--y', '0', '--z', '0', 
            '--qx', '0', '--qy', '0', '--qz', '1', '--qw', '0', 
            '--frame-id', 'map', '--child-frame-id', 'velodyne'
        ]
    )

    map_to_camera_tf_publisher = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_camera_tf',
        parameters=[{'use_sim_time': sim_time_config}],
        arguments=[
            '--x', '0', '--y', '0', '--z', '0', 
            '--qx', '0', '--qy', '0', '--qz', '0', '--qw', '1', 
            '--frame-id', 'map', 
            '--child-frame-id', 'camera' 
        ]
    )

    def launch_setup(context, *args, **kwargs):
        bag_path_string = LaunchConfiguration('bag_path').perform(context)
        rate_string = LaunchConfiguration('play_rate').perform(context)
        delay_float = float(LaunchConfiguration('playback_delay').perform(context))

        bag_play_process = ExecuteProcess(
            cmd=[
                'ros2', 'bag', 'play', bag_path_string, 
                '--clock', 
                '--rate', rate_string
            ],
            output='screen'
        )

        delayed_bag_action = TimerAction(
            period=delay_float,
            actions=[bag_play_process]
        )

        delay_bag_play = RegisterEventHandler(
            event_handler=OnProcessStart(
                target_action=rviz_node,
                on_start=[delayed_bag_action]
            )
        )        
        return [delay_bag_play]

    return LaunchDescription([   
        use_sim_time_arg,  
        bag_path_arg, 
        play_rate_arg,
        playback_delay_arg,
        log_level_arg,
        #lidar_node,
        camera_node,
        #fusion_node,
        #bev_node,
        #map_to_velodyne_tf_publisher,
        #map_to_camera_tf_publisher,
        rviz_node,
        OpaqueFunction(function=launch_setup)
    ])
