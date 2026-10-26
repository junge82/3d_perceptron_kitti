import os
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, ExecuteProcess, LogInfo, RegisterEventHandler, OpaqueFunction, TimerAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.event_handlers import OnProcessExit, OnProcessStart
from launch.events import Shutdown
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
    default_bag_path = (workspace_root / "2011_09_29_drive_0071_sync").resolve() # Adjust this to your actual dataset folder 

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
    
    # Control playback speed
    play_rate_arg = DeclareLaunchArgument(
        'play_rate',
        default_value='1.0',
        description='Playback rate speed multiplier for ros2 bag play'
    )

    playback_delay_arg = DeclareLaunchArgument(
        'playback_delay',
        default_value='0.0',
        description='Seconds to wait after RViz starts before playing the bag'
    )

    shutdown_on_bag_exit_arg = DeclareLaunchArgument(
        'shutdown_on_bag_exit',
        default_value='false',
        description='Shut down the launch system when bag playback exits'
    )

    # Lidar cluster tuning parameters
    voxel_size_arg = DeclareLaunchArgument(
        'voxel_size',
        default_value='0.15',
        description='Voxel size for lidar downsampling'
    )
    ground_distance_threshold_arg = DeclareLaunchArgument(
        'ground_distance_threshold',
        default_value='0.3',
        description='Distance threshold for ground plane segmentation'
    )
    cluster_tolerance_arg = DeclareLaunchArgument(
        'cluster_tolerance',
        default_value='0.6',
        description='DBSCAN tolerance for lidar clustering'
    )
    min_cluster_size_arg = DeclareLaunchArgument(
        'min_cluster_size',
        default_value='20',
        description='Minimum number of points to keep a cluster'
    )
    max_cluster_size_arg = DeclareLaunchArgument(
        'max_cluster_size',
        default_value='1000',
        description='Maximum number of points for a cluster'
    )

    # BEV ROI parameters
    bev_min_x_arg = DeclareLaunchArgument(
        'bev_min_x',
        default_value='0.1',
        description='BEV minimum forward distance'
    )
    bev_max_x_arg = DeclareLaunchArgument(
        'bev_max_x',
        default_value='50.0',
        description='BEV maximum forward distance'
    )
    bev_min_y_arg = DeclareLaunchArgument(
        'bev_min_y',
        default_value='-25.0',
        description='BEV minimum lateral distance'
    )
    bev_max_y_arg = DeclareLaunchArgument(
        'bev_max_y',
        default_value='25.0',
        description='BEV maximum lateral distance'
    )
    bev_resolution_arg = DeclareLaunchArgument(
        'bev_resolution',
        default_value='0.10',
        description='BEV image resolution in meters per pixel'
    )
    bev_min_z_arg = DeclareLaunchArgument(
        'bev_min_z',
        default_value='-1.8',
        description='BEV minimum height'
    )
    bev_max_z_arg = DeclareLaunchArgument(
        'bev_max_z',
        default_value='2.0',
        description='BEV maximum height'
    )

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

    # ⚡ DIRECT PYTHON NODE DECLARATION (FIXED EXECUTABLE NAME)
    # foxglove_bridge_node = Node(
    #     package='foxglove_bridge',
    #     executable='foxglove_bridge',  # ⚡ Fixed from foxglove_bridge_node
    #     name='foxglove_bridge',
    #     output='screen',
    #     parameters=[{
    #         'use_sim_time': sim_time_config,
    #         'port': 8765,                 
    #         'address': '0.0.0.0',         
    #         'send_buffer_limit': 10000000 
    #     }]
    # )

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

    # 1. World Frame to Core Camera Link (Identity Base)
# 1. TF: map -> camera (Kamera rögzítése egyenesen az origóhoz)
    map_to_camera_tf_publisher = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_camera_tf',
        parameters=[{'use_sim_time': sim_time_config}],
        arguments=[
            '--x', '0.0', '--y', '0.0', '--z', '0.0', 
            '--qx', '0.0', '--qy', '0.0', '--qz', '0.0', '--qw', '1.0', 
            '--frame-id', 'map', 
            '--child-frame-id', 'camera'
        ]
    )

    # 2. TF: map -> velodyne (Közvetlen kapcsolat a map-ből, az elején működő 180 fokos pörgéssel!)
    map_to_velodyne_tf_publisher = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_velodyne_tf',
        parameters=[{'use_sim_time': sim_time_config}],
        arguments=[
            '--x', '0.2729', '--y', '-0.0020', '--z', '-0.0723', # Éles fizikai eltolások (a calib.txt T oszlopából)
            '--qx', '0.0', '--qy', '0.0', '--qz', '1.0', '--qw', '0.0', # Az eredeti, jól működő 180 fokos irányzék!
            '--frame-id', 'map', 
            '--child-frame-id', 'velodyne'
        ]
    )

    def launch_setup(context, *args, **kwargs):
        bag_path_string = LaunchConfiguration('bag_path').perform(context)
        rate_string = LaunchConfiguration('play_rate').perform(context)
        delay_float = float(LaunchConfiguration('playback_delay').perform(context))
        shutdown_on_bag_exit = LaunchConfiguration('shutdown_on_bag_exit').perform(context).lower() in ('1', 'true', 'yes')

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

        actions = [delay_bag_play]
        if shutdown_on_bag_exit:
            actions.insert(0, RegisterEventHandler(
                event_handler=OnProcessStart(
                    target_action=bag_play_process,
                    on_start=[LogInfo(msg='BENCHMARK_BAG_PLAYBACK_STARTED')],
                )
            ))

            def on_bag_exit(event, context):
                exit_code = event.returncode
                return [
                    LogInfo(msg=f'BENCHMARK_BAG_EXIT_CODE={exit_code}'),
                    EmitEvent(event=Shutdown(reason='Bag playback process exited')),
                ]

            actions.insert(0, RegisterEventHandler(
                event_handler=OnProcessExit(
                    target_action=bag_play_process,
                    on_exit=on_bag_exit,
                )
            ))
        return actions

    return LaunchDescription([   
        use_sim_time_arg,  
        bag_path_arg, 
        log_level_arg,
        play_rate_arg,
        playback_delay_arg,
        shutdown_on_bag_exit_arg,
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
        # foxglove_bridge_node,
        fusion_node,
        bev_node,
        map_to_camera_tf_publisher,
        map_to_velodyne_tf_publisher,
        rviz_node,
        OpaqueFunction(function=launch_setup)
    ])
