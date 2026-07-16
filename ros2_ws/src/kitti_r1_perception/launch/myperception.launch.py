import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, RegisterEventHandler, OpaqueFunction, TimerAction
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import xacro

def generate_launch_description():
    
    # Common parameter to enforce across all tracking nodes
    sim_time_param = {'use_sim_time': True}

    # Declare the launch argument for the bag path
    bag_path_arg = DeclareLaunchArgument(
        'bag_path',
        default_value='/mnt/fastDisk/3d_perceptron_kitti/2011_09_29_drive_0071_sync/',
        description='Absolute path to the ROS2 bag directory'
    )
    # NEW: Control playback speed (e.g., 0.5 for half speed, 2.0 for double speed)
    play_rate_arg = DeclareLaunchArgument(
        'play_rate',
        default_value='1.0',
        description='Playback rate speed multiplier for ros2 bag play'
    )

    # NEW: Control the initial startup delay in seconds
    playback_delay_arg = DeclareLaunchArgument(
        'playback_delay',
        default_value='4.0',
        description='Seconds to wait after RViz starts before playing the bag'
    )

    # Lidar Cluster Node
    lidar_node = Node(
        package='kitti_r1_perception',
        executable='lidar_cluster_node',
        name='lidar_cluster_node',
        output='screen',
        emulate_tty=True,
        parameters=[sim_time_param]
    )

    # Camera Detector Node
    camera_node = Node(
        package='kitti_r1_perception',
        executable='camera_detector_node',
        name='camera_detector_node',
        output='screen',
        emulate_tty=True,
        parameters=[{'use_sim_time': True, 'model_version': 'yolov8n'}]
    )

    # Frustum Fusion Node
    fusion_node = Node(
        package='kitti_r1_perception',
        executable='frustum_fusion_node',
        name='frustum_fusion_node',
        output='screen',
        emulate_tty=True,
        parameters=[sim_time_param]
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
        arguments=['-d', rviz_config_file],
        parameters=[sim_time_param]
    )

    # Static transform publishers
    tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_velodyne_tf',
        arguments=[
            '--x', '0', '--y', '0', '--z', '0', 
            '--qx', '0', '--qy', '0', '--qz', '1', '--qw', '0', 
            '--frame-id', 'map', '--child-frame-id', 'velodyne'
        ]
    )

    tf_camera = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_camera_tf',
        arguments=[
            '--x', '0', '--y', '0', '--z', '0', 
            '--qx', '0', '--qy', '0', '--qz', '0', '--qw', '1', 
            '--frame-id', 'map', 
            '--child-frame-id', 'camera' 
        ]
    )

    # FIX: Define the setup function inside the main scope to natively capture 'rviz_node'
    def launch_setup(context, *args, **kwargs):
        # Safely retrieve the string path from the launch context
        bag_path_string = LaunchConfiguration('bag_path').perform(context)
        rate_string = LaunchConfiguration('play_rate').perform(context)
        delay_float = float(LaunchConfiguration('playback_delay').perform(context))

        # Define the bag process using the evaluated path
        bag_play_process = ExecuteProcess(
            cmd=['ros2', 'bag', 'play', bag_path_string, '--clock', '--rate', rate_string],
            output='screen'
        )

        delayed_bag_action = TimerAction(
            period=delay_float,
            actions=[bag_play_process]
        )

        # Wrap the bag process inside the OnProcessStart event handler
        delay_bag_play = RegisterEventHandler(
            event_handler=OnProcessStart(
                target_action=rviz_node,
                on_start=[delayed_bag_action]
            )
        )        
        return [delay_bag_play]

    return LaunchDescription([   
        bag_path_arg, 
        play_rate_arg,
        playback_delay_arg,
        lidar_node,
        camera_node,
        fusion_node,
        tf,
        tf_camera,
        rviz_node,
        # OpaqueFunction now runs cleanly without dict/argument mismatches
        OpaqueFunction(function=launch_setup)
    ])
