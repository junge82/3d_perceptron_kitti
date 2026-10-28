from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    package_share_dir = Path(get_package_share_directory('kitti_r1_perception'))
    workspace_root = package_share_dir.parents[3]
    default_bag_path = (workspace_root / "2011_09_29_drive_0071_sync").resolve()

    bag_path_arg = DeclareLaunchArgument(
        'bag_path',
        default_value=str(default_bag_path),
        description='Path to the input ROS 2 bag folder'
    )
    
    play_rate_arg = DeclareLaunchArgument(
        'play_rate',
        default_value='1.0',
        description='Playback rate speed multiplier'
    )

    bag_play_process = ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'play', 
            LaunchConfiguration('bag_path'), 
            '--clock', 
            '--rate', LaunchConfiguration('play_rate')
        ],
        output='screen'
    )

    return LaunchDescription([
        bag_path_arg,
        play_rate_arg,
        bag_play_process
    ])
