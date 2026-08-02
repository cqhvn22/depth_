from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    share_dir = Path(get_package_share_directory('stereo_camera_ros2'))

    camera_config = LaunchConfiguration('camera_config')
    imu_config = LaunchConfiguration('imu_config')
    start_imu = LaunchConfiguration('start_imu')

    return LaunchDescription([
        DeclareLaunchArgument(
            'camera_config',
            default_value=str(share_dir / 'config' / 'camera.yaml'),
        ),
        DeclareLaunchArgument(
            'imu_config',
            default_value=str(share_dir / 'config' / 'imu.yaml'),
        ),
        DeclareLaunchArgument('start_imu', default_value='true'),
        Node(
            package='stereo_camera_ros2',
            executable='stereo_camera_node',
            name='stereo_camera_node',
            output='screen',
            parameters=[camera_config],
        ),
        Node(
            package='stereo_camera_ros2',
            executable='icm20948_node',
            name='icm20948_node',
            output='screen',
            parameters=[imu_config],
            condition=IfCondition(start_imu),
        ),
    ])
