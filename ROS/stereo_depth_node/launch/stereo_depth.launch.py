from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    package_share = Path(get_package_share_directory("stereo_depth_node"))
    default_calibration = str(package_share / "config" / "stereo_calibration.npz")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "left_topic",
                default_value="/stereo/left/image_raw",
            ),
            DeclareLaunchArgument(
                "right_topic",
                default_value="/stereo/right/image_raw",
            ),
            DeclareLaunchArgument(
                "calibration_file",
                default_value=default_calibration,
            ),
            DeclareLaunchArgument(
                "calibration_unit",
                default_value="mm",
                description="Unit used for the stereo calibration translation: m, cm, or mm",
            ),
            DeclareLaunchArgument("downscale", default_value="0.5"),
            DeclareLaunchArgument("sync_queue_size", default_value="10"),
            DeclareLaunchArgument("sync_slop", default_value="0.03"),
            Node(
                package="stereo_depth_node",
                executable="stereo_depth_node",
                name="stereo_depth_node",
                output="screen",
                emulate_tty=True,
                parameters=[
                    {
                        "left_topic": LaunchConfiguration("left_topic"),
                        "right_topic": LaunchConfiguration("right_topic"),
                        "calibration_file": LaunchConfiguration(
                            "calibration_file"
                        ),
                        "calibration_unit": LaunchConfiguration(
                            "calibration_unit"
                        ),
                        "downscale": ParameterValue(
                            LaunchConfiguration("downscale"), value_type=float
                        ),
                        "sync_queue_size": ParameterValue(
                            LaunchConfiguration("sync_queue_size"), value_type=int
                        ),
                        "sync_slop": ParameterValue(
                            LaunchConfiguration("sync_slop"), value_type=float
                        ),
                    }
                ],
            ),
        ]
    )
