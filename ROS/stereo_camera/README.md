# stereo_camera (stereo_camera_ros2)

## Overview

ROS 2 Humble package that drives the IMX219-83 stereo camera pair and the ICM-20948 IMU on the Jetson Orin Nano, from C++, and publishes them as standard ROS 2 topics. It bundles:

- a C++ dual-CSI camera node (`stereo_camera_node`),
- a C++ IMU node (`icm20948_node`),
- ROS `CameraInfo` calibration generated from this repo's `stereo_calibration.npz`,
- a Python example subscriber showing how to consume the published images from Python (e.g. for depth processing).

Both images in a stereo pair receive the same ROS timestamp; a published `sync_delta_ms` reports the actual delay between the two software `retrieve()` calls (this measures software synchronization, not hardware sync).

## Structure

```text
stereo_camera/
├── src/
│   ├── stereo_camera.cpp          # Dual-CSI camera capture implementation
│   ├── stereo_camera_node.cpp     # ROS 2 node wrapping the camera capture
│   ├── icm20948_node.cpp          # ROS 2 node wrapping the IMU driver
│   └── ICM20948.c                 # ICM-20948 driver (see ../../IMU/README.md for origin)
├── include/stereo_camera_ros2/
│   ├── stereo_camera.hpp
│   └── ICM20948.h
├── config/
│   ├── camera.yaml                 # stereo_camera_node parameters
│   ├── imu.yaml                    # icm20948_node parameters
│   ├── left_camera_info.yaml       # ROS CameraInfo for the left camera
│   ├── right_camera_info.yaml      # ROS CameraInfo for the right camera
│   ├── stereo_calibration.npz      # Source OpenCV calibration (see CALIBRATION.md)
│   └── CALIBRATION.md              # How the .npz maps to the CameraInfo YAML files
├── launch/
│   └── stereo_camera.launch.py     # Launches camera node (+ optional IMU node)
├── examples/
│   ├── python_stereo_subscriber.py # Minimal Python subscriber for the stereo images
│   └── stereo_imu_test.py          # Subscribes to stereo images + IMU together
├── tools/
│   └── npz_to_ros_camera_info.py   # Converts an OpenCV .npz calibration to CameraInfo YAML
├── CMakeLists.txt
├── package.xml
└── README.md                       # This file
```

### Published topics

**`stereo_camera_node`**

- `/stereo/left/image_raw` (`sensor_msgs/msg/Image`, `bgr8`)
- `/stereo/left/camera_info` (`sensor_msgs/msg/CameraInfo`)
- `/stereo/right/image_raw` (`sensor_msgs/msg/Image`, `bgr8`)
- `/stereo/right/camera_info` (`sensor_msgs/msg/CameraInfo`)
- `/stereo/sync_delta_ms` (`std_msgs/msg/Float32`)

**`icm20948_node`**

- `/imu/data`
- `/imu/mag`

## Dependencies

```bash
sudo apt update
sudo apt install ros-humble-camera-calibration-parsers
sudo apt install ros-humble-camera-info-manager
sudo apt install \
  libopencv-dev \
  ros-humble-cv-bridge \
  ros-humble-message-filters
```

Or, from inside the ROS 2 workspace:

```bash
rosdep install --from-paths src --ignore-src -r -y
```

`cv_bridge`, `message_filters`, `rclpy`, and `python3-opencv` are only required for the optional Python example subscriber (see `package.xml`) — the camera and IMU nodes themselves are pure C++.

Hardware: Jetson Orin Nano, both IMX219-83 CSI sensors connected, and (optionally) the ICM-20948 IMU wired over I2C — see `../../Pinout.png`.

## Build

```bash
colcon build --symlink-install --packages-select stereo_camera_ros2
source install/setup.bash
```

Verify both C++ executables were installed:

```bash
ls -l install/stereo_camera_ros2/lib/stereo_camera_ros2/
```

Expected:

```text
icm20948_node
stereo_camera_node
```

## Examples

### Launch camera + IMU

```bash
ros2 launch stereo_camera_ros2 stereo_camera.launch.py
```

### Launch camera only (no IMU)

```bash
ros2 launch stereo_camera_ros2 stereo_camera.launch.py start_imu:=false
```

### Inspect topics

```bash
ros2 topic list
ros2 topic hz /stereo/left/image_raw
ros2 topic echo /stereo/sync_delta_ms
```

### Consume the images from Python

After sourcing the workspace:

```bash
python3 install/stereo_camera_ros2/share/stereo_camera_ros2/examples/python_stereo_subscriber.py
```

or, for a combined stereo + IMU example:

```bash
python3 install/stereo_camera_ros2/share/stereo_camera_ros2/examples/stereo_imu_test.py
```

A downstream depth node can subscribe to the same `/stereo/left/image_raw` and `/stereo/right/image_raw` topics and pass the converted `left`/`right` arrays directly into rectification and `StereoSGBM` — this is exactly what `../stereo_depth_node` does.

### Calibration

`config/camera.yaml` already points at the included `left_camera_info.yaml` / `right_camera_info.yaml`, generated from `config/stereo_calibration.npz` (see `config/CALIBRATION.md` for unit-handling details). To point at a different calibration, edit `camera.yaml`:

```yaml
left_camera_info_url: file:///absolute/path/left.yaml
right_camera_info_url: file:///absolute/path/right.yaml
```

`package://package_name/path/to/file.yaml` is also supported. Calibration resolution must match the configured output resolution (`output_width` / `output_height` in `camera.yaml`).

To regenerate the CameraInfo YAML from a new `.npz`, use `tools/npz_to_ros_camera_info.py`.

## Outputs

This is the sample result of the stereo camera and IMU data.

<p align="center">
  <img src="../../images/ROS Stereo Camera and IMU.png" alt="ROS Stereo Camera and IMU.png">
  <br>
  <em>ROS Stereo Camera and IMU data</em>
</p>
