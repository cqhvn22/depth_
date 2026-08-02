# stereo_camera_ros2 (C++ camera version)

ROS 2 Humble package for:

- two Jetson CSI IMX219 cameras, captured and published from C++;
- an ICM-20948 IMU node;
- Python depth processing through standard ROS 2 image topics.

## Nodes

### `stereo_camera_node` (C++)

Publishes:

- `/stereo/left/image_raw` (`sensor_msgs/msg/Image`, `bgr8`)
- `/stereo/left/camera_info` (`sensor_msgs/msg/CameraInfo`)
- `/stereo/right/image_raw` (`sensor_msgs/msg/Image`, `bgr8`)
- `/stereo/right/camera_info` (`sensor_msgs/msg/CameraInfo`)
- `/stereo/sync_delta_ms` (`std_msgs/msg/Float32`)

Both images in a pair receive the same ROS timestamp. The `sync_delta_ms`
value measures the delay between the two software `retrieve()` calls; it does
not prove hardware synchronization.

### `icm20948_node` (C++)

Publishes:

- `/imu/data`
- `/imu/mag`

## Why Python depth still works

ROS 2 messages are language independent. The C++ camera node publishes
`sensor_msgs/msg/Image`; a Python node subscribes and converts each message to
a NumPy/OpenCV array with `CvBridge.imgmsg_to_cv2()`.

An example is installed from:

```text
examples/python_stereo_subscriber.py
```

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

Or install package dependencies with rosdep:

```bash
rosdep install --from-paths src --ignore-src -r -y
```

## Build

```bash
cd ~/GitHub/camera-test-main/ROS2/stereo_camera
rm -rf build install log
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

## Run

```bash
ros2 launch stereo_camera_ros2 stereo_camera.launch.py
```

Camera only:

```bash
ros2 launch stereo_camera_ros2 stereo_camera.launch.py start_imu:=false
```

## Calibration

Edit `config/camera.yaml`:

```yaml
left_camera_info_url: file:///absolute/path/left.yaml
right_camera_info_url: file:///absolute/path/right.yaml
```

`package://package_name/path/to/file.yaml` is also supported.

Calibration resolution should match the configured output resolution.

## Use the C++ images in Python

After sourcing the workspace:

```bash
python3 install/stereo_camera_ros2/share/stereo_camera_ros2/examples/python_stereo_subscriber.py
```

Your depth node can use the same subscriptions and pass `left` and `right`
directly to rectification and StereoSGBM.

## Included stereo calibration

This package variant includes the supplied `stereo_calibration.npz`, generated ROS
`CameraInfo` YAML files, and a reusable NPZ-to-YAML converter. The camera parameter
file is already configured to load them.
