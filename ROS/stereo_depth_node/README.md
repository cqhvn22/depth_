# stereo_depth_node

## Overview

Independent ROS 2 Humble package that computes stereo depth/disparity in Python. It contains **no camera capture code** and has no dependency on the `stereo_camera` package — it only subscribes to a pair of `sensor_msgs/msg/Image` topics, so it can consume images published by `../stereo_camera` (C++, this repo), the Python driver in `../../Python/`, or any other source that publishes rectifiable stereo image pairs on ROS topics. The two input images are synchronized by their ROS header timestamps.

Internally it rectifies each incoming pair, computes disparity with OpenCV's `StereoSGBM` plus a right-matcher and WLS filter (same approach as `../../Python/stereo_depth.py`), reprojects to depth using the stereo calibration `Q` matrix, and republishes everything.

## Structure

```text
stereo_depth_node/
├── src/
│   └── stereo_depth_node.py   # Node: subscribes to stereo images, publishes depth/disparity
├── config/
│   └── README.txt             # Where to place / how to pass the calibration file
├── launch/
│   └── stereo_depth.launch.py # Launch file with configurable topics/calibration/params
├── example/
│   └── depth_viewer.py        # Example subscriber that displays depth/disparity/center distance
├── CMakeLists.txt
├── package.xml
└── README.md                  # This file
```

### Subscribed topics

- `/stereo/left/image_raw`
- `/stereo/right/image_raw`

(topic names configurable — see Examples below)

### Published topics

- `/stereo/depth/image_raw` — `sensor_msgs/Image`, `32FC1`, meters
- `/stereo/disparity/image_raw` — `sensor_msgs/Image`, `32FC1`, pixels
- `/stereo/disparity/color` — `sensor_msgs/Image`, `bgr8`
- `/stereo/left/image_rect` — `sensor_msgs/Image`, `bgr8`
- `/stereo/right/image_rect` — `sensor_msgs/Image`, `bgr8`
- `/stereo/depth/center` — `std_msgs/Float32`, meters; `0.0` means unavailable
- `/stereo/sync/delta_ms` — `std_msgs/Float32`, milliseconds

## Dependencies

ROS 2 Humble, plus (see `package.xml`):

```bash
sudo apt update
sudo apt install \
  ros-humble-cv-bridge \
  ros-humble-message-filters \
  python3-numpy \
  python3-opencv

# cv2.ximgproc (WLS disparity filter) requires opencv-contrib:
pip3 install opencv-contrib-python
```

Or, from inside the ROS 2 workspace:

```bash
rosdep install --from-paths src --ignore-src -r -y
```

The node requires OpenCV **contrib** because it uses `cv2.ximgproc`.

A stereo calibration `.npz` file (see `../../Calibration/README.md`) is required at runtime — it must contain `left_map_x`, `left_map_y`, `right_map_x`, `right_map_y`, and `Q`.

## Build

Copy the entire `stereo_depth_node` directory into your ROS workspace `src`:

```bash
cd ~/GitHub/camera-test-main/ROS2/stereo_camera
cp -r /path/to/stereo_depth_node src/

rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-select stereo_depth_node
source install/setup.bash
```

## Examples

### Run with an explicit calibration path

```bash
ros2 launch stereo_depth_node stereo_depth.launch.py \
  calibration_file:=/absolute/path/to/stereo_calibration.npz \
  calibration_unit:=mm
```

### Run with the calibration file placed alongside the package

Place the file at `config/stereo_calibration.npz` (see `config/README.txt`) before building, then:

```bash
ros2 launch stereo_depth_node stereo_depth.launch.py
```

### Override input topics or processing parameters

```bash
ros2 launch stereo_depth_node stereo_depth.launch.py \
  left_topic:=/stereo/left/image_raw \
  right_topic:=/stereo/right/image_raw \
  downscale:=0.5 \
  sync_queue_size:=10 \
  sync_slop:=0.03
```

### View the output

```bash
python3 install/stereo_depth_node/share/stereo_depth_node/example/depth_viewer.py
```

`depth_viewer.py` subscribes to `/stereo/depth/image_raw`, `/stereo/disparity/color`, and `/stereo/depth/center`, and displays them live.

### Pairing with `stereo_camera`

Run the two packages together to go from raw CSI capture to live depth:

```bash
ros2 launch stereo_camera_ros2 stereo_camera.launch.py
ros2 launch stereo_depth_node stereo_depth.launch.py
```

## Input requirements

The left and right images must match the resolution used to create the calibration maps. Both input messages must have valid timestamps in their headers.

## Outputs

This the sample data of ROS depth node.

<p align="center">
  <img src="../../images/ROS depth.png" alt="ROS depth.png">
  <br>
  <em>ROS depth data</em>
</p>
