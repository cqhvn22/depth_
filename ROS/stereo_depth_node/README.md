# stereo_depth_node

This is an independent ROS 2 Humble package. It contains no camera capture code
and has no dependency on a camera package. It only subscribes to:

- `/stereo/left/image_raw`
- `/stereo/right/image_raw`

The two messages are synchronized by their ROS header timestamps.

## Package layout

```text
stereo_depth_node/
├── CMakeLists.txt
├── package.xml
├── config/
│   └── README.txt
├── launch/
│   └── stereo_depth.launch.py
└── src/
    └── stereo_depth_node.py
```

## Published topics

- `/stereo/depth/image_raw` — `sensor_msgs/Image`, `32FC1`, meters
- `/stereo/disparity/image_raw` — `sensor_msgs/Image`, `32FC1`, pixels
- `/stereo/disparity/color` — `sensor_msgs/Image`, `bgr8`
- `/stereo/left/image_rect` — `sensor_msgs/Image`, `bgr8`
- `/stereo/right/image_rect` — `sensor_msgs/Image`, `bgr8`
- `/stereo/depth/center` — `std_msgs/Float32`, meters; `0.0` means unavailable
- `/stereo/sync/delta_ms` — `std_msgs/Float32`, milliseconds

## Build

Copy the entire `stereo_depth_node` directory into your ROS workspace `src`:

```bash
cd ~/GitHub/camera-test-main/ROS2/stereo_camera
cp -r /path/to/stereo_depth_node src/

rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-select stereo_depth_node
source install/setup.bash
```

The node requires OpenCV contrib because it uses `cv2.ximgproc`.

## Run

```bash
ros2 launch stereo_depth_node stereo_depth.launch.py \
  calibration_file:=/absolute/path/to/stereo_calibration.npz \
  calibration_unit:=mm
```

Or place the file at `config/stereo_calibration.npz` before building, then run:

```bash
ros2 launch stereo_depth_node stereo_depth.launch.py
```

## Input requirements

The left and right images must match the resolution used to create the
calibration maps. Both input messages must have valid timestamps in their
headers.
