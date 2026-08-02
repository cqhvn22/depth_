# Calibration integration

The source `stereo_calibration.npz` was calibrated at **960 x 540**.

## Included files

- `config/stereo_calibration.npz`: original OpenCV calibration, rectification maps, and Q.
- `config/left_camera_info.yaml`: ROS CameraInfo calibration for the left camera.
- `config/right_camera_info.yaml`: ROS CameraInfo calibration for the right camera.
- `tools/npz_to_ros_camera_info.py`: reusable converter.

## Unit handling

The calibration chessboard square size is expressed in millimetres, so `T`, `P_right[0,3]`,
and `Q` encode millimetre-based stereo geometry.

The ROS right CameraInfo YAML converts only `P_right[0,3]` from pixel-mm to pixel-m.
This gives ROS stereo consumers a baseline in metres.

The original NPZ is unchanged. When Python calls `cv2.reprojectImageTo3D(disparity, Q)`,
the returned X/Y/Z coordinates are in millimetres. Divide them by 1000 for metres.

## Camera node

`config/camera.yaml` already points to the two generated calibration YAML files using
`package://stereo_camera_ros2/config/...`.
