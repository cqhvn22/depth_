Place your calibration file in this directory with this exact name:

    stereo_calibration.npz

It must contain these NumPy arrays:

    left_map_x
    left_map_y
    right_map_x
    right_map_y
    Q

Alternatively, leave the file elsewhere and pass its absolute path:

    ros2 launch stereo_depth_node stereo_depth.launch.py \
      calibration_file:=/absolute/path/to/stereo_calibration.npz
