# ROS

## Overview

This folder contains the **ROS 2 (Humble)** packages for the IMX219-83 stereo camera rig on the Jetson Orin Nano. Together, the two packages split camera/IMU capture and depth processing into independent nodes connected purely through standard ROS 2 topics:

- **`stereo_camera`** — C++ nodes that capture the two IMX219-83 CSI sensors and the ICM-20948 IMU, and publish them as standard `sensor_msgs` topics (`image_raw`, `camera_info`, `imu/data`, `imu/mag`).
- **`stereo_depth_node`** — a Python node with no camera code and no dependency on `stereo_camera`. It subscribes to any synchronized left/right `image_raw` topics and publishes disparity, depth, and rectified images.

Because they only talk to each other through ROS 2 messages, `stereo_depth_node` can consume images from `stereo_camera` (this repo's C++ driver), the Python driver in `../Python/`, or any other node that publishes `sensor_msgs/msg/Image` on the expected topics.

## Structure

```text
ROS/
├── stereo_camera/         # C++ camera + IMU driver package (stereo_camera_ros2)
│   └── README.md
└── stereo_depth_node/     # Python depth/disparity processing package
    └── README.md
```

See each package's own `README.md` for dependencies, build/run instructions, and topic details.

## Outputs

_Sample rqt_graph / topic list screenshots will be added here._

<!-- Example:
### Node graph
![ros node graph](../docs/images/ros_node_graph.png)
-->
