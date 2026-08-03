# Cpp

## Overview

This folder contains a standalone **C++** implementation of dual IMX219-83 CSI camera capture on the Jetson Orin Nano, equivalent in purpose to the Python version in `../Python/` but with no ROS or Python dependency. It opens both CSI sensors through GStreamer/`nvarguscamerasrc`, reads each camera on its own background thread, and displays a live left/right preview. This is useful as a minimal, dependency-light way to verify both cameras are working, and as a starting point for the C++ code used in the ROS 2 `stereo_camera` package.

## Structure

```text
Cpp/
├── dual_csi_camera.cpp   # Source: threaded dual-CSI capture + live preview
└── dual_csi_camera       # Pre-built binary (compiled from dual_csi_camera.cpp)
```

`dual_csi_camera.cpp` defines a small `Camera` class that:

- opens one CSI sensor via a `nvarguscamerasrc` GStreamer pipeline (`cv::VideoCapture` with `cv::CAP_GSTREAMER`),
- reads frames continuously on a background `std::thread`, guarded by a `std::mutex`,
- exposes a thread-safe `read()` that copies out the latest frame.

`main()` opens sensor IDs `0` and `1` as the left/right cameras, starts both reader threads, and shows `Cam Left` / `Cam Right` preview windows until `Esc` is pressed.

## Installation / Dependencies

Runs on the Jetson Orin Nano with both IMX219-83 CSI sensors connected (capture is done through `nvarguscamerasrc`, so this will not work on a machine without the Jetson camera stack).

```bash
sudo apt update
sudo apt install build-essential libopencv-dev
```

The default pipeline in `dual_csi_camera.cpp` requests `sensor-mode=3` (1280x720, ~20 fps), `flip-method=0`. Edit the constants at the top of `Camera::open()` (`sensor_mode`, `capture_width`, `capture_height`, `display_width`, `display_height`, `framerate`, `flip_method`) if your sensors need different settings, then rebuild.

## Running

### Build

```bash
cd Cpp
g++ dual_csi_camera.cpp -o dual_csi_camera `pkg-config --cflags --libs opencv4` -lpthread
```

### Run

```bash
./dual_csi_camera
```

Opens `Cam Left` and `Cam Right` preview windows. Press `Esc` to quit.

A pre-built `dual_csi_camera` binary is already included in this folder; if it doesn't run on your setup (e.g. different architecture or OpenCV version), rebuild it with the command above.

If a camera fails to open, the pipeline string that was attempted is printed to `stderr` — useful for checking sensor IDs, mode, or resolution against `v4l2-ctl`/`gst-launch-1.0` diagnostics.