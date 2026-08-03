# Python

## Overview

This folder contains standalone Python scripts for capturing and processing the dual IMX219-83 CSI stereo camera **without ROS**. It provides a threaded stereo camera reader, a live dual-preview test script, a stereo depth/disparity engine (rectification + SGBM + WLS filtering), and a live depth-visualization test script. This is the quickest way to verify the camera rig and calibration are working before moving to the C++ or ROS 2 packages elsewhere in the repo.

## Structure

```text
Python/
├── camera.py          # StereoCamera: threaded dual-CSI GStreamer capture, returns synced left/right frames
├── camera_test.py      # Live preview of both cameras side by side (no depth)
├── stereo_depth.py     # StereoDepth: rectification + SGBM disparity + WLS filtering + depth
├── depth_test.py       # Live depth/disparity visualization using camera.py + stereo_depth.py
└── README.txt          # Note on where to place the calibration file
```

- `camera.py` — defines `StereoFrame` (left/right images, per-camera software timestamps, timestamp delta, and a sequence number) and `StereoCamera`, which opens both sensors via `nvarguscamerasrc`/GStreamer pipelines and grabs synchronized pairs on a background thread.
- `stereo_depth.py` — defines `StereoDepth`, which loads the rectification maps and `Q` matrix from a calibration `.npz` file, rectifies each incoming pair, computes disparity with `cv2.StereoSGBM` plus a right-matcher, applies a WLS disparity filter, and reprojects to a per-pixel depth map.
- `camera_test.py` and `depth_test.py` are runnable entry points built on top of the two modules above.

## Installation / Dependencies

Runs on the Jetson Orin Nano with both IMX219-83 CSI sensors connected (frames are captured through `nvarguscamerasrc` via GStreamer, so this will not work on a machine without the Jetson camera stack).

```bash
sudo apt update
sudo apt install python3-opencv python3-numpy
# stereo_depth.py uses cv2.ximgproc (WLS filter), which needs opencv-contrib:
pip3 install opencv-contrib-python
```

### Calibration file

`depth_test.py` requires a stereo calibration file named exactly `stereo_calibration.npz`, placed in this folder. Either:

- copy the pre-computed one from `../Calibration/stereo_calibration.npz`, or
- generate your own by following `../Calibration/README.md`.

(See `README.txt` in this folder for the same note.)

## Running

### Preview both cameras (no depth)

```bash
python3 camera_test.py
```

Opens two windows, `Cam Left` and `Cam Right`, showing live frames from both sensors. Prints the software retrieve timestamp delta between the two cameras every 30 frames. Press `Esc` to quit.

### Live depth / disparity

```bash
cp ../Calibration/stereo_calibration.npz .   # if not already present
python3 depth_test.py
```

Opens two windows:

- `Rectified Left` — the rectified left frame with the frame-center marked, the estimated distance at that center point overlaid, and the left/right capture timestamp delta.
- `Disparity` — a color-mapped (Turbo colormap) view of the disparity map.

Press `q` or `Esc` to quit.

### Using `StereoCamera` / `StereoDepth` directly

```python
from camera import StereoCamera
from stereo_depth import StereoDepth

camera = StereoCamera(left_sensor_id=0, right_sensor_id=1, output_width=960, output_height=540, framerate=30)
depth_estimator = StereoDepth("stereo_calibration.npz")

camera.open()
camera.start()

ok, pair = camera.read()
if ok:
    left_rectified, disparity, depth = depth_estimator.process(pair.left, pair.right)
```

## Outputs

_Sample camera preview and depth/disparity visualization images will be added here._

<!-- Example:
### Dual camera preview
![camera preview](../docs/images/python_camera_preview.png)

### Live depth visualization
![depth visualization](../docs/images/python_depth_test.png)
-->
