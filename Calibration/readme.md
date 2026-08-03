# Calibration

## Overview

This folder contains the tools used to stereo-calibrate the two IMX219-83 CSI cameras on the Jetson Orin Nano. It covers the full calibration workflow: capturing synchronized chessboard image pairs from both sensors, running stereo calibration/rectification on those pairs, and producing the `stereo_calibration.npz` file consumed by the depth scripts in `Python/` and the ROS 2 packages in `ROS/`.

A pre-computed `stereo_calibration.npz` for the reference camera rig is included, so calibration only needs to be re-run if you are using a different physical rig, baseline, or resolution.

## Structure

```text
Calibration/
├── stereo_calibration_capture.py   # Captures synchronized left/right chessboard image pairs
├── calibration.py                  # Runs stereo calibration + rectification on captured pairs
├── stereo_calibration.npz          # Pre-computed calibration output for the reference rig
└── README.md                       # This file
```

`stereo_calibration_capture.py` writes captured images into a `calibration_images/` directory it creates alongside itself:

```text
calibration_images/
├── left/      # Saved left-camera chessboard frames
├── right/     # Saved right-camera chessboard frames
└── preview/   # Preview frames with detected corners drawn, for sanity-checking
```

## Installation / Dependencies

```bash
sudo apt update
sudo apt install python3-opencv python3-numpy
```

Runs on the Jetson Orin Nano with both IMX219-83 CSI sensors connected (`stereo_calibration_capture.py` opens them through GStreamer/`nvarguscamerasrc`). `calibration.py` only reads image files and can be run on the Jetson or any machine with the images copied over.

A physical chessboard calibration target is required (default: 9x6 internal corners, 25 mm squares — configurable, see below).

## Running

### 1. Capture chessboard pairs

```bash
python3 stereo_calibration_capture.py \
    --left-sensor 0 \
    --right-sensor 1 \
    --board-size 9x6 \
    --target-pairs 35
```

Controls while the capture window is open:

- `SPACE` or `S` — save the current stereo pair
- `A` — toggle automatic capture (captures at a fixed interval)
- `R` — reset the saved-pair counter
- `Q` or `ESC` — quit

Other useful options (all optional, shown with their defaults):

```text
--left-sensor 0            --right-sensor 1
--sensor-mode 2
--capture-width 960       --capture-height 540
--output-width 960         --output-height 540
--framerate 30
--flip-method 2
--target-pairs 30
```

Move the chessboard through different positions, angles, and distances (including the edges of the frame) to get a diverse set of pairs. Aim for at least 15 valid pairs, preferably 25–50.

### 2. Run calibration

`calibration.py` reads from `calibration_images/left` and `calibration_images/right` by default. Edit the constants at the top of the script (`BOARD_SIZE`, `SQUARE_SIZE_MM`, `LEFT_DIR`, `RIGHT_DIR`, `OUTPUT_FILE`) to match your capture, then run:

```bash
python3 calibration.py
```

This will:

1. Load and pair up the left/right images.
2. Detect chessboard corners in each pair (skipping any pair where detection fails or resolution is inconsistent).
3. Calibrate each camera individually, then jointly (stereo calibration) with the intrinsics held fixed.
4. Compute stereo rectification maps.
5. Print the RMS reprojection errors and the estimated baseline (for the reference rig this should be close to 60 mm).
6. Save everything to `stereo_calibration.npz`.

### 3. Verify calibration

```bash
python3 verify_stereo_calibration.py \
    --calibration stereo_calibration.npz \
    --left-dir calibration_images/left \
    --right-dir calibration_images/right \
    --board-size 9x6 \
    --expected-baseline 60
```

> Note: `verify_stereo_calibration.py` is referenced by the original calibration workflow but is not currently included in this folder. Add it here if you want an automated verification step, or verify manually by checking the printed RMS errors and baseline from step 2, and by visually inspecting rectified image pairs.

## Output

`stereo_calibration.npz` contains, among other arrays: `K_left`, `D_left`, `K_right`, `D_right` (intrinsics/distortion), `R`, `T`, `E`, `F` (stereo geometry), `R_left`, `R_right`, `P_left`, `P_right`, `Q` (rectification), and `left_map_x`/`left_map_y`/`right_map_x`/`right_map_y` (remap lookup tables). This is the file expected by `Python/stereo_depth.py`, `Python/depth_test.py`, and the ROS 2 `stereo_depth_node`.

## Outputs

_Sample chessboard capture and rectification images will be added here._

<!-- Example:
### Detected chessboard corners
![chessboard corners](../docs/images/calibration_corners.png)

### Rectified stereo pair
![rectified pair](../docs/images/rectified_pair.png)
-->