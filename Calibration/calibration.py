from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

LEFT_DIR = Path("calibration_images/left")
RIGHT_DIR = Path("calibration_images/right")
OUTPUT_FILE = "stereo_calibration.npz"

# Number of INTERNAL chessboard corners: columns, rows
BOARD_SIZE = (9, 6)

# Physical square size. Using millimetres makes T and Q use mm.
SQUARE_SIZE_MM = 25.0

# Set to your known capture resolution if desired.
# The script will obtain it from the first valid image.
EXPECTED_IMAGE_SIZE = None  # Example: (960, 540)


# ---------------------------------------------------------
# Calibration pattern model
# ---------------------------------------------------------

def create_object_points(
    board_size: tuple[int, int],
    square_size: float,
) -> np.ndarray:
    """Create planar chessboard points in real-world units."""
    cols, rows = board_size

    points = np.zeros((rows * cols, 3), dtype=np.float32)
    points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    points *= square_size

    return points


def find_corners(
    gray: np.ndarray,
    board_size: tuple[int, int],
) -> tuple[bool, np.ndarray | None]:
    """Detect and refine chessboard corners."""

    # SB is generally robust for modern OpenCV versions.
    found, corners = cv2.findChessboardCornersSB(
        gray,
        board_size,
        flags=(
            cv2.CALIB_CB_NORMALIZE_IMAGE
            | cv2.CALIB_CB_EXHAUSTIVE
            | cv2.CALIB_CB_ACCURACY
        ),
    )

    if found:
        return True, corners.astype(np.float32)

    # Fallback for installations where the SB detector struggles.
    found, corners = cv2.findChessboardCorners(
        gray,
        board_size,
        flags=(
            cv2.CALIB_CB_ADAPTIVE_THRESH
            | cv2.CALIB_CB_NORMALIZE_IMAGE
        ),
    )

    if not found:
        return False, None

    termination = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        50,
        1e-4,
    )

    refined = cv2.cornerSubPix(
        gray,
        corners,
        winSize=(11, 11),
        zeroZone=(-1, -1),
        criteria=termination,
    )

    return True, refined


# ---------------------------------------------------------
# Load corresponding image pairs
# ---------------------------------------------------------

left_paths = sorted(
    list(LEFT_DIR.glob("*.png"))
    + list(LEFT_DIR.glob("*.jpg"))
    + list(LEFT_DIR.glob("*.jpeg"))
)

right_paths = sorted(
    list(RIGHT_DIR.glob("*.png"))
    + list(RIGHT_DIR.glob("*.jpg"))
    + list(RIGHT_DIR.glob("*.jpeg"))
)

if not left_paths or not right_paths:
    raise RuntimeError("No calibration images found.")

if len(left_paths) != len(right_paths):
    raise RuntimeError(
        f"Different image counts: left={len(left_paths)}, "
        f"right={len(right_paths)}"
    )

object_template = create_object_points(BOARD_SIZE, SQUARE_SIZE_MM)

object_points: list[np.ndarray] = []
left_image_points: list[np.ndarray] = []
right_image_points: list[np.ndarray] = []

image_size = None

for index, (left_path, right_path) in enumerate(
    zip(left_paths, right_paths)
):
    left = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
    right = cv2.imread(str(right_path), cv2.IMREAD_COLOR)

    if left is None or right is None:
        print(f"Skipping pair {index}: image could not be loaded.")
        continue

    if left.shape[:2] != right.shape[:2]:
        print(f"Skipping pair {index}: different image dimensions.")
        continue

    current_size = (left.shape[1], left.shape[0])

    if image_size is None:
        image_size = current_size

    if current_size != image_size:
        print(f"Skipping pair {index}: inconsistent resolution.")
        continue

    if EXPECTED_IMAGE_SIZE and current_size != EXPECTED_IMAGE_SIZE:
        raise RuntimeError(
            f"Expected {EXPECTED_IMAGE_SIZE}, received {current_size}"
        )

    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    left_found, left_corners = find_corners(left_gray, BOARD_SIZE)
    right_found, right_corners = find_corners(right_gray, BOARD_SIZE)

    if not left_found or not right_found:
        print(
            f"Skipping pair {index}: "
            f"left={left_found}, right={right_found}"
        )
        continue

    object_points.append(object_template.copy())
    left_image_points.append(left_corners)
    right_image_points.append(right_corners)

    print(f"Accepted pair {index}: {left_path.name}, {right_path.name}")


if image_size is None:
    raise RuntimeError("No readable image pairs found.")

if len(object_points) < 15:
    raise RuntimeError(
        f"Only {len(object_points)} valid pairs. "
        "Capture at least 15; preferably 25–50 diverse pairs."
    )

print(f"\nUsing {len(object_points)} valid stereo pairs.")
print(f"Calibration image size: {image_size}")


# ---------------------------------------------------------
# Calibrate each camera separately
# ---------------------------------------------------------

mono_criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    100,
    1e-7,
)

left_rms, K_left, D_left, _, _ = cv2.calibrateCamera(
    object_points,
    left_image_points,
    image_size,
    None,
    None,
    criteria=mono_criteria,
)

right_rms, K_right, D_right, _, _ = cv2.calibrateCamera(
    object_points,
    right_image_points,
    image_size,
    None,
    None,
    criteria=mono_criteria,
)

print("\nIndividual calibration:")
print(f"Left RMS reprojection error:  {left_rms:.4f} px")
print(f"Right RMS reprojection error: {right_rms:.4f} px")


# ---------------------------------------------------------
# Stereo calibration
# ---------------------------------------------------------

stereo_criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    200,
    1e-8,
)

# Keep the independently estimated intrinsics fixed and solve
# for the relative rotation and translation.
stereo_flags = cv2.CALIB_FIX_INTRINSIC

(
    stereo_rms,
    K_left,
    D_left,
    K_right,
    D_right,
    R,
    T,
    E,
    F,
) = cv2.stereoCalibrate(
    object_points,
    left_image_points,
    right_image_points,
    K_left,
    D_left,
    K_right,
    D_right,
    image_size,
    criteria=stereo_criteria,
    flags=stereo_flags,
)

baseline_mm = float(np.linalg.norm(T))

print("\nStereo calibration:")
print(f"Stereo RMS error: {stereo_rms:.4f} px")
print(f"Estimated translation T, mm:\n{T}")
print(f"Estimated baseline: {baseline_mm:.3f} mm")
print(f"Expected baseline: approximately 60 mm")
print(f"Rotation matrix R:\n{R}")


# ---------------------------------------------------------
# Stereo rectification
# ---------------------------------------------------------

(
    R_left,
    R_right,
    P_left,
    P_right,
    Q,
    valid_roi_left,
    valid_roi_right,
) = cv2.stereoRectify(
    K_left,
    D_left,
    K_right,
    D_right,
    image_size,
    R,
    T,
    flags=cv2.CALIB_ZERO_DISPARITY,
    alpha=0,
)

left_map_x, left_map_y = cv2.initUndistortRectifyMap(
    K_left,
    D_left,
    R_left,
    P_left,
    image_size,
    cv2.CV_32FC1,
)

right_map_x, right_map_y = cv2.initUndistortRectifyMap(
    K_right,
    D_right,
    R_right,
    P_right,
    image_size,
    cv2.CV_32FC1,
)


# ---------------------------------------------------------
# Save parameters
# ---------------------------------------------------------

np.savez_compressed(
    OUTPUT_FILE,
    image_size=np.array(image_size),
    board_size=np.array(BOARD_SIZE),
    square_size_mm=np.array(SQUARE_SIZE_MM),
    left_rms=np.array(left_rms),
    right_rms=np.array(right_rms),
    stereo_rms=np.array(stereo_rms),
    K_left=K_left,
    D_left=D_left,
    K_right=K_right,
    D_right=D_right,
    R=R,
    T=T,
    E=E,
    F=F,
    R_left=R_left,
    R_right=R_right,
    P_left=P_left,
    P_right=P_right,
    Q=Q,
    left_map_x=left_map_x,
    left_map_y=left_map_y,
    right_map_x=right_map_x,
    right_map_y=right_map_y,
    valid_roi_left=np.array(valid_roi_left),
    valid_roi_right=np.array(valid_roi_right),
)

print(f"\nSaved calibration to: {OUTPUT_FILE}")