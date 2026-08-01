import cv2
import numpy as np

from camera import StereoCamera
from stereo_depth import StereoDepth


def disparity_to_color(disparity):
    valid = disparity > 0
    display = np.zeros(disparity.shape, dtype=np.uint8)

    if np.any(valid):
        valid_disparity = disparity[valid]

        minimum = np.percentile(valid_disparity, 2)
        maximum = np.percentile(valid_disparity, 98)

        if maximum > minimum:
            normalized = np.clip(
                (disparity - minimum) / (maximum - minimum),
                0,
                1,
            )

            display = (normalized * 255).astype(np.uint8)

    display[~valid] = 0

    return cv2.applyColorMap(
        display,
        cv2.COLORMAP_TURBO,
    )


def get_center_depth(depth, calibration_unit="mm"):
    height, width = depth.shape
    center_x = width // 2
    center_y = height // 2

    # Use a region instead of one noisy pixel.
    region = depth[
        center_y - 5:center_y + 6,
        center_x - 5:center_x + 6,
    ]

    valid_depths = region[
        np.isfinite(region) & (region > 0)
    ]

    if valid_depths.size == 0:
        return None

    distance = float(np.median(valid_depths))

    # Q produces values in the same unit used for calibration.
    if calibration_unit == "mm":
        distance /= 1000.0
    elif calibration_unit == "cm":
        distance /= 100.0

    return distance


def main():
    camera = StereoCamera(
        left_sensor_id=0,
        right_sensor_id=1,
        output_width=960,
        output_height=540,
        framerate=30,
    )

    depth_estimator = StereoDepth(
        "stereo_calibration.npz"
    )

    if not camera.open():
        raise RuntimeError("Could not open stereo camera.")

    if not camera.start():
        camera.release()
        raise RuntimeError("Could not start stereo camera.")

    last_sequence = None

    try:
        while True:
            success, stereo_frame = camera.read(
                last_sequence=last_sequence,
                copy_frames=False,
            )

            # No new stereo pair yet.
            if not success or stereo_frame is None:
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q") or key == 27:
                    break

                continue

            last_sequence = stereo_frame.sequence

            left_rectified, disparity, depth = (
                depth_estimator.process(
                    stereo_frame.left,
                    stereo_frame.right,
                )
            )

            center_depth_m = get_center_depth(
                depth,
                calibration_unit="mm",
            )

            height, width = depth.shape
            center = (width // 2, height // 2)

            cv2.drawMarker(
                left_rectified,
                center,
                (0, 255, 0),
                cv2.MARKER_CROSS,
                20,
                2,
            )

            if center_depth_m is None:
                depth_text = "Depth: unavailable"
            else:
                depth_text = f"Depth: {center_depth_m:.2f} m"

            cv2.putText(
                left_rectified,
                depth_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                left_rectified,
                f"Pair delta: {stereo_frame.delta_ms:.2f} ms",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            disparity_display = disparity_to_color(
                disparity
            )

            cv2.imshow(
                "Rectified Left",
                left_rectified,
            )

            cv2.imshow(
                "Disparity",
                disparity_display,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
