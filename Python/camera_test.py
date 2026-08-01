import cv2
import sys

from camera import StereoCamera

def main():
    stereo = StereoCamera(
        left_sensor_id=0,
        right_sensor_id=1,
        sensor_mode=2,
        capture_width=960,
        capture_height=540,
        output_width=960,
        output_height=540,
        framerate=30,
        flip_method=0,
    )

    if not stereo.open():
        sys.exit("Could not open the stereo cameras.")

    if not stereo.start():
        stereo.release()
        sys.exit("Could not start stereo capture.")

    cv2.namedWindow("Cam Left", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Cam Right", cv2.WINDOW_AUTOSIZE)

    last_sequence = None
    frame_counter = 0

    try:
        while True:
            available, pair = stereo.read(
                last_sequence=last_sequence,
                copy_frames=False,
            )

            if not available:
                key = cv2.waitKey(1) & 0xFF

                if key == 27:
                    break

                continue

            last_sequence = pair.sequence
            frame_counter += 1

            # Print occasionally instead of flooding the terminal.
            if frame_counter % 30 == 0:
                print(
                    f"Pair {pair.sequence}: "
                    f"software retrieve delta = {pair.delta_ms:.3f} ms"
                )

            cv2.imshow("Cam Left", pair.left)
            cv2.imshow("Cam Right", pair.right)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break

    finally:
        stereo.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
