import os
import cv2
import time
from datetime import datetime
from camera_stream import Camera

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def safe_read(cam):
    try:
        grabbed, frame = cam.read()
    except Exception:
        return False, None
    return grabbed, frame

def take_and_save_pair(left_cam, right_cam, out_dir):
    grabbed_l, left_frame = safe_read(left_cam)
    grabbed_r, right_frame = safe_read(right_cam)

    if not grabbed_l or left_frame is None:
        print("Warning: left frame not available, skipping save")
        return False
    if not grabbed_r or right_frame is None:
        print("Warning: right frame not available, skipping save")
        return False

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    left_path = os.path.join(out_dir, f"left_{ts}.png")
    right_path = os.path.join(out_dir, f"right_{ts}.png")

    cv2.imwrite(left_path, left_frame)
    cv2.imwrite(right_path, right_frame)
    print(f"Saved: {left_path}  {right_path}")
    return True

def main():
    out_dir = os.path.join(os.getcwd(), "captures")
    ensure_dir(out_dir)

    left = Camera()
    left.open(0)
    left.start()

    right = Camera()
    right.open(1)
    right.start()

    try:
        if left.video_capture is None or not left.video_capture.isOpened() or \
           right.video_capture is None or not right.video_capture.isOpened():
            print("Unable to open one or both cameras")
            return

        cv2.namedWindow("Cam Left", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Cam Right", cv2.WINDOW_AUTOSIZE)

        print("Press 's' or Space to save a stereo pair. ESC to exit.")
        while True:
            _, limg = safe_read(left)
            _, rimg = safe_read(right)

            if limg is not None:
                cv2.imshow("Cam Left", limg)
            if rimg is not None:
                cv2.imshow("Cam Right", rimg)

            key = cv2.waitKey(30) & 0xFF
            if key == 27:  # ESC
                break
            if key == ord('s') or key == 32:  # 's' or Space
                take_and_save_pair(left, right, out_dir)
    finally:
        left.stop()
        left.release()
        right.stop()
        right.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()