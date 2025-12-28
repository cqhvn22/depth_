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
    if not grabbed or frame is None:
        return False, None
    return True, frame

def wait_for_frame(cam, timeout=1.0, interval=0.05):
    """
    Try to obtain a valid frame from cam within timeout seconds.
    Returns (True, frame) or (False, None).
    """
    start = time.time()
    while time.time() - start < timeout:
        ok, frame = safe_read(cam)
        if ok:
            return True, frame
        time.sleep(interval)
    return False, None

def take_and_save_pair(left_cam, right_cam, out_dir, timeout=1.0):
    """
    Attempt to get a fresh frame from both cameras (with short waiting).
    Save into out_dir/left and out_dir/right. Returns True if both saved.
    """
    left_dir = os.path.join(out_dir, "left")
    right_dir = os.path.join(out_dir, "right")
    ensure_dir(left_dir)
    ensure_dir(right_dir)

    ok_l, left_frame = wait_for_frame(left_cam, timeout=timeout)
    ok_r, right_frame = wait_for_frame(right_cam, timeout=timeout)

    if not ok_l:
        print("Warning: left frame not available (timeout), skipping save")
        return False
    if not ok_r:
        print("Warning: right frame not available (timeout), skipping save")
        return False

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    left_path = os.path.join(left_dir, f"left_{ts}.png")
    right_path = os.path.join(right_dir, f"right_{ts}.png")

    # Attempt to write and verify success
    ok_write_l = cv2.imwrite(left_path, left_frame)
    ok_write_r = cv2.imwrite(right_path, right_frame)

    if not ok_write_l or not ok_write_r:
        print(f"Error: failed to write images. left_ok={ok_write_l} right_ok={ok_write_r}")
        # Clean up any partial file
        try:
            if os.path.exists(left_path) and not ok_write_l:
                os.remove(left_path)
            if os.path.exists(right_path) and not ok_write_r:
                os.remove(right_path)
        except Exception:
            pass
        return False

    print(f"Saved stereo pair:\n  {left_path}\n  {right_path}")
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
            ok_l, limg = safe_read(left)
            ok_r, rimg = safe_read(right)

            if ok_l and limg is not None:
                cv2.imshow("Cam Left", limg)
            else:
                # show a black frame if no image to keep window responsive
                cv2.imshow("Cam Left", cv2.imread(os.devnull) if False else (limg if limg is not None else 255 * (np.zeros((10,10,3), dtype='uint8'))))

            if ok_r and rimg is not None:
                cv2.imshow("Cam Right", rimg)
            else:
                cv2.imshow("Cam Right", cv2.imread(os.devnull) if False else (rimg if rimg is not None else 255 * (np.zeros((10,10,3), dtype='uint8'))))

            key = cv2.waitKey(30) & 0xFF
            if key == 27:  # ESC
                break
            if key == ord('s') or key == 32:  # 's' or Space
                take_and_save_pair(left, right, out_dir, timeout=1.0)
    finally:
        left.stop()
        left.release()
        right.stop()
        right.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Delay a bit to allow camera threads to initialize reliably on startup
    main()