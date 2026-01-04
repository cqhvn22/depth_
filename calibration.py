import os
import glob
import time
import numpy as np
import cv2 as cv
from camera_stream import Camera

# Calibration params (adjust as needed)
chessboardSize = (6, 9)  # internal corners (columns, rows)
square_size = 23  # mm (or any unit)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

snap_dir = "snapshots"
os.makedirs(snap_dir, exist_ok=True)


def run_calibration(image_dir, chessboard_size, square_size):
    images = glob.glob(os.path.join(image_dir, "*.png"))
    if not images:
        print("No images found for calibration in", image_dir)
        return

    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []
    imgpoints = []

    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)
        if ret:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            cv.drawChessboardCorners(img, chessboard_size, corners2, ret)
            cv.imshow("Detected Corners", img)
            cv.waitKey(200)
        else:
            print("Chessboard not detected in", fname)

    cv.destroyAllWindows()

    if not objpoints:
        print("No valid chessboard detections. Calibration aborted.")
        return

    sample = cv.imread(images[0])
    h, w = sample.shape[:2]

    ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (w, h), None, None)
    print("Calibration RMS:", ret)
    print("Camera matrix:\n", cameraMatrix)
    print("Distortion coefficients:\n", distCoeffs.ravel())

    np.savez(os.path.join(image_dir, "calibration_data.npz"), cameraMatrix=cameraMatrix, distCoeffs=distCoeffs)

    # Undistort example (remap)
    newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, (w, h), 1, (w, h))
    mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, distCoeffs, None, newCameraMatrix, (w, h), 5)
    remapped = cv.remap(sample, mapx, mapy, cv.INTER_LINEAR)
    x, y, w_roi, h_roi = roi
    if w_roi > 0 and h_roi > 0:
        remapped = remapped[y: y + h_roi, x: x + w_roi]
    out_path = os.path.join(image_dir, "undistorted_sample.png")
    cv.imwrite(out_path, remapped)
    print("Undistorted sample written to", out_path)

    # Reprojection error
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        total_error += error
    print("Mean reprojection error:", total_error / len(objpoints))


def main():
    cam = Camera()
    cam.open(sensor_id=0) #sensor_id = 0 is the right one, sensor_id = 1 is the one on the left
    cam.start()

    cv.namedWindow("Calibration", cv.WINDOW_AUTOSIZE)
    print("SPACE/S to save snapshot, C to calibrate, ESC to exit")

    try:
        while cv.getWindowProperty("Calibration", 0) >= 0:
            grabbed, frame = cam.read()
            if not grabbed or frame is None:
                continue

            display = frame.copy()
            cv.putText(display, "S:save  C:calibrate  ESC:quit", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv.imshow("Calibration", display)

            key = cv.waitKey(30) & 0xFF
            if key == 27:  # ESC
                break
            if key == ord(' ') or key == ord('s'):
                filename = os.path.join(snap_dir, f"snapshot_{int(time.time())}.png")
                cv.imwrite(filename, frame)
                print("Saved", filename)
            if key == ord('c'):
                print("Running calibration on snapshots...")
                run_calibration(snap_dir, chessboardSize, square_size)
    finally:
        cam.stop()
        cam.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()