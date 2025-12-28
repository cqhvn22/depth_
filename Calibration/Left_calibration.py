import numpy as np
import cv2 as cv
import glob
import os

chessboardSize = (9, 9)          
square_size = 18.0               # mm
frameSize = (1280, 720)          # Enter IMX219-83 camera resolution

image_dir = "left"
output_file = "left_calibration.npz"

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []
imgpoints = []

images = glob.glob(os.path.join(image_dir, "*.png"))

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

    if ret:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv.imshow("LEFT Camera Corners", img)
        cv.waitKey(300)
    else:
        print(f"[LEFT] Chessboard not detected in {fname}")

cv.destroyAllWindows()

ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv.calibrateCamera(
    objpoints, imgpoints, frameSize, None, None
)

print("LEFT Camera Matrix:\n", cameraMatrix)
print("LEFT Distortion Coeffs:\n", distCoeffs.ravel())

np.savez(
    output_file,
    cameraMatrix=cameraMatrix,
    distCoeffs=distCoeffs,
    rvecs=rvecs,
    tvecs=tvecs
)

if images:
    img = cv.imread(images[0])
    h, w = img.shape[:2]
    newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(
        cameraMatrix, distCoeffs, (w, h), 1, (w, h)
    )

    mapx, mapy = cv.initUndistortRectifyMap(
        cameraMatrix, distCoeffs, None, newCameraMatrix, (w, h), 5
    )
    undistorted = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

    x, y, w, h = roi
    undistorted = undistorted[y:y+h, x:x+w]
    cv.imwrite("left_undistorted.png", undistorted)

total_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(
        objpoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs
    )
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    total_error += error

print("LEFT Mean reprojection error:", total_error / len(objpoints))
