import time
import os
import cv2
import numpy as np
import vpi
from camera_stream import Camera

MAX_DISP = 64
WINDOW_SIZE = 10
# LEFT_CAL = "snapshots/calibration_data_left.npz"
# RIGHT_CAL = "snapshots/calibration_data_left.npz"


# def load_single_camera_maps(npz_path, image_size):
#     """
#     Load single-camera calibration saved by your Left_calibration.py / right_calibraion.py
#     and compute undistort maps using getOptimalNewCameraMatrix + initUndistortRectifyMap.
#     Note: this performs per-camera undistortion only. For proper stereo rectification
#     (epipolar alignment) you need stereo extrinsics (R, T) and cv2.stereoRectify to
#     generate rectification transforms.
#     """
#     if not os.path.exists(npz_path):
#         raise FileNotFoundError(npz_path)

#     data = np.load(npz_path, allow_pickle=True)
#     K = data["cameraMatrix"]
#     dist = data["distCoeffs"]

#     newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, image_size, 1, image_size)
#     mapx, mapy = cv2.initUndistortRectifyMap(K, dist, None, newK, image_size, cv2.CV_32FC1)
#     return (mapx, mapy), newK, roi


def main():
    # input camera resolution
    image_size = (1280, 720)

    # left_maps, left_newK, left_roi = load_single_camera_maps(LEFT_CAL, image_size)
    # right_maps, right_newK, right_roi = load_single_camera_maps(RIGHT_CAL, image_size)

    cam_l = Camera()
    cam_r = Camera()
    cam_l.open(0)
    cam_r.open(1)
    cam_l.start()
    cam_r.start()

    cv2.namedWindow("Left Undistorted", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Right Undistorted", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Disparity", cv2.WINDOW_AUTOSIZE)

    try:
        with vpi.Backend.CUDA:
            it = 0
            while True:
                t0 = time.perf_counter()
                g_l, frame_l = cam_l.read()
                g_r, frame_r = cam_r.read()
                if not g_l or not g_r or frame_l is None or frame_r is None:
                    time.sleep(0.01)
                    continue

                # Undistort using stored maps (this is NOT stereo-rectification)
                # frame_l_ud = cv2.remap(frame_l, *left_maps, interpolation=cv2.INTER_LINEAR)
                # frame_r_ud = cv2.remap(frame_r, *right_maps, interpolation=cv2.INTER_LINEAR)

                # Optionally crop by roi to remove black borders (uncomment if desired)
                # x,y,w,h = left_roi; frame_l_ud = frame_l_ud[y:y+h, x:x+w]
                # x,y,w,h = right_roi; frame_r_ud = frame_r_ud[y:y+h, x:x+w]

                # Resize to reduce workload
                # frame_l_rs = cv2.resize(frame_l_ud, (480, 270))
                # frame_r_rs = cv2.resize(frame_r_ud, (480, 270))
                
                # Resize to reduce workload
                frame_l_rs = cv2.resize(frame_l, (480, 270))
                frame_r_rs = cv2.resize(frame_r, (480, 270))

                # Convert to VPI images and compute disparity
                vpi_l = vpi.asimage(frame_l_rs)
                vpi_r = vpi.asimage(frame_r_rs)
                vpi_l_16 = vpi_l.convert(vpi.Format.U16, scale=1)
                vpi_r_16 = vpi_r.convert(vpi.Format.U16, scale=1)

                disp_16 = vpi.stereodisp(
                    vpi_l_16,
                    vpi_r_16,
                    out_confmap=None,
                    backend=vpi.Backend.CUDA,
                    window=WINDOW_SIZE,
                    maxdisp=MAX_DISP,
                )
                disp_8 = disp_16.convert(vpi.Format.U8, scale=255.0 / (32 * MAX_DISP))
                disp_arr = disp_8.cpu()

                disp_color = cv2.applyColorMap(disp_arr, cv2.COLORMAP_TURBO)

                cv2.imshow("Left Undistorted", frame_l_rs)
                cv2.imshow("Right Undistorted", frame_r_rs)
                cv2.imshow("Disparity", disp_color)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break

                if it % 30 == 0:
                    t1 = time.perf_counter()
                    print(f"Iter {it} loop {1000*(t1-t0):0.2f} ms")
                it += 1

    except KeyboardInterrupt:
        pass
    finally:
        cam_l.stop(); cam_r.stop()
        cam_l.release(); cam_r.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()