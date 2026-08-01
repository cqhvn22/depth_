import cv2
import numpy as np


class StereoDepth:
    def __init__(self, calibration_file):
        data = np.load(calibration_file)

        self.left_map_x = data["left_map_x"]
        self.left_map_y = data["left_map_y"]
        self.right_map_x = data["right_map_x"]
        self.right_map_y = data["right_map_y"]
        self.Q = data["Q"]

        block_size = 5

        self.matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=128,  # Must be divisible by 16
            blockSize=block_size,
            P1=8 * block_size**2,
            P2=32 * block_size**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    def process(self, left, right):
        left_rectified = cv2.remap(
            left,
            self.left_map_x,
            self.left_map_y,
            cv2.INTER_LINEAR,
        )

        right_rectified = cv2.remap(
            right,
            self.right_map_x,
            self.right_map_y,
            cv2.INTER_LINEAR,
        )

        left_gray = cv2.cvtColor(
            left_rectified,
            cv2.COLOR_BGR2GRAY,
        )

        right_gray = cv2.cvtColor(
            right_rectified,
            cv2.COLOR_BGR2GRAY,
        )

        disparity = self.matcher.compute(
            left_gray,
            right_gray,
        ).astype(np.float32) / 16.0

        points_3d = cv2.reprojectImageTo3D(
            disparity,
            self.Q,
        )

        depth = points_3d[:, :, 2]

        valid = (
            np.isfinite(depth)
            & (disparity > 0)
            & (depth > 0)
        )

        depth[~valid] = 0

        return left_rectified, disparity, depth
