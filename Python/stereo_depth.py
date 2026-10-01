import cv2
import numpy as np


class StereoDepth:
    def __init__(self, calibration_file, downscale=0.5):
        data = np.load(calibration_file)

        self.left_map_x = data["left_map_x"]
        self.left_map_y = data["left_map_y"]
        self.right_map_x = data["right_map_x"]
        self.right_map_y = data["right_map_y"]
        self.Q = data["Q"]

        self.downscale = downscale

        block_size = 7
        num_disp = 128 if downscale >= 1.0 else 96  # can drop further if needed

        self.matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disp,
            blockSize=block_size,
            P1=8 * block_size**2,
            P2=32 * block_size**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,  # back to fast mode
        )

        self.right_matcher = cv2.ximgproc.createRightMatcher(self.matcher)
        self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(
            matcher_left=self.matcher
        )
        self.wls_filter.setLambda(8000)
        self.wls_filter.setSigmaColor(1.5)

    def process(self, left, right):
        left_rectified = cv2.remap(
            left, self.left_map_x, self.left_map_y, cv2.INTER_LINEAR,
        )
        right_rectified = cv2.remap(
            right, self.right_map_x, self.right_map_y, cv2.INTER_LINEAR,
        )

        left_gray = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)

        if self.downscale != 1.0:
            small_left = cv2.resize(
                left_gray, None, fx=self.downscale, fy=self.downscale,
                interpolation=cv2.INTER_AREA,
            )
            small_right = cv2.resize(
                right_gray, None, fx=self.downscale, fy=self.downscale,
                interpolation=cv2.INTER_AREA,
            )
        else:
            small_left, small_right = left_gray, right_gray

        disp_left = self.matcher.compute(small_left, small_right)
        disp_right = self.right_matcher.compute(small_right, small_left)

        filtered = self.wls_filter.filter(
            disp_left, small_left, disparity_map_right=disp_right
        )

        disparity = filtered.astype(np.float32) / 16.0

        if self.downscale != 1.0:
            # Upscale disparity to full res; disparity values scale with image size
            disparity = cv2.resize(
                disparity,
                (left_gray.shape[1], left_gray.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            ) / self.downscale

        points_3d = cv2.reprojectImageTo3D(disparity, self.Q)
        depth = points_3d[:, :, 2]

        valid = (
            np.isfinite(depth)
            & (disparity > 0)
            & (depth > 0)
        )
        depth[~valid] = 0

        return left_rectified, disparity, depth