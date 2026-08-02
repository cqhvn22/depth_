#!/usr/bin/env python3

"""Independent ROS 2 stereo-depth subscriber node.

This node contains no camera capture code. It only subscribes to left and right
sensor_msgs/Image topics, synchronizes them, computes rectified disparity and
metric depth using a stereo calibration .npz file, and publishes the results.
"""

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Float32


class StereoDepth:
    """StereoSGBM + right matcher + WLS depth estimator."""

    REQUIRED_CALIBRATION_KEYS = (
        "left_map_x",
        "left_map_y",
        "right_map_x",
        "right_map_y",
        "Q",
    )

    def __init__(self, calibration_file: str, downscale: float = 0.5) -> None:
        calibration_path = Path(calibration_file).expanduser()

        if not calibration_path.is_file():
            raise FileNotFoundError(
                f"Calibration file does not exist: {calibration_path}"
            )

        if not 0.0 < downscale <= 1.0:
            raise ValueError("downscale must be greater than 0 and at most 1.0")

        if not hasattr(cv2, "ximgproc"):
            raise RuntimeError(
                "OpenCV ximgproc is unavailable. Install/build OpenCV contrib "
                "with Python bindings."
            )

        with np.load(str(calibration_path), allow_pickle=False) as data:
            missing = [
                key for key in self.REQUIRED_CALIBRATION_KEYS if key not in data
            ]
            if missing:
                raise KeyError(
                    "Calibration file is missing keys: " + ", ".join(missing)
                )

            self.left_map_x = data["left_map_x"].copy()
            self.left_map_y = data["left_map_y"].copy()
            self.right_map_x = data["right_map_x"].copy()
            self.right_map_y = data["right_map_y"].copy()
            self.q_matrix = data["Q"].copy()

        if self.left_map_x.shape != self.left_map_y.shape:
            raise ValueError("Left calibration maps have different shapes")
        if self.right_map_x.shape != self.right_map_y.shape:
            raise ValueError("Right calibration maps have different shapes")
        if self.left_map_x.shape[:2] != self.right_map_x.shape[:2]:
            raise ValueError("Left and right calibration maps have different sizes")
        if self.q_matrix.shape != (4, 4):
            raise ValueError(f"Q must have shape (4, 4), got {self.q_matrix.shape}")

        self.image_height, self.image_width = self.left_map_x.shape[:2]
        self.downscale = float(downscale)

        block_size = 7
        num_disparities = 128 if self.downscale >= 1.0 else 96

        self.left_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=8 * block_size**2,
            P2=32 * block_size**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

        self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)
        self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(
            matcher_left=self.left_matcher
        )
        self.wls_filter.setLambda(8000.0)
        self.wls_filter.setSigmaColor(1.5)

    def process(
        self, left_bgr: np.ndarray, right_bgr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return rectified left/right images, disparity, and raw-unit depth."""

        self._validate_input_image(left_bgr, "left")
        self._validate_input_image(right_bgr, "right")

        left_rectified = cv2.remap(
            left_bgr,
            self.left_map_x,
            self.left_map_y,
            cv2.INTER_LINEAR,
        )
        right_rectified = cv2.remap(
            right_bgr,
            self.right_map_x,
            self.right_map_y,
            cv2.INTER_LINEAR,
        )

        left_gray = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)

        if self.downscale != 1.0:
            small_left = cv2.resize(
                left_gray,
                None,
                fx=self.downscale,
                fy=self.downscale,
                interpolation=cv2.INTER_AREA,
            )
            small_right = cv2.resize(
                right_gray,
                None,
                fx=self.downscale,
                fy=self.downscale,
                interpolation=cv2.INTER_AREA,
            )
        else:
            small_left = left_gray
            small_right = right_gray

        disparity_left = self.left_matcher.compute(small_left, small_right)
        disparity_right = self.right_matcher.compute(small_right, small_left)

        filtered_disparity = self.wls_filter.filter(
            disparity_left,
            small_left,
            disparity_map_right=disparity_right,
        )
        disparity = filtered_disparity.astype(np.float32) / 16.0

        if self.downscale != 1.0:
            disparity = cv2.resize(
                disparity,
                (left_gray.shape[1], left_gray.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            ) / self.downscale

        points_3d = cv2.reprojectImageTo3D(disparity, self.q_matrix)
        depth = points_3d[:, :, 2].astype(np.float32, copy=False)

        valid = np.isfinite(depth) & (disparity > 0.0) & (depth > 0.0)
        depth[~valid] = 0.0

        return left_rectified, right_rectified, disparity, depth

    def _validate_input_image(self, image: np.ndarray, side: str) -> None:
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"{side} image must be a three-channel BGR image")

        height, width = image.shape[:2]
        if (height, width) != (self.image_height, self.image_width):
            raise ValueError(
                f"{side} image is {width}x{height}, but calibration maps are "
                f"{self.image_width}x{self.image_height}"
            )


class StereoDepthNode(Node):
    """Subscribe to stereo image topics and publish depth products."""

    def __init__(self) -> None:
        super().__init__("stereo_depth_node")

        self.declare_parameter("left_topic", "/stereo/left/image_raw")
        self.declare_parameter("right_topic", "/stereo/right/image_raw")
        self.declare_parameter("calibration_file", "")
        self.declare_parameter("calibration_unit", "mm")
        self.declare_parameter("downscale", 0.5)
        self.declare_parameter("sync_queue_size", 10)
        self.declare_parameter("sync_slop", 0.03)

        self.declare_parameter("depth_topic", "/stereo/depth/image_raw")
        self.declare_parameter(
            "disparity_topic", "/stereo/disparity/image_raw"
        )
        self.declare_parameter(
            "disparity_color_topic", "/stereo/disparity/color"
        )
        self.declare_parameter("left_rect_topic", "/stereo/left/image_rect")
        self.declare_parameter("right_rect_topic", "/stereo/right/image_rect")
        self.declare_parameter("center_depth_topic", "/stereo/depth/center")
        self.declare_parameter("sync_delta_topic", "/stereo/sync/delta_ms")

        left_topic = self.get_parameter("left_topic").value
        right_topic = self.get_parameter("right_topic").value
        calibration_file = self.get_parameter("calibration_file").value
        self.calibration_unit = str(
            self.get_parameter("calibration_unit").value
        ).lower()
        downscale = float(self.get_parameter("downscale").value)
        queue_size = int(self.get_parameter("sync_queue_size").value)
        sync_slop = float(self.get_parameter("sync_slop").value)

        if not calibration_file:
            raise RuntimeError(
                "Parameter 'calibration_file' is empty. Pass the absolute path "
                "to stereo_calibration.npz."
            )
        if self.calibration_unit not in ("m", "cm", "mm"):
            raise ValueError("calibration_unit must be one of: m, cm, mm")
        if queue_size < 1:
            raise ValueError("sync_queue_size must be at least 1")
        if sync_slop < 0.0:
            raise ValueError("sync_slop cannot be negative")

        self.bridge = CvBridge()
        self.depth_estimator = StereoDepth(calibration_file, downscale)

        self.depth_pub = self.create_publisher(
            Image, self.get_parameter("depth_topic").value, 10
        )
        self.disparity_pub = self.create_publisher(
            Image, self.get_parameter("disparity_topic").value, 10
        )
        self.disparity_color_pub = self.create_publisher(
            Image, self.get_parameter("disparity_color_topic").value, 10
        )
        self.left_rect_pub = self.create_publisher(
            Image, self.get_parameter("left_rect_topic").value, 10
        )
        self.right_rect_pub = self.create_publisher(
            Image, self.get_parameter("right_rect_topic").value, 10
        )
        self.center_depth_pub = self.create_publisher(
            Float32, self.get_parameter("center_depth_topic").value, 10
        )
        self.sync_delta_pub = self.create_publisher(
            Float32, self.get_parameter("sync_delta_topic").value, 10
        )

        # These subscribers receive only ROS image messages. There is no camera
        # opening, GStreamer pipeline, sensor ID, or camera package dependency.
        self.left_subscriber = Subscriber(
            self,
            Image,
            left_topic,
            qos_profile=qos_profile_sensor_data,
        )
        self.right_subscriber = Subscriber(
            self,
            Image,
            right_topic,
            qos_profile=qos_profile_sensor_data,
        )

        self.synchronizer = ApproximateTimeSynchronizer(
            [self.left_subscriber, self.right_subscriber],
            queue_size=queue_size,
            slop=sync_slop,
            allow_headerless=False,
        )
        self.synchronizer.registerCallback(self.stereo_callback)

        self.get_logger().info("Independent stereo depth node started")
        self.get_logger().info(f"Left input:  {left_topic}")
        self.get_logger().info(f"Right input: {right_topic}")
        self.get_logger().info(f"Calibration: {calibration_file}")
        self.get_logger().info(
            "Expected image size: "
            f"{self.depth_estimator.image_width}x"
            f"{self.depth_estimator.image_height}"
        )
        self.get_logger().info(
            f"Published depth unit: meters; downscale: {downscale}"
        )

    def stereo_callback(self, left_msg: Image, right_msg: Image) -> None:
        try:
            left_bgr = self.bridge.imgmsg_to_cv2(
                left_msg, desired_encoding="bgr8"
            )
            right_bgr = self.bridge.imgmsg_to_cv2(
                right_msg, desired_encoding="bgr8"
            )

            (
                left_rectified,
                right_rectified,
                disparity,
                depth_calibration_unit,
            ) = self.depth_estimator.process(left_bgr, right_bgr)

            depth_m = self._depth_to_meters(depth_calibration_unit)
            center_depth_m = self._get_center_depth(depth_m)

            self._publish_image(
                self.depth_pub,
                depth_m.astype(np.float32, copy=False),
                "32FC1",
                left_msg,
            )
            self._publish_image(
                self.disparity_pub,
                disparity.astype(np.float32, copy=False),
                "32FC1",
                left_msg,
            )

            if self.left_rect_pub.get_subscription_count() > 0:
                self._publish_image(
                    self.left_rect_pub,
                    left_rectified,
                    "bgr8",
                    left_msg,
                )

            if self.right_rect_pub.get_subscription_count() > 0:
                self._publish_image(
                    self.right_rect_pub,
                    right_rectified,
                    "bgr8",
                    right_msg,
                )

            if self.disparity_color_pub.get_subscription_count() > 0:
                disparity_color = self._disparity_to_color(disparity)
                self._publish_image(
                    self.disparity_color_pub,
                    disparity_color,
                    "bgr8",
                    left_msg,
                )

            center_message = Float32()
            center_message.data = (
                float(center_depth_m) if center_depth_m is not None else 0.0
            )
            self.center_depth_pub.publish(center_message)

            sync_message = Float32()
            sync_message.data = float(
                abs(self._stamp_to_nanoseconds(left_msg)
                    - self._stamp_to_nanoseconds(right_msg))
                / 1_000_000.0
            )
            self.sync_delta_pub.publish(sync_message)

        except CvBridgeError as error:
            self.get_logger().error(f"cv_bridge conversion failed: {error}")
        except (ValueError, cv2.error) as error:
            self.get_logger().error(f"Depth processing failed: {error}")
        except Exception as error:  # Keep the ROS process alive on bad frames.
            self.get_logger().error(
                f"Unexpected stereo callback error: {type(error).__name__}: "
                f"{error}"
            )

    def _depth_to_meters(self, depth: np.ndarray) -> np.ndarray:
        if self.calibration_unit == "mm":
            return depth / 1000.0
        if self.calibration_unit == "cm":
            return depth / 100.0
        return depth.copy()

    @staticmethod
    def _get_center_depth(depth_m: np.ndarray) -> Optional[float]:
        height, width = depth_m.shape
        center_x = width // 2
        center_y = height // 2

        x0 = max(0, center_x - 5)
        x1 = min(width, center_x + 6)
        y0 = max(0, center_y - 5)
        y1 = min(height, center_y + 6)

        region = depth_m[y0:y1, x0:x1]
        valid_depths = region[np.isfinite(region) & (region > 0.0)]

        if valid_depths.size == 0:
            return None
        return float(np.median(valid_depths))

    @staticmethod
    def _disparity_to_color(disparity: np.ndarray) -> np.ndarray:
        valid = np.isfinite(disparity) & (disparity > 0.0)
        display = np.zeros(disparity.shape, dtype=np.uint8)

        if np.any(valid):
            valid_disparity = disparity[valid]
            minimum = float(np.percentile(valid_disparity, 2))
            maximum = float(np.percentile(valid_disparity, 98))

            if maximum > minimum:
                normalized = np.clip(
                    (disparity - minimum) / (maximum - minimum),
                    0.0,
                    1.0,
                )
                display = (normalized * 255.0).astype(np.uint8)

        display[~valid] = 0
        return cv2.applyColorMap(display, cv2.COLORMAP_TURBO)

    def _publish_image(
        self,
        publisher,
        image: np.ndarray,
        encoding: str,
        source_message: Image,
    ) -> None:
        output = self.bridge.cv2_to_imgmsg(
            np.ascontiguousarray(image), encoding=encoding
        )
        output.header = source_message.header
        publisher.publish(output)

    @staticmethod
    def _stamp_to_nanoseconds(message: Image) -> int:
        return (
            int(message.header.stamp.sec) * 1_000_000_000
            + int(message.header.stamp.nanosec)
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None

    try:
        node = StereoDepthNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as error:
        if node is not None:
            node.get_logger().fatal(
                f"Stereo depth node stopped: {type(error).__name__}: {error}"
            )
        else:
            print(
                f"Stereo depth node failed to start: "
                f"{type(error).__name__}: {error}"
            )
        raise
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
