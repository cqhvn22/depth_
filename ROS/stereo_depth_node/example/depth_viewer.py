#!/usr/bin/env python3

import cv2

import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float32


class DepthViewer(Node):

    def __init__(self):
        super().__init__("depth_viewer")

        self.bridge = CvBridge()

        self.depth = None
        self.disparity = None
        self.center_depth = None

        self.create_subscription(
            Image,
            "/stereo/depth/image_raw",
            self.depth_callback,
            10,
        )

        self.create_subscription(
            Image,
            "/stereo/disparity/color",
            self.disparity_callback,
            10,
        )

        self.create_subscription(
            Float32,
            "/stereo/depth/center",
            self.center_callback,
            10,
        )

        self.timer = self.create_timer(
            0.03,
            self.display,
        )

    def depth_callback(self, msg):
        self.depth = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding="32FC1",
        )

    def disparity_callback(self, msg):
        self.disparity = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding="bgr8",
        )

    def center_callback(self, msg):
        self.center_depth = msg.data

    def display(self):

        if self.disparity is not None:

            image = self.disparity.copy()

            if self.center_depth > 0:

                text = f"Depth: {self.center_depth:.2f} m"

            else:

                text = "Depth: Invalid"

            cv2.putText(
                image,
                text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Disparity", image)

        if self.depth is not None:

            depth = self.depth.copy()

            valid = depth > 0

            display = depth.copy()

            if valid.any():

                minimum = depth[valid].min()
                maximum = depth[valid].max()

                display = (display - minimum) / (maximum - minimum + 1e-6)
                display = (display * 255).astype("uint8")
                display = cv2.applyColorMap(
                    display,
                    cv2.COLORMAP_TURBO,
                )

            cv2.imshow("Depth", display)

        cv2.waitKey(1)


def main():

    rclpy.init()

    node = DepthViewer()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    cv2.destroyAllWindows()

    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()