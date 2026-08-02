#!/usr/bin/env python3

import cv2
import message_filters
import rclpy

from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class PythonStereoSubscriber(Node):
    def __init__(self) -> None:
        super().__init__('python_stereo_subscriber')

        self.bridge = CvBridge()

        self.left_sub = message_filters.Subscriber(
            self,
            Image,
            '/stereo/left/image_raw',
            qos_profile=qos_profile_sensor_data,
        )

        self.right_sub = message_filters.Subscriber(
            self,
            Image,
            '/stereo/right/image_raw',
            qos_profile=qos_profile_sensor_data,
        )

        self.synchronizer = message_filters.TimeSynchronizer(
            [self.left_sub, self.right_sub],
            queue_size=5,
        )

        self.synchronizer.registerCallback(self.on_pair)

        self.get_logger().info('Waiting for stereo image pairs...')

    def on_pair(self, left_msg: Image, right_msg: Image) -> None:
        try:
            left = self.bridge.imgmsg_to_cv2(
                left_msg,
                desired_encoding='bgr8',
            )

            right = self.bridge.imgmsg_to_cv2(
                right_msg,
                desired_encoding='bgr8',
            )

            cv2.imshow('Left camera', left)
            cv2.imshow('Right camera', right)
            cv2.waitKey(1)

        except Exception as error:
            self.get_logger().error(
                f'Failed to process stereo pair: {error}'
            )

    def destroy_node(self) -> None:
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)

    node = PythonStereoSubscriber()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
