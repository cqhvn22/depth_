#!/usr/bin/env python3

import cv2
import message_filters
import rclpy

from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from sensor_msgs.msg import Imu


class StereoIMUTest(Node):

    def __init__(self):
        super().__init__("stereo_imu_test")

        self.bridge = CvBridge()

        # Stereo subscribers
        self.left_sub = message_filters.Subscriber(
            self,
            Image,
            "/stereo/left/image_raw",
            qos_profile=qos_profile_sensor_data,
        )

        self.right_sub = message_filters.Subscriber(
            self,
            Image,
            "/stereo/right/image_raw",
            qos_profile=qos_profile_sensor_data,
        )

        self.sync = message_filters.TimeSynchronizer(
            [self.left_sub, self.right_sub],
            queue_size=5,
        )

        self.sync.registerCallback(self.stereo_callback)

        # IMU subscriber
        self.imu_sub = self.create_subscription(
            Imu,
            "/imu/data",
            self.imu_callback,
            qos_profile_sensor_data,
        )

        self.get_logger().info("Waiting for stereo images and IMU...")

    ############################################################
    # Stereo callback
    ############################################################

    def stereo_callback(self, left_msg, right_msg):

        left = self.bridge.imgmsg_to_cv2(
            left_msg,
            desired_encoding="bgr8",
        )

        right = self.bridge.imgmsg_to_cv2(
            right_msg,
            desired_encoding="bgr8",
        )

        cv2.imshow("Left", left)
        cv2.imshow("Right", right)

        cv2.waitKey(1)

    ############################################################
    # IMU callback
    ############################################################

    def imu_callback(self, msg):

        ax = msg.linear_acceleration.x
        ay = msg.linear_acceleration.y
        az = msg.linear_acceleration.z

        gx = msg.angular_velocity.x
        gy = msg.angular_velocity.y
        gz = msg.angular_velocity.z

        qx = msg.orientation.x
        qy = msg.orientation.y
        qz = msg.orientation.z
        qw = msg.orientation.w

        print(
            "\n==================== IMU ===================="
        )

        print(
            f"Accel : {ax:7.3f}  {ay:7.3f}  {az:7.3f}  m/s²"
        )

        print(
            f"Gyro  : {gx:7.3f}  {gy:7.3f}  {gz:7.3f}  rad/s"
        )

        print(
            f"Quat  : {qx:6.3f} {qy:6.3f} {qz:6.3f} {qw:6.3f}"
        )


def main(args=None):

    rclpy.init(args=args)

    node = StereoIMUTest()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    cv2.destroyAllWindows()

    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()