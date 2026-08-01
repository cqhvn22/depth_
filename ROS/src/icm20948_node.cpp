#include <array>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/magnetic_field.hpp"
#include "stereo_camera_ros2/ICM20948.h"

namespace
{
constexpr double kPi = 3.14159265358979323846;
constexpr double kDegreesToRadians = kPi / 180.0;
constexpr double kGravity = 9.80665;
constexpr double kMicroteslaToTesla = 1.0e-6;

std::array<double, 4> quaternion_from_rpy(double roll, double pitch, double yaw)
{
  const double cr = std::cos(roll * 0.5);
  const double sr = std::sin(roll * 0.5);
  const double cp = std::cos(pitch * 0.5);
  const double sp = std::sin(pitch * 0.5);
  const double cy = std::cos(yaw * 0.5);
  const double sy = std::sin(yaw * 0.5);

  return {
    sr * cp * cy - cr * sp * sy,
    cr * sp * cy + sr * cp * sy,
    cr * cp * sy - sr * sp * cy,
    cr * cp * cy + sr * sp * sy,
  };
}
}  // namespace

class Icm20948Node : public rclcpp::Node
{
public:
  Icm20948Node()
  : Node("icm20948_node")
  {
    const auto i2c_device = declare_parameter<std::string>("i2c_device", "/dev/i2c-7");
    frame_id_ = declare_parameter<std::string>("frame_id", "imu_link");
    const double publish_rate = declare_parameter<double>("publish_rate", 20.0);

    if (publish_rate <= 0.0) {
      throw std::invalid_argument("publish_rate must be greater than zero");
    }
    if (!imuSetI2CDevice(i2c_device.c_str())) {
      throw std::invalid_argument("Invalid I2C device path");
    }

    IMU_EN_SENSOR_TYPE sensor_type = IMU_EN_SENSOR_TYPE_NULL;
    imuInit(&sensor_type);
    if (sensor_type != IMU_EN_SENSOR_TYPE_ICM20948) {
      imuClose();
      throw std::runtime_error("ICM-20948 was not detected on " + i2c_device);
    }

    const auto qos = rclcpp::SensorDataQoS();
    imu_publisher_ = create_publisher<sensor_msgs::msg::Imu>("imu/data", qos);
    magnetic_publisher_ =
      create_publisher<sensor_msgs::msg::MagneticField>("imu/mag", qos);

    const auto period = std::chrono::duration<double>(1.0 / publish_rate);
    timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      std::bind(&Icm20948Node::publish_sample, this));

    RCLCPP_INFO(
      get_logger(), "ICM-20948 started on %s at %.1f Hz", i2c_device.c_str(), publish_rate);
  }

  ~Icm20948Node() override
  {
    imuClose();
  }

private:
  void publish_sample()
  {
    IMU_ST_ANGLES_DATA angles{};
    IMU_ST_SENSOR_DATA gyro{};
    IMU_ST_SENSOR_DATA accel{};
    IMU_ST_SENSOR_DATA magnetic{};
    imuDataGet(&angles, &gyro, &accel, &magnetic);

    const auto stamp = now();
    sensor_msgs::msg::Imu imu_msg;
    imu_msg.header.stamp = stamp;
    imu_msg.header.frame_id = frame_id_;

    const auto q = quaternion_from_rpy(
      angles.fRoll * kDegreesToRadians,
      angles.fPitch * kDegreesToRadians,
      angles.fYaw * kDegreesToRadians);
    imu_msg.orientation.x = q[0];
    imu_msg.orientation.y = q[1];
    imu_msg.orientation.z = q[2];
    imu_msg.orientation.w = q[3];

    // The vendor driver returns dps and g; sensor_msgs/Imu requires rad/s and m/s^2.
    imu_msg.angular_velocity.x = gyro.fX * kDegreesToRadians;
    imu_msg.angular_velocity.y = gyro.fY * kDegreesToRadians;
    imu_msg.angular_velocity.z = gyro.fZ * kDegreesToRadians;
    imu_msg.linear_acceleration.x = accel.fX * kGravity;
    imu_msg.linear_acceleration.y = accel.fY * kGravity;
    imu_msg.linear_acceleration.z = accel.fZ * kGravity;

    // Zero covariance means unknown. Replace these with measured values after calibration.
    imu_msg.orientation_covariance.fill(0.0);
    imu_msg.angular_velocity_covariance.fill(0.0);
    imu_msg.linear_acceleration_covariance.fill(0.0);

    sensor_msgs::msg::MagneticField magnetic_msg;
    magnetic_msg.header = imu_msg.header;
    magnetic_msg.magnetic_field.x = magnetic.fX * kMicroteslaToTesla;
    magnetic_msg.magnetic_field.y = magnetic.fY * kMicroteslaToTesla;
    magnetic_msg.magnetic_field.z = magnetic.fZ * kMicroteslaToTesla;
    magnetic_msg.magnetic_field_covariance.fill(0.0);

    imu_publisher_->publish(imu_msg);
    magnetic_publisher_->publish(magnetic_msg);
  }

  std::string frame_id_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::MagneticField>::SharedPtr magnetic_publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<Icm20948Node>());
  } catch (const std::exception & exception) {
    RCLCPP_FATAL(rclcpp::get_logger("icm20948_node"), "%s", exception.what());
  }
  rclcpp::shutdown();
  return 0;
}
