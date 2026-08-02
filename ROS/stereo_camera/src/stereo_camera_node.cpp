#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <camera_calibration_parsers/parse.hpp>
#include <opencv2/core.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/header.hpp>

#include "stereo_camera_ros2/stereo_camera.hpp"

namespace stereo_camera_ros2
{

class StereoCameraNode final : public rclcpp::Node
{
public:
  StereoCameraNode()
  : Node("stereo_camera_node")
  {
    declareParameters();
    readParameters();
    validateParameters();

    const auto qos = rclcpp::SensorDataQoS();

    left_image_pub_ = create_publisher<sensor_msgs::msg::Image>(
      topic_prefix_ + "/left/image_raw", qos);
    right_image_pub_ = create_publisher<sensor_msgs::msg::Image>(
      topic_prefix_ + "/right/image_raw", qos);
    left_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>(
      topic_prefix_ + "/left/camera_info", qos);
    right_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>(
      topic_prefix_ + "/right/camera_info", qos);

    if (publish_sync_delta_) {
      sync_delta_pub_ = create_publisher<std_msgs::msg::Float32>(
        topic_prefix_ + "/sync_delta_ms", qos);
    }

    left_camera_info_ = loadCameraInfo(
      left_camera_info_url_, left_camera_name_, output_width_, output_height_);
    right_camera_info_ = loadCameraInfo(
      right_camera_info_url_, right_camera_name_, output_width_, output_height_);

    camera_ = std::make_unique<StereoCamera>(
      left_sensor_id_,
      right_sensor_id_,
      sensor_mode_,
      capture_width_,
      capture_height_,
      output_width_,
      output_height_,
      framerate_,
      flip_method_);

    if (!camera_->open()) {
      throw std::runtime_error(
              "Could not open both CSI cameras. Check sensor IDs, GStreamer, and nvargus-daemon.");
    }
    if (!camera_->start()) {
      throw std::runtime_error("Could not start the stereo capture thread.");
    }

    const auto period = std::chrono::nanoseconds(
      static_cast<std::int64_t>(1'000'000'000LL / framerate_));
    publish_timer_ = create_wall_timer(
      period, std::bind(&StereoCameraNode::publishPair, this));

    RCLCPP_INFO(
      get_logger(),
      "Stereo camera started: %d FPS, %dx%d, topics under /%s",
      framerate_, output_width_, output_height_, topic_prefix_.c_str());
  }

  ~StereoCameraNode() override
  {
    if (camera_) {
      camera_->release();
    }
  }

private:
  void declareParameters()
  {
    declare_parameter<int>("left_sensor_id", 0);
    declare_parameter<int>("right_sensor_id", 1);
    declare_parameter<int>("sensor_mode", 2);
    declare_parameter<int>("capture_width", 960);
    declare_parameter<int>("capture_height", 540);
    declare_parameter<int>("output_width", 960);
    declare_parameter<int>("output_height", 540);
    declare_parameter<int>("framerate", 30);
    declare_parameter<int>("flip_method", 0);

    declare_parameter<std::string>("topic_prefix", "stereo");
    declare_parameter<std::string>("left_frame_id", "stereo_left_optical_frame");
    declare_parameter<std::string>("right_frame_id", "stereo_right_optical_frame");
    declare_parameter<std::string>("left_camera_name", "imx219_left_960x540");
    declare_parameter<std::string>("right_camera_name", "imx219_right_960x540");
    declare_parameter<std::string>("left_camera_info_url", "");
    declare_parameter<std::string>("right_camera_info_url", "");
    declare_parameter<bool>("publish_sync_delta", true);
  }

  void readParameters()
  {
    left_sensor_id_ = get_parameter("left_sensor_id").as_int();
    right_sensor_id_ = get_parameter("right_sensor_id").as_int();
    sensor_mode_ = get_parameter("sensor_mode").as_int();
    capture_width_ = get_parameter("capture_width").as_int();
    capture_height_ = get_parameter("capture_height").as_int();
    output_width_ = get_parameter("output_width").as_int();
    output_height_ = get_parameter("output_height").as_int();
    framerate_ = get_parameter("framerate").as_int();
    flip_method_ = get_parameter("flip_method").as_int();

    topic_prefix_ = trimSlashes(get_parameter("topic_prefix").as_string());
    left_frame_id_ = get_parameter("left_frame_id").as_string();
    right_frame_id_ = get_parameter("right_frame_id").as_string();
    left_camera_name_ = get_parameter("left_camera_name").as_string();
    right_camera_name_ = get_parameter("right_camera_name").as_string();
    left_camera_info_url_ = get_parameter("left_camera_info_url").as_string();
    right_camera_info_url_ = get_parameter("right_camera_info_url").as_string();
    publish_sync_delta_ = get_parameter("publish_sync_delta").as_bool();
  }

  void validateParameters() const
  {
    if (left_sensor_id_ == right_sensor_id_) {
      throw std::invalid_argument("left_sensor_id and right_sensor_id must be different.");
    }
    if (
      capture_width_ <= 0 || capture_height_ <= 0 ||
      output_width_ <= 0 || output_height_ <= 0)
    {
      throw std::invalid_argument("Camera image dimensions must be greater than zero.");
    }
    if (framerate_ <= 0) {
      throw std::invalid_argument("framerate must be greater than zero.");
    }
    if (topic_prefix_.empty()) {
      throw std::invalid_argument("topic_prefix cannot be empty.");
    }
  }

  static std::string trimSlashes(std::string value)
  {
    while (!value.empty() && value.front() == '/') {
      value.erase(value.begin());
    }
    while (!value.empty() && value.back() == '/') {
      value.pop_back();
    }
    return value;
  }

  static std::string resolveCalibrationPath(const std::string & url)
  {
    constexpr const char * file_prefix = "file://";
    constexpr const char * package_prefix = "package://";

    if (url.rfind(file_prefix, 0) == 0) {
      return url.substr(std::strlen(file_prefix));
    }

    if (url.rfind(package_prefix, 0) == 0) {
      const std::string package_path = url.substr(std::strlen(package_prefix));
      const auto separator = package_path.find('/');
      if (separator == std::string::npos) {
        throw std::invalid_argument(
                "package:// calibration URL must include a path inside the package: " + url);
      }

      const std::string package_name = package_path.substr(0, separator);
      const std::string relative_path = package_path.substr(separator + 1);
      return (
        std::filesystem::path(
          ament_index_cpp::get_package_share_directory(package_name)) /
        relative_path).string();
    }

    // Also accept a normal absolute or relative filesystem path.
    return url;
  }

  sensor_msgs::msg::CameraInfo loadCameraInfo(
    const std::string & url,
    const std::string & expected_camera_name,
    const int width,
    const int height)
  {
    sensor_msgs::msg::CameraInfo info;
    info.width = static_cast<std::uint32_t>(width);
    info.height = static_cast<std::uint32_t>(height);

    if (url.empty()) {
      RCLCPP_WARN(
        get_logger(),
        "No calibration URL configured for %s; publishing uncalibrated CameraInfo.",
        expected_camera_name.c_str());
      return info;
    }

    const std::string path = resolveCalibrationPath(url);
    std::string loaded_camera_name;
    sensor_msgs::msg::CameraInfo loaded_info;

    if (!camera_calibration_parsers::readCalibration(
        path, loaded_camera_name, loaded_info))
    {
      RCLCPP_ERROR(
        get_logger(),
        "Could not load camera calibration for %s from %s; publishing uncalibrated CameraInfo.",
        expected_camera_name.c_str(), path.c_str());
      return info;
    }

    if (loaded_info.width != static_cast<std::uint32_t>(width) ||
      loaded_info.height != static_cast<std::uint32_t>(height))
    {
      RCLCPP_WARN(
        get_logger(),
        "Calibration for %s is %ux%u, but the output stream is %dx%d.",
        expected_camera_name.c_str(), loaded_info.width, loaded_info.height, width, height);
    }

    if (!loaded_camera_name.empty() && loaded_camera_name != expected_camera_name) {
      RCLCPP_WARN(
        get_logger(),
        "Calibration camera name is '%s', while the configured name is '%s'.",
        loaded_camera_name.c_str(), expected_camera_name.c_str());
    }

    RCLCPP_INFO(
      get_logger(), "Loaded calibration for %s from %s",
      expected_camera_name.c_str(), path.c_str());
    return loaded_info;
  }

  static std::unique_ptr<sensor_msgs::msg::Image> makeImageMessage(
    const cv::Mat & image,
    const std_msgs::msg::Header & header)
  {
    if (image.empty() || image.type() != CV_8UC3) {
      throw std::runtime_error("Expected a non-empty CV_8UC3 BGR image from OpenCV.");
    }

    auto message = std::make_unique<sensor_msgs::msg::Image>();
    message->header = header;
    message->height = static_cast<std::uint32_t>(image.rows);
    message->width = static_cast<std::uint32_t>(image.cols);
    message->encoding = sensor_msgs::image_encodings::BGR8;
    message->is_bigendian = false;
    message->step = static_cast<std::uint32_t>(image.cols * image.elemSize());
    message->data.resize(static_cast<std::size_t>(message->step) * message->height);

    if (image.isContinuous()) {
      std::memcpy(message->data.data(), image.data, message->data.size());
    } else {
      for (int row = 0; row < image.rows; ++row) {
        std::memcpy(
          message->data.data() + static_cast<std::size_t>(row) * message->step,
          image.ptr(row),
          message->step);
      }
    }

    return message;
  }

  void publishPair()
  {
    StereoFrame pair;
    if (!camera_->read(pair, last_sequence_)) {
      return;
    }
    last_sequence_ = pair.sequence;

    // OpenCV/Argus does not expose a reliable common hardware acquisition
    // timestamp here. Give both images exactly the same ROS timestamp so a
    // stereo subscriber receives one unambiguous pair.
    const auto stamp = now();

    std_msgs::msg::Header left_header;
    left_header.stamp = stamp;
    left_header.frame_id = left_frame_id_;

    std_msgs::msg::Header right_header;
    right_header.stamp = stamp;
    right_header.frame_id = right_frame_id_;

    try {
      auto left_image = makeImageMessage(pair.left, left_header);
      auto right_image = makeImageMessage(pair.right, right_header);

      auto left_info = left_camera_info_;
      left_info.header = left_header;
      auto right_info = right_camera_info_;
      right_info.header = right_header;

      left_image_pub_->publish(std::move(left_image));
      left_info_pub_->publish(left_info);
      right_image_pub_->publish(std::move(right_image));
      right_info_pub_->publish(right_info);

      if (sync_delta_pub_) {
        std_msgs::msg::Float32 delta;
        delta.data = static_cast<float>(pair.delta_ms);
        sync_delta_pub_->publish(delta);
      }
    } catch (const std::exception & error) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Failed to publish stereo pair: %s", error.what());
    }
  }

  int left_sensor_id_{0};
  int right_sensor_id_{1};
  int sensor_mode_{2};
  int capture_width_{960};
  int capture_height_{540};
  int output_width_{960};
  int output_height_{540};
  int framerate_{30};
  int flip_method_{0};

  std::string topic_prefix_;
  std::string left_frame_id_;
  std::string right_frame_id_;
  std::string left_camera_name_;
  std::string right_camera_name_;
  std::string left_camera_info_url_;
  std::string right_camera_info_url_;
  bool publish_sync_delta_{true};

  sensor_msgs::msg::CameraInfo left_camera_info_;
  sensor_msgs::msg::CameraInfo right_camera_info_;

  std::unique_ptr<StereoCamera> camera_;
  std::optional<std::uint64_t> last_sequence_;

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr left_image_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr right_image_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr left_info_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr right_info_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr sync_delta_pub_;
  rclcpp::TimerBase::SharedPtr publish_timer_;
};

}  // namespace stereo_camera_ros2

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  try {
    rclcpp::spin(std::make_shared<stereo_camera_ros2::StereoCameraNode>());
  } catch (const std::exception & error) {
    RCLCPP_FATAL(
      rclcpp::get_logger("stereo_camera_node"),
      "Stereo camera node stopped: %s", error.what());
  }

  rclcpp::shutdown();
  return 0;
}
