#ifndef STEREO_CAMERA_ROS2__STEREO_CAMERA_HPP_
#define STEREO_CAMERA_ROS2__STEREO_CAMERA_HPP_

#include <atomic>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

namespace stereo_camera_ros2
{

struct StereoFrame
{
  cv::Mat left;
  cv::Mat right;
  double left_timestamp_seconds{0.0};
  double right_timestamp_seconds{0.0};
  double delta_ms{0.0};
  std::uint64_t sequence{0};
};

class StereoCamera
{
public:
  StereoCamera(
    int left_sensor_id,
    int right_sensor_id,
    int sensor_mode,
    int capture_width,
    int capture_height,
    int output_width,
    int output_height,
    int framerate,
    int flip_method);

  ~StereoCamera();

  StereoCamera(const StereoCamera &) = delete;
  StereoCamera & operator=(const StereoCamera &) = delete;

  bool open();
  bool start();
  void stop();
  void release();

  [[nodiscard]] bool isOpened() const;

  // Returns a safe copy of the newest stereo pair. When last_sequence is
  // supplied, false is returned until a newer pair is available.
  bool read(StereoFrame & output, std::optional<std::uint64_t> last_sequence = std::nullopt) const;

private:
  [[nodiscard]] std::string makeGstreamerPipeline(int sensor_id) const;
  bool capturePair();
  void captureLoop();

  int left_sensor_id_;
  int right_sensor_id_;
  int sensor_mode_;
  int capture_width_;
  int capture_height_;
  int output_width_;
  int output_height_;
  int framerate_;
  int flip_method_;

  cv::VideoCapture left_capture_;
  cv::VideoCapture right_capture_;

  mutable std::mutex frame_mutex_;
  std::optional<StereoFrame> latest_frame_;
  std::uint64_t sequence_{0};

  std::atomic_bool running_{false};
  std::thread capture_thread_;
};

}  // namespace stereo_camera_ros2

#endif  // STEREO_CAMERA_ROS2__STEREO_CAMERA_HPP_
