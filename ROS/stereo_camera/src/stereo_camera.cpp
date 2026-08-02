#include "stereo_camera_ros2/stereo_camera.hpp"

#include <chrono>
#include <cmath>
#include <sstream>
#include <utility>

namespace stereo_camera_ros2
{

StereoCamera::StereoCamera(
  const int left_sensor_id,
  const int right_sensor_id,
  const int sensor_mode,
  const int capture_width,
  const int capture_height,
  const int output_width,
  const int output_height,
  const int framerate,
  const int flip_method)
: left_sensor_id_(left_sensor_id),
  right_sensor_id_(right_sensor_id),
  sensor_mode_(sensor_mode),
  capture_width_(capture_width),
  capture_height_(capture_height),
  output_width_(output_width),
  output_height_(output_height),
  framerate_(framerate),
  flip_method_(flip_method)
{
}

StereoCamera::~StereoCamera()
{
  release();
}

std::string StereoCamera::makeGstreamerPipeline(const int sensor_id) const
{
  std::ostringstream pipeline;
  pipeline
    << "nvarguscamerasrc sensor-id=" << sensor_id << ' '
    << "sensor-mode=" << sensor_mode_ << " ! "
    << "video/x-raw(memory:NVMM), "
    << "width=(int)" << capture_width_ << ", "
    << "height=(int)" << capture_height_ << ", "
    << "format=(string)NV12, "
    << "framerate=(fraction)" << framerate_ << "/1 ! "
    << "nvvidconv flip-method=" << flip_method_ << " ! "
    << "video/x-raw, "
    << "width=(int)" << output_width_ << ", "
    << "height=(int)" << output_height_ << ", "
    << "format=(string)BGRx ! "
    << "videoconvert ! "
    << "video/x-raw, format=(string)BGR ! "
    << "appsink drop=true max-buffers=1 sync=false";

  return pipeline.str();
}

bool StereoCamera::open()
{
  if (isOpened()) {
    return true;
  }

  release();

  const std::string left_pipeline = makeGstreamerPipeline(left_sensor_id_);
  const std::string right_pipeline = makeGstreamerPipeline(right_sensor_id_);

  if (!left_capture_.open(left_pipeline, cv::CAP_GSTREAMER)) {
    release();
    return false;
  }

  if (!right_capture_.open(right_pipeline, cv::CAP_GSTREAMER)) {
    release();
    return false;
  }

  // Argus commonly needs several frames before exposure and white balance
  // settle. These frames are intentionally discarded.
  for (int i = 0; i < 5; ++i) {
    if (!capturePair()) {
      release();
      return false;
    }
  }

  return true;
}

bool StereoCamera::start()
{
  if (running_.load()) {
    return true;
  }

  if (!isOpened()) {
    return false;
  }

  running_.store(true);
  capture_thread_ = std::thread(&StereoCamera::captureLoop, this);
  return true;
}

void StereoCamera::stop()
{
  running_.store(false);
  if (capture_thread_.joinable()) {
    capture_thread_.join();
  }
}

void StereoCamera::release()
{
  stop();

  if (left_capture_.isOpened()) {
    left_capture_.release();
  }
  if (right_capture_.isOpened()) {
    right_capture_.release();
  }

  std::lock_guard<std::mutex> lock(frame_mutex_);
  latest_frame_.reset();
}

bool StereoCamera::isOpened() const
{
  return left_capture_.isOpened() && right_capture_.isOpened();
}

bool StereoCamera::read(
  StereoFrame & output,
  const std::optional<std::uint64_t> last_sequence) const
{
  std::lock_guard<std::mutex> lock(frame_mutex_);

  if (!latest_frame_.has_value()) {
    return false;
  }

  if (last_sequence.has_value() && latest_frame_->sequence == *last_sequence) {
    return false;
  }

  // Clone while holding the mutex. OpenCV matrices otherwise share the same
  // pixel buffer and could be replaced by the capture thread during publish.
  output.left = latest_frame_->left.clone();
  output.right = latest_frame_->right.clone();
  output.left_timestamp_seconds = latest_frame_->left_timestamp_seconds;
  output.right_timestamp_seconds = latest_frame_->right_timestamp_seconds;
  output.delta_ms = latest_frame_->delta_ms;
  output.sequence = latest_frame_->sequence;
  return true;
}

bool StereoCamera::capturePair()
{
  if (!isOpened()) {
    return false;
  }

  // Advance both pipelines before retrieving either image. This minimizes the
  // software-induced offset, although it is not hardware synchronization.
  const bool left_grabbed = left_capture_.grab();
  const bool right_grabbed = right_capture_.grab();
  if (!left_grabbed || !right_grabbed) {
    return false;
  }

  cv::Mat left_frame;
  cv::Mat right_frame;

  const bool left_ok = left_capture_.retrieve(left_frame);
  const auto left_time = std::chrono::steady_clock::now();

  const bool right_ok = right_capture_.retrieve(right_frame);
  const auto right_time = std::chrono::steady_clock::now();

  if (!left_ok || !right_ok || left_frame.empty() || right_frame.empty()) {
    return false;
  }

  const double left_seconds =
    std::chrono::duration<double>(left_time.time_since_epoch()).count();
  const double right_seconds =
    std::chrono::duration<double>(right_time.time_since_epoch()).count();

  StereoFrame next;
  next.left = std::move(left_frame);
  next.right = std::move(right_frame);
  next.left_timestamp_seconds = left_seconds;
  next.right_timestamp_seconds = right_seconds;
  next.delta_ms = std::abs(right_seconds - left_seconds) * 1000.0;

  {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    next.sequence = ++sequence_;
    latest_frame_ = std::move(next);
  }

  return true;
}

void StereoCamera::captureLoop()
{
  while (running_.load()) {
    if (!capturePair()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
  }
}

}  // namespace stereo_camera_ros2
