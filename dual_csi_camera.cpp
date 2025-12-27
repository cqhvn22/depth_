#include <opencv4/opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <atomic>
#include <iostream>

class Camera {
public:
    Camera()
        : grabbed(false), running(false) {}

    ~Camera() {
        stop();
        release();
    }

    void open(int sensor_id = 0) {
        int sensor_mode = 3;          // 1280x720 @ ~60 fps
        int capture_width = 1280;
        int capture_height = 720;
        int display_width = 1280;
        int display_height = 720;
        int framerate = 20;
        int flip_method = 0;

        std::string gstreamer_pipeline =
            "nvarguscamerasrc sensor-id=" + std::to_string(sensor_id) +
            " sensor-mode=" + std::to_string(sensor_mode) + " ! "
            "video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) +
            ", height=(int)" + std::to_string(capture_height) +
            ", format=(string)NV12, framerate=(fraction)" + std::to_string(framerate) + "/1 ! "
            "nvvidconv flip-method=" + std::to_string(flip_method) + " ! "
            "video/x-raw, width=(int)" + std::to_string(display_width) +
            ", height=(int)" + std::to_string(display_height) +
            ", format=(string)BGRx ! "
            "videoconvert ! video/x-raw, format=(string)BGR ! appsink";

        video_capture.open(gstreamer_pipeline, cv::CAP_GSTREAMER);

        if (!video_capture.isOpened()) {
            std::cerr << "Unable to open camera " << sensor_id << std::endl;
            std::cerr << "Pipeline: " << gstreamer_pipeline << std::endl;
            return;
        }

        video_capture.read(frame);
        grabbed = !frame.empty();
    }

    void start() {
        if (running) {
            std::cout << "Video capturing already running\n";
            return;
        }
        running = true;
        read_thread = std::thread(&Camera::updateCamera, this);
    }

    void stop() {
        running = false;
        if (read_thread.joinable())
            read_thread.join();
    }

    bool read(cv::Mat &output_frame) {
        std::lock_guard<std::mutex> lock(read_lock);
        if (!grabbed || frame.empty())
            return false;
        frame.copyTo(output_frame);
        return true;
    }

    bool isOpened() const {
        return video_capture.isOpened();
    }

    void release() {
        if (video_capture.isOpened())
            video_capture.release();
    }

private:
    void updateCamera() {
        while (running) {
            cv::Mat temp;
            bool ok = video_capture.read(temp);
            {
                std::lock_guard<std::mutex> lock(read_lock);
                grabbed = ok;
                if (ok)
                    frame = temp;
            }
        }
    }

    cv::VideoCapture video_capture;
    cv::Mat frame;
    bool grabbed;

    std::thread read_thread;
    std::mutex read_lock;
    std::atomic<bool> running;
};

int main() {
    Camera left_camera;
    Camera right_camera;

    left_camera.open(0);
    right_camera.open(1);

    if (!left_camera.isOpened() || !right_camera.isOpened()) {
        std::cerr << "Unable to open both cameras\n";
        return -1;
    }

    left_camera.start();
    right_camera.start();

    cv::namedWindow("Cam Left", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Cam Right", cv::WINDOW_AUTOSIZE);

    while (true) {
        cv::Mat left_image, right_image;

        left_camera.read(left_image);
        right_camera.read(right_image);

        if (!left_image.empty())
            cv::imshow("Cam Left", left_image);
        if (!right_image.empty())
            cv::imshow("Cam Right", right_image);

        int key = cv::waitKey(30);
        if (key == 27)  // ESC
            break;
    }

    left_camera.stop();
    right_camera.stop();

    left_camera.release();
    right_camera.release();

    cv::destroyAllWindows();
    return 0;
}
