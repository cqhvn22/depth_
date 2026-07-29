import cv2
import threading
import numpy as np
import sys

# This class manages the camera capture for each camera
class Camera:

    def __init__(self):
        self.video_capture = None
        self.frame = None
        self.grabbed = False
        self.read_thread = None
        self.read_lock = threading.Lock()
        self.running = False

    def open(self, sensor_id=0):
        # GStreamer pipeline string for capturing video from the CSI camera
        sensor_mode = 2  
        capture_width = 1920/2
        capture_height = 1080/2
        display_width = 1920/2
        display_height = 1080/2
        framerate = 30
        flip_method = 2
        
        gstreamer_pipeline_string = (
            "nvarguscamerasrc sensor-id=%d sensor-mode=%d ! "
            "video/x-raw(memory:NVMM), "
            "width=(int)%d, height=(int)%d, "
            "format=(string)NV12, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            % (
                sensor_id,
                sensor_mode,
                capture_width,
                capture_height,
                framerate,
                flip_method,
                display_width,
                display_height,
            )
        )

        try:
            self.video_capture = cv2.VideoCapture(gstreamer_pipeline_string, cv2.CAP_GSTREAMER)
        except RuntimeError:
            self.video_capture = None
            print("Unable to open camera")
            print("Pipeline: " + gstreamer_pipeline_string)
            return

        # Grab the first frame to start capturing
        self.grabbed, self.frame = self.video_capture.read()

    def start(self):
        if self.running:
            print('Video capturing is already running')
            return
        if self.video_capture is not None:
            self.running = True
            self.read_thread = threading.Thread(target=self.updateCamera)
            self.read_thread.start()

    def stop(self):
        self.running = False
        if self.read_thread is not None:
            self.read_thread.join()

    def updateCamera(self):
        while self.running:
            try:
                grabbed, frame = self.video_capture.read()
                with self.read_lock:
                    self.grabbed = grabbed
                    self.frame = frame
            except RuntimeError:
                print("Error reading from camera")

    def read(self):
        with self.read_lock:
            frame = self.frame.copy() if self.frame is not None else None
            grabbed = self.grabbed
        return grabbed, frame

    def release(self):
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
        if self.read_thread is not None:
            self.read_thread.join()


class StereoDepthEstimator:
    def __init__(self, calib_file='stereo_calib.npz'):
        """Initialize stereo depth estimation with calibration parameters"""
        # Load calibration parameters
        try:
            calib_data = np.load(calib_file)
            self.K_left = calib_data['K_left']
            self.dist_left = calib_data['dist_left']
            self.K_right = calib_data['K_right']
            self.dist_right = calib_data['dist_right']
            self.R = calib_data['R']
            self.T = calib_data['T']
            self.img_size = tuple(calib_data['img_size'])
        except FileNotFoundError:
            print(f"Calibration file {calib_file} not found!")
            sys.exit(1)
        except KeyError as e:
            print(f"Missing calibration parameter: {e}")
            sys.exit(1)

        # Setup rectification
        self.setup_rectification()
        
        # Create SGBM matcher
        self.setup_sgbm_matcher()

    def setup_rectification(self):
        """Setup stereo rectification maps"""
        # Compute rectification transforms
        R1, R2, P1, P2, self.Q, _, _ = cv2.stereoRectify(
            self.K_left, self.dist_left,
            self.K_right, self.dist_right,
            self.img_size,
            self.R, self.T,
            alpha=0  # 0=cropped, 1=full image with black borders
        )

        # Create rectification maps
        self.left_map_x, self.left_map_y = cv2.initUndistortRectifyMap(
            self.K_left, self.dist_left, R1, P1, self.img_size, cv2.CV_32FC1)
        self.right_map_x, self.right_map_y = cv2.initUndistortRectifyMap(
            self.K_right, self.dist_right, R2, P2, self.img_size, cv2.CV_32FC1)

    def setup_sgbm_matcher(self):
        """Setup Semi-Global Block Matching parameters"""
        block_size = 5  # Must be odd
        min_disp = 0
        max_disp = 128  # Must be divisible by 16
        num_disp = max_disp - min_disp

        self.stereo = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=block_size,
            P1=8 * 3 * block_size**2,
            P2=32 * 3 * block_size**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

    def compute_depth(self, left_frame, right_frame):
        """Compute depth map from stereo pair"""
        # Rectify frames
        rectified_left = cv2.remap(left_frame, self.left_map_x, self.left_map_y, cv2.INTER_LINEAR)
        rectified_right = cv2.remap(right_frame, self.right_map_x, self.right_map_y, cv2.INTER_LINEAR)

        # Convert to grayscale
        gray_left = cv2.cvtColor(rectified_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(rectified_right, cv2.COLOR_BGR2GRAY)

        # Compute disparity
        disparity_raw = self.stereo.compute(gray_left, gray_right)
        disparity_map = disparity_raw.astype(np.float32) / 16.0

        # Compute depth map
        depth_map = cv2.reprojectImageTo3D(disparity_map, self.Q)
        depth_visual = depth_map[:, :, 2]

        return disparity_map, depth_visual


# Initialize cameras
left_camera = Camera()
left_camera.open(0)
left_camera.start()

right_camera = Camera()
right_camera.open(1)
right_camera.start()

# Initialize depth estimator
depth_estimator = StereoDepthEstimator('stereo_calib.npz')

# Create windows for display
cv2.namedWindow("Cam Left", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("Cam Right", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("Disparity", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("Depth", cv2.WINDOW_AUTOSIZE)

# Check if both cameras opened successfully
if (left_camera.video_capture is None or 
    right_camera.video_capture is None or 
    not left_camera.video_capture.isOpened() or 
    not right_camera.video_capture.isOpened()):
    print("Unable to open cameras")
    sys.exit(0)

# Main processing loop
while cv2.getWindowProperty("Cam Left", 0) >= 0:
    # Read frames from both cameras
    _, left_image = left_camera.read()
    _, right_image = right_camera.read()

    if left_image is None or right_image is None:
        continue

    # Compute depth and disparity
    try:
        disparity_map, depth_visual = depth_estimator.compute_depth(left_image, right_image)

        # Normalize for display
        disp_norm = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_norm = cv2.normalize(depth_visual, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # Display results
        cv2.imshow("Cam Left", left_image)
        cv2.imshow("Cam Right", right_image)
        cv2.imshow("Disparity", disp_norm)
        cv2.imshow("Depth", depth_norm)
    except Exception as e:
        print(f"Error computing depth: {e}")
        # Still show camera feeds even if depth computation fails
        cv2.imshow("Cam Left", left_image)
        cv2.imshow("Cam Right", right_image)

    # Check for the ESC key to exit
    keycode = cv2.waitKey(1) & 0xFF
    if keycode == 27:
        break

# Stop and release both cameras
left_camera.stop()
left_camera.release()
right_camera.stop()
right_camera.release()
cv2.destroyAllWindows()
