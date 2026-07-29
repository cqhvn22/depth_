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
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame

    def release(self):
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
        if self.read_thread is not None:
            self.read_thread.join()



left_camera = Camera()
left_camera.open(0)
left_camera.start()

right_camera = Camera()
right_camera.open(1)
right_camera.start()

#cv2.namedWindow("CSI Cameras", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("Cam Left", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("Cam Right", cv2.WINDOW_AUTOSIZE)

# Check if both cameras opened successfully
if not left_camera.video_capture.isOpened() or not right_camera.video_capture.isOpened():
    print("Unable to open cameras")
    sys.exit(0)

while cv2.getWindowProperty("Cam Left", 0) >= 0:
    # Read frames from both cameras
    _, left_image = left_camera.read()
    _, right_image = right_camera.read()


    #cv2.imshow("CSI Cameras", camera_images)
    cv2.imshow("Cam Left", left_image)
    cv2.imshow("Cam Right", right_image)
    

    # Check for the ESC key to exit
    keycode = cv2.waitKey(30) & 0xFF
    if keycode == 27:
        break

    # Stop and release both cameras
left_camera.stop()
left_camera.release()
right_camera.stop()
right_camera.release()
cv2.destroyAllWindows()
