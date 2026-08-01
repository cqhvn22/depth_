import cv2
import threading
import time
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class StereoFrame:
    left: np.ndarray
    right: np.ndarray

    # Software timestamps taken immediately after retrieve().
    left_timestamp: float
    right_timestamp: float

    # Difference between the two software retrieval timestamps.
    delta_ms: float

    # Incremented for every new stereo pair.
    sequence: int


class StereoCamera:
    def __init__(
        self,
        left_sensor_id: int = 0,
        right_sensor_id: int = 1,
        sensor_mode: int = 2,
        capture_width: int = 960,
        capture_height: int = 540,
        output_width: int = 960,
        output_height: int = 540,
        framerate: int = 30,
        flip_method: int = 0,
    ):
        self.left_sensor_id = left_sensor_id
        self.right_sensor_id = right_sensor_id

        self.sensor_mode = sensor_mode
        self.capture_width = capture_width
        self.capture_height = capture_height
        self.output_width = output_width
        self.output_height = output_height
        self.framerate = framerate
        self.flip_method = flip_method

        self.left_capture: Optional[cv2.VideoCapture] = None
        self.right_capture: Optional[cv2.VideoCapture] = None

        self._frame_lock = threading.Lock()
        self._capture_thread: Optional[threading.Thread] = None

        self._latest_frame: Optional[StereoFrame] = None
        self._sequence = 0
        self._running = False

    def _gstreamer_pipeline(self, sensor_id: int) -> str:
        return (
            f"nvarguscamerasrc sensor-id={sensor_id} "
            f"sensor-mode={self.sensor_mode} ! "
            f"video/x-raw(memory:NVMM), "
            f"width=(int){self.capture_width}, "
            f"height=(int){self.capture_height}, "
            f"format=(string)NV12, "
            f"framerate=(fraction){self.framerate}/1 ! "
            f"nvvidconv flip-method={self.flip_method} ! "
            f"video/x-raw, "
            f"width=(int){self.output_width}, "
            f"height=(int){self.output_height}, "
            f"format=(string)BGRx ! "
            f"videoconvert ! "
            f"video/x-raw, format=(string)BGR ! "
            f"appsink drop=true max-buffers=1 sync=false"
        )

    def open(self) -> bool:
        if self.is_opened():
            return True

        left_pipeline = self._gstreamer_pipeline(self.left_sensor_id)
        right_pipeline = self._gstreamer_pipeline(self.right_sensor_id)

        self.left_capture = cv2.VideoCapture(
            left_pipeline,
            cv2.CAP_GSTREAMER,
        )

        if not self.left_capture.isOpened():
            print("Unable to open the left camera.")
            print("Pipeline:", left_pipeline)
            self.release()
            return False

        self.right_capture = cv2.VideoCapture(
            right_pipeline,
            cv2.CAP_GSTREAMER,
        )

        if not self.right_capture.isOpened():
            print("Unable to open the right camera.")
            print("Pipeline:", right_pipeline)
            self.release()
            return False

        # Read a few startup frames because Argus may need time to stabilize
        # auto-exposure and auto-white-balance.
        for _ in range(5):
            if not self._capture_pair():
                print("Unable to read initial stereo frames.")
                self.release()
                return False

        return True

    def is_opened(self) -> bool:
        return (
            self.left_capture is not None
            and self.right_capture is not None
            and self.left_capture.isOpened()
            and self.right_capture.isOpened()
        )

    def start(self) -> bool:
        if self._running:
            return True

        if not self.is_opened():
            print("Stereo cameras are not open.")
            return False

        self._running = True
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
        )
        self._capture_thread.start()

        return True

    def _capture_loop(self) -> None:
        while self._running:
            if not self._capture_pair():
                # Prevent a tight CPU loop if a camera temporarily fails.
                time.sleep(0.005)

    def _capture_pair(self) -> bool:
        if not self.is_opened():
            return False

        # Ask both pipelines to advance to their next available frame before
        # decoding/copying either image.
        left_grabbed = self.left_capture.grab()
        right_grabbed = self.right_capture.grab()

        if not left_grabbed or not right_grabbed:
            return False

        left_ok, left_frame = self.left_capture.retrieve()
        left_timestamp = time.monotonic_ns() / 1_000_000_000.0

        right_ok, right_frame = self.right_capture.retrieve()
        right_timestamp = time.monotonic_ns() / 1_000_000_000.0

        if (
            not left_ok
            or not right_ok
            or left_frame is None
            or right_frame is None
        ):
            return False

        delta_ms = abs(right_timestamp - left_timestamp) * 1000.0

        with self._frame_lock:
            self._sequence += 1

            self._latest_frame = StereoFrame(
                left=left_frame,
                right=right_frame,
                left_timestamp=left_timestamp,
                right_timestamp=right_timestamp,
                delta_ms=delta_ms,
                sequence=self._sequence,
            )

        return True

    def read(
        self,
        last_sequence: Optional[int] = None,
        copy_frames: bool = True,
    ) -> Tuple[bool, Optional[StereoFrame]]:
        """
        Return the latest stereo pair.

        last_sequence:
            When supplied, read() returns False if no newer pair exists.

        copy_frames:
            True is safer because the caller receives independent arrays.
            False avoids copies and is faster, but the images must be treated
            as read-only.
        """
        with self._frame_lock:
            if self._latest_frame is None:
                return False, None

            if (
                last_sequence is not None
                and self._latest_frame.sequence == last_sequence
            ):
                return False, None

            frame = self._latest_frame

            if copy_frames:
                result = StereoFrame(
                    left=frame.left.copy(),
                    right=frame.right.copy(),
                    left_timestamp=frame.left_timestamp,
                    right_timestamp=frame.right_timestamp,
                    delta_ms=frame.delta_ms,
                    sequence=frame.sequence,
                )
            else:
                result = frame

        return True, result

    def stop(self) -> None:
        self._running = False

        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2.0)
            self._capture_thread = None

    def release(self) -> None:
        self.stop()

        if self.left_capture is not None:
            self.left_capture.release()
            self.left_capture = None

        if self.right_capture is not None:
            self.right_capture.release()
            self.right_capture = None

        with self._frame_lock:
            self._latest_frame = None

    def __enter__(self):
        if not self.open():
            raise RuntimeError("Unable to open stereo cameras.")

        if not self.start():
            self.release()
            raise RuntimeError("Unable to start stereo capture.")

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
