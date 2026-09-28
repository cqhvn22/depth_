import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    from picamera2 import Picamera2
except ImportError:
    raise ImportError(
        "picamera2 is not installed. Install it with:\n"
        "  sudo apt install python3-picamera2\n"
        "or:\n"
        "  pip install picamera2"
    )


@dataclass
class StereoFrame:
    left: np.ndarray
    right: np.ndarray

    # Software timestamps taken immediately after capture.
    left_timestamp: float
    right_timestamp: float

    # Difference between the two software retrieval timestamps.
    delta_ms: float

    # Incremented for every new stereo pair.
    sequence: int


class StereoCamera:
    """
    Stereo camera class for Raspberry Pi 5 using picamera2.

    Raspberry Pi 5 has two CSI camera ports (CAM0 and CAM1).
    left_camera_num  → typically 0 (CAM0 port)
    right_camera_num → typically 1 (CAM1 port)
    """

    def __init__(
        self,
        left_camera_num: int = 0,
        right_camera_num: int = 1,
        capture_width: int = 960,
        capture_height: int = 540,
        framerate: int = 30,
        # Deprecated parameters kept for backward compatibility
        left_sensor_id: Optional[int] = None,
        right_sensor_id: Optional[int] = None,
        sensor_mode: int = 2,
        output_width: Optional[int] = None,
        output_height: Optional[int] = None,
        flip_method: int = 0,
    ):
        # Support old positional API (left_sensor_id / right_sensor_id)
        if left_sensor_id is not None:
            left_camera_num = left_sensor_id
        if right_sensor_id is not None:
            right_camera_num = right_sensor_id

        self.left_camera_num = left_camera_num
        self.right_camera_num = right_camera_num

        self.capture_width = capture_width
        self.capture_height = capture_height

        # output size defaults to capture size (no hardware downscale needed)
        self.output_width = output_width if output_width is not None else capture_width
        self.output_height = output_height if output_height is not None else capture_height

        self.framerate = framerate
        self.flip_method = flip_method  # kept for compatibility, applied via cv2.rotate

        self._left_cam: Optional[Picamera2] = None
        self._right_cam: Optional[Picamera2] = None

        self._frame_lock = threading.Lock()
        self._capture_thread: Optional[threading.Thread] = None

        self._latest_frame: Optional[StereoFrame] = None
        self._sequence = 0
        self._running = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_config(self, cam: Picamera2) -> dict:
        """Build a picamera2 still/video configuration."""
        config = cam.create_video_configuration(
            main={
                "size": (self.capture_width, self.capture_height),
                "format": "BGR888",
            },
            controls={
                "FrameRate": float(self.framerate),
            },
            buffer_count=4,
        )
        return config

    def _apply_flip(self, frame: np.ndarray) -> np.ndarray:
        """
        Replicate Jetson nvvidconv flip-method behaviour via OpenCV.

        flip_method:
            0 → no flip
            1 → counterclockwise 90°
            2 → rotate 180°
            3 → clockwise 90°
            4 → horizontal flip
            5 → upper-right diagonal flip
            6 → vertical flip
            7 → upper-left diagonal flip
        """
        if self.flip_method == 0:
            return frame
        elif self.flip_method == 1:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif self.flip_method == 2:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif self.flip_method == 3:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.flip_method == 4:
            return cv2.flip(frame, 1)
        elif self.flip_method == 5:
            return cv2.transpose(frame)
        elif self.flip_method == 6:
            return cv2.flip(frame, 0)
        elif self.flip_method == 7:
            return cv2.flip(cv2.transpose(frame), 1)
        return frame

    def _resize_if_needed(self, frame: np.ndarray) -> np.ndarray:
        """Resize output if output resolution differs from capture resolution."""
        if (self.output_width != self.capture_width or
                self.output_height != self.capture_height):
            frame = cv2.resize(
                frame,
                (self.output_width, self.output_height),
                interpolation=cv2.INTER_LINEAR,
            )
        return frame

    def _grab_frame(self, cam: Picamera2) -> Tuple[np.ndarray, float]:
        """Capture a single frame from one Picamera2 instance."""
        frame = cam.capture_array("main")
        timestamp = time.monotonic_ns() / 1_000_000_000.0

        # picamera2 returns BGR888 directly, but ensure contiguous array
        frame = np.ascontiguousarray(frame)
        frame = self._apply_flip(frame)
        frame = self._resize_if_needed(frame)
        return frame, timestamp

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def open(self) -> bool:
        if self.is_opened():
            return True

        try:
            self._left_cam = Picamera2(camera_num=self.left_camera_num)
            left_config = self._make_config(self._left_cam)
            self._left_cam.configure(left_config)
            self._left_cam.start()
        except Exception as exc:
            print(f"Unable to open the left camera (CAM{self.left_camera_num}): {exc}")
            self.release()
            return False

        try:
            self._right_cam = Picamera2(camera_num=self.right_camera_num)
            right_config = self._make_config(self._right_cam)
            self._right_cam.configure(right_config)
            self._right_cam.start()
        except Exception as exc:
            print(f"Unable to open the right camera (CAM{self.right_camera_num}): {exc}")
            self.release()
            return False

        # Warm up: discard a few startup frames so AE/AWB can stabilise.
        for _ in range(5):
            if not self._capture_pair():
                print("Unable to read initial stereo frames.")
                self.release()
                return False

        return True

    def is_opened(self) -> bool:
        return (
            self._left_cam is not None
            and self._right_cam is not None
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

        try:
            left_frame, left_timestamp = self._grab_frame(self._left_cam)
            right_frame, right_timestamp = self._grab_frame(self._right_cam)
        except Exception as exc:
            print(f"Frame capture error: {exc}")
            return False

        if left_frame is None or right_frame is None:
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

        if self._left_cam is not None:
            try:
                self._left_cam.stop()
                self._left_cam.close()
            except Exception:
                pass
            self._left_cam = None

        if self._right_cam is not None:
            try:
                self._right_cam.stop()
                self._right_cam.close()
            except Exception:
                pass
            self._right_cam = None

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
