"""
stereo_calibration_capture.py

Capture stereo chessboard image pairs from two CSI cameras
on a Raspberry Pi 5 using picamera2.

Controls:
    SPACE or S : Save the current stereo pair
    A          : Toggle automatic capture
    R          : Reset the saved-pair counter
    Q or ESC   : Quit

Example:
    python3 stereo_calibration_capture.py
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

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


class Camera:
    """Threaded CSI camera reader for Raspberry Pi 5 using picamera2."""

    def __init__(self, name: str = "camera") -> None:
        self.name = name
        self._picam: Picamera2 | None = None

        self.frame: np.ndarray | None = None
        self.grabbed = False
        self.frame_id = 0
        self.timestamp_ns = 0

        self.read_thread: threading.Thread | None = None
        self.read_lock = threading.Lock()
        self.running = False

    def open(
        self,
        sensor_id: int,
        # The parameters below are kept for API compatibility with the
        # Jetson version but have different semantics on the Pi:
        sensor_mode: int = 2,         # unused – picamera2 uses sensor modes differently
        capture_width: int = 1920,
        capture_height: int = 1080,
        output_width: int = 960,
        output_height: int = 540,
        framerate: int = 30,
        flip_method: int = 0,
    ) -> None:
        if self._picam is not None:
            self.release()

        self._output_width = output_width
        self._output_height = output_height
        self._flip_method = flip_method

        print(f"{self.name}: opening CAM{sensor_id} "
              f"@ {capture_width}x{capture_height} → "
              f"{output_width}x{output_height}, {framerate} fps")

        try:
            self._picam = Picamera2(camera_num=sensor_id)
            config = self._picam.create_video_configuration(
                main={
                    "size": (capture_width, capture_height),
                    "format": "BGR888",
                },
                controls={"FrameRate": float(framerate)},
                buffer_count=4,
            )
            self._picam.configure(config)
            self._picam.start()
        except Exception as exc:
            if self._picam is not None:
                try:
                    self._picam.close()
                except Exception:
                    pass
                self._picam = None
            raise RuntimeError(
                f"Could not open {self.name}, sensor-id={sensor_id}: {exc}"
            )

        # Read an initial frame before starting the thread.
        try:
            raw = self._picam.capture_array("main")
        except Exception as exc:
            self._picam.close()
            self._picam = None
            raise RuntimeError(
                f"{self.name} opened but did not return an initial frame: {exc}"
            )

        frame = self._process_frame(raw)

        with self.read_lock:
            self.grabbed = True
            self.frame = frame
            self.frame_id = 1
            self.timestamp_ns = time.monotonic_ns()

    def _process_frame(self, raw: np.ndarray) -> np.ndarray:
        """Apply flip and output resize."""
        frame = np.ascontiguousarray(raw)

        flip = self._flip_method
        if flip == 1:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif flip == 2:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif flip == 3:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif flip == 4:
            frame = cv2.flip(frame, 1)
        elif flip == 5:
            frame = cv2.transpose(frame)
        elif flip == 6:
            frame = cv2.flip(frame, 0)
        elif flip == 7:
            frame = cv2.flip(cv2.transpose(frame), 1)

        h, w = frame.shape[:2]
        if w != self._output_width or h != self._output_height:
            frame = cv2.resize(
                frame,
                (self._output_width, self._output_height),
                interpolation=cv2.INTER_LINEAR,
            )
        return frame

    def start(self) -> None:
        if self._picam is None:
            raise RuntimeError(f"{self.name} must be opened before start().")

        if self.running:
            return

        self.running = True
        self.read_thread = threading.Thread(
            target=self._read_loop,
            name=f"{self.name}-reader",
            daemon=True,
        )
        self.read_thread.start()

    def _read_loop(self) -> None:
        while self.running:
            if self._picam is None:
                break

            try:
                raw = self._picam.capture_array("main")
                timestamp_ns = time.monotonic_ns()
            except Exception:
                time.sleep(0.005)
                continue

            frame = self._process_frame(raw)

            with self.read_lock:
                self.grabbed = True
                self.frame = frame
                self.frame_id += 1
                self.timestamp_ns = timestamp_ns

    def read(self) -> tuple[bool, np.ndarray | None, int, int]:
        """
        Return:
            success, copied frame, frame ID, timestamp in nanoseconds
        """
        with self.read_lock:
            if not self.grabbed or self.frame is None:
                return False, None, self.frame_id, self.timestamp_ns

            return (
                True,
                self.frame.copy(),
                self.frame_id,
                self.timestamp_ns,
            )

    def stop(self) -> None:
        self.running = False

        if self.read_thread is not None:
            self.read_thread.join(timeout=2.0)
            self.read_thread = None

    def release(self) -> None:
        self.stop()

        if self._picam is not None:
            try:
                self._picam.stop()
                self._picam.close()
            except Exception:
                pass
            self._picam = None

        with self.read_lock:
            self.frame = None
            self.grabbed = False


class StereoCalibrationCapture:
    """Capture and validate stereo chessboard image pairs."""

    def __init__(
        self,
        left_sensor_id: int,
        right_sensor_id: int,
        output_directory: str,
        board_size: tuple[int, int],
        sensor_mode: int = 2,
        capture_width: int = 960,
        capture_height: int = 540,
        output_width: int = 960,
        output_height: int = 540,
        framerate: int = 30,
        flip_method: int = 0,
        target_pairs: int = 30,
        automatic_interval: float = 1.5,
        maximum_timestamp_difference_ms: float = 50.0,
    ) -> None:
        self.left_camera = Camera("left camera")
        self.right_camera = Camera("right camera")

        self.left_sensor_id = left_sensor_id
        self.right_sensor_id = right_sensor_id

        self.sensor_mode = sensor_mode
        self.capture_width = capture_width
        self.capture_height = capture_height
        self.output_width = output_width
        self.output_height = output_height
        self.framerate = framerate
        self.flip_method = flip_method

        self.board_size = board_size
        self.target_pairs = target_pairs
        self.automatic_interval = automatic_interval
        self.maximum_timestamp_difference_ms = (
            maximum_timestamp_difference_ms
        )

        self.output_directory = Path(output_directory)
        self.left_directory = self.output_directory / "left"
        self.right_directory = self.output_directory / "right"
        self.preview_directory = self.output_directory / "preview"

        self.left_directory.mkdir(parents=True, exist_ok=True)
        self.right_directory.mkdir(parents=True, exist_ok=True)
        self.preview_directory.mkdir(parents=True, exist_ok=True)

        self.saved_pairs = self._find_next_index()
        self.automatic_capture = False
        self.last_automatic_capture_time = 0.0

    def _find_next_index(self) -> int:
        """Continue numbering after existing calibration images."""

        existing_indices: list[int] = []

        for image_path in self.left_directory.glob("left_*.png"):
            try:
                existing_indices.append(int(image_path.stem.split("_")[-1]))
            except ValueError:
                continue

        if not existing_indices:
            return 0

        return max(existing_indices) + 1

    def _camera_arguments(self, sensor_id: int) -> dict:
        return {
            "sensor_id": sensor_id,
            "sensor_mode": self.sensor_mode,
            "capture_width": self.capture_width,
            "capture_height": self.capture_height,
            "output_width": self.output_width,
            "output_height": self.output_height,
            "framerate": self.framerate,
            "flip_method": self.flip_method,
        }

    def open(self) -> None:
        self.left_camera.open(
            **self._camera_arguments(self.left_sensor_id)
        )

        try:
            self.right_camera.open(
                **self._camera_arguments(self.right_sensor_id)
            )
        except Exception:
            self.left_camera.release()
            raise

        self.left_camera.start()
        self.right_camera.start()

    def close(self) -> None:
        self.left_camera.release()
        self.right_camera.release()
        cv2.destroyAllWindows()

    def detect_board(
        self,
        frame: np.ndarray,
    ) -> tuple[bool, np.ndarray | None]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flags = (
            cv2.CALIB_CB_NORMALIZE_IMAGE
            | cv2.CALIB_CB_EXHAUSTIVE
            | cv2.CALIB_CB_ACCURACY
        )

        found, corners = cv2.findChessboardCornersSB(
            gray,
            self.board_size,
            flags=flags,
        )

        return found, corners

    def create_preview(
        self,
        left_frame: np.ndarray,
        right_frame: np.ndarray,
        left_found: bool,
        right_found: bool,
        left_corners: np.ndarray | None,
        right_corners: np.ndarray | None,
        timestamp_difference_ms: float,
    ) -> np.ndarray:
        left_preview = left_frame.copy()
        right_preview = right_frame.copy()

        if left_found and left_corners is not None:
            cv2.drawChessboardCorners(
                left_preview,
                self.board_size,
                left_corners,
                left_found,
            )

        if right_found and right_corners is not None:
            cv2.drawChessboardCorners(
                right_preview,
                self.board_size,
                right_corners,
                right_found,
            )

        left_status = "BOARD FOUND" if left_found else "NO BOARD"
        right_status = "BOARD FOUND" if right_found else "NO BOARD"

        cv2.putText(
            left_preview,
            f"LEFT: {left_status}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0) if left_found else (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            right_preview,
            f"RIGHT: {right_status}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0) if right_found else (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        combined = np.hstack((left_preview, right_preview))

        automatic_text = "ON" if self.automatic_capture else "OFF"

        info_lines = [
            f"Saved pairs: {self.saved_pairs}/{self.target_pairs}",
            f"Automatic capture: {automatic_text}",
            f"Frame timestamp difference: {timestamp_difference_ms:.1f} ms",
            "SPACE/S: save | A: auto | Q/ESC: quit",
        ]

        panel_height = 105
        panel = np.zeros(
            (panel_height, combined.shape[1], 3),
            dtype=np.uint8,
        )

        for line_index, text in enumerate(info_lines):
            cv2.putText(
                panel,
                text,
                (15, 24 + line_index * 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return np.vstack((combined, panel))

    def save_pair(
        self,
        left_frame: np.ndarray,
        right_frame: np.ndarray,
        preview: np.ndarray | None = None,
    ) -> bool:
        index = self.saved_pairs

        left_path = self.left_directory / f"left_{index:03d}.png"
        right_path = self.right_directory / f"right_{index:03d}.png"
        preview_path = self.preview_directory / f"pair_{index:03d}.jpg"

        left_success = cv2.imwrite(str(left_path), left_frame)
        right_success = cv2.imwrite(str(right_path), right_frame)

        if not left_success or not right_success:
            # Remove a partial pair so directories remain consistent.
            left_path.unlink(missing_ok=True)
            right_path.unlink(missing_ok=True)

            print("Could not save stereo pair.")
            return False

        if preview is not None:
            cv2.imwrite(
                str(preview_path),
                preview,
                [cv2.IMWRITE_JPEG_QUALITY, 85],
            )

        print(
            f"Saved pair {index:03d}: "
            f"{left_path.name}, {right_path.name}"
        )

        self.saved_pairs += 1
        return True

    def reset_counter(self) -> None:
        """
        Reset only the numbering counter.

        Existing images are not deleted and may be overwritten, so this
        option is included mainly for controlled test sessions.
        """
        self.saved_pairs = 0
        print("Capture counter reset to zero.")

    def run(self) -> None:
        self.open()

        print("\nStereo calibration capture started.")
        print(f"Board inner corners: {self.board_size}")
        print(f"Output directory: {self.output_directory.resolve()}")
        print("Move the board around the complete image area.")
        print("Use different distances, tilts, and rotations.\n")

        try:
            while True:
                (
                    left_success,
                    left_frame,
                    left_frame_id,
                    left_timestamp_ns,
                ) = self.left_camera.read()

                (
                    right_success,
                    right_frame,
                    right_frame_id,
                    right_timestamp_ns,
                ) = self.right_camera.read()

                if (
                    not left_success
                    or not right_success
                    or left_frame is None
                    or right_frame is None
                ):
                    time.sleep(0.005)
                    continue

                timestamp_difference_ms = abs(
                    left_timestamp_ns - right_timestamp_ns
                ) / 1_000_000.0

                left_found, left_corners = self.detect_board(left_frame)
                right_found, right_corners = self.detect_board(right_frame)

                preview = self.create_preview(
                    left_frame=left_frame,
                    right_frame=right_frame,
                    left_found=left_found,
                    right_found=right_found,
                    left_corners=left_corners,
                    right_corners=right_corners,
                    timestamp_difference_ms=timestamp_difference_ms,
                )

                cv2.imshow("Stereo calibration capture", preview)

                key = cv2.waitKey(1) & 0xFF
                manual_save_requested = key in (ord("s"), ord(" "))

                if key in (ord("q"), 27):
                    break

                if key == ord("a"):
                    self.automatic_capture = not self.automatic_capture
                    self.last_automatic_capture_time = time.monotonic()

                    print(
                        "Automatic capture:",
                        "enabled" if self.automatic_capture else "disabled",
                    )

                if key == ord("r"):
                    self.reset_counter()

                current_time = time.monotonic()
                automatic_save_requested = (
                    self.automatic_capture
                    and current_time - self.last_automatic_capture_time
                    >= self.automatic_interval
                )

                save_requested = (
                    manual_save_requested
                    or automatic_save_requested
                )

                if not save_requested:
                    continue

                if not left_found or not right_found:
                    print(
                        "Pair not saved: chessboard must be visible "
                        "in both cameras."
                    )
                    self.last_automatic_capture_time = current_time
                    continue

                if (
                    timestamp_difference_ms
                    > self.maximum_timestamp_difference_ms
                ):
                    print(
                        "Pair not saved: frame timing difference "
                        f"{timestamp_difference_ms:.1f} ms exceeds "
                        f"{self.maximum_timestamp_difference_ms:.1f} ms."
                    )
                    self.last_automatic_capture_time = current_time
                    continue

                saved = self.save_pair(
                    left_frame=left_frame,
                    right_frame=right_frame,
                    preview=preview,
                )

                self.last_automatic_capture_time = current_time

                if saved and self.saved_pairs >= self.target_pairs:
                    print(
                        f"\nTarget of {self.target_pairs} pairs reached."
                    )
                    self.automatic_capture = False

        finally:
            self.close()


def parse_board_size(value: str) -> tuple[int, int]:
    """Parse a board size such as 9x6."""

    normalized = value.lower().replace(",", "x")
    parts = normalized.split("x")

    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            "Board size must use the format columnsxrows, for example 9x6."
        )

    try:
        columns = int(parts[0])
        rows = int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Board dimensions must be integers."
        ) from exc

    if columns < 2 or rows < 2:
        raise argparse.ArgumentTypeError(
            "Board dimensions must both be at least 2."
        )

    return columns, rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture stereo chessboard calibration images (Raspberry Pi 5 / picamera2)."
    )

    parser.add_argument("--left-sensor", type=int, default=0,
                        help="Camera number for the left camera (CAM port index, default: 0)")
    parser.add_argument("--right-sensor", type=int, default=1,
                        help="Camera number for the right camera (CAM port index, default: 1)")
    parser.add_argument("--sensor-mode", type=int, default=2,
                        help="Sensor mode (informational only, not used by picamera2 directly)")

    parser.add_argument("--capture-width", type=int, default=1920)
    parser.add_argument("--capture-height", type=int, default=1080)
    parser.add_argument("--output-width", type=int, default=960)
    parser.add_argument("--output-height", type=int, default=540)
    parser.add_argument("--framerate", type=int, default=30)
    parser.add_argument("--flip-method", type=int, default=0,
                        help="0=none, 2=180°, 4=hflip, 6=vflip (same convention as nvvidconv)")

    parser.add_argument(
        "--board-size",
        type=parse_board_size,
        default=(9, 6),
        help="Number of internal corners, for example 9x6.",
    )

    parser.add_argument(
        "--output",
        default="calibration_images",
    )

    parser.add_argument("--target-pairs", type=int, default=30)

    parser.add_argument(
        "--auto-interval",
        type=float,
        default=1.5,
        help="Seconds between automatic captures.",
    )

    parser.add_argument(
        "--max-time-difference",
        type=float,
        default=50.0,
        help="Maximum accepted left/right frame timing difference in ms.",
    )

    args = parser.parse_args()

    application = StereoCalibrationCapture(
        left_sensor_id=args.left_sensor,
        right_sensor_id=args.right_sensor,
        output_directory=args.output,
        board_size=args.board_size,
        sensor_mode=args.sensor_mode,
        capture_width=args.capture_width,
        capture_height=args.capture_height,
        output_width=args.output_width,
        output_height=args.output_height,
        framerate=args.framerate,
        flip_method=args.flip_method,
        target_pairs=args.target_pairs,
        automatic_interval=args.auto_interval,
        maximum_timestamp_difference_ms=args.max_time_difference,
    )

    application.run()


if __name__ == "__main__":
    main()
