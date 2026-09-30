"""Record synchronized left and right camera streams to MP4 files."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import cv2

from camera import StereoCamera


def make_writer(path: Path, frame_size: tuple[int, int], fps: float) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        frame_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {path}")
    return writer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record synchronized videos from the left and right cameras."
    )
    parser.add_argument("--left-sensor", type=int, default=0)
    parser.add_argument("--right-sensor", type=int, default=1)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output-dir", default="recordings")
    parser.add_argument("--left-file", default="")
    parser.add_argument("--right-file", default="")
    parser.add_argument("--flip-method", type=int, default=0)
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Do not show the camera preview while recording",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    left_path = Path(args.left_file) if args.left_file else output_dir / f"left_{timestamp}.mp4"
    right_path = Path(args.right_file) if args.right_file else output_dir / f"right_{timestamp}.mp4"
    frame_size = (args.width, args.height)

    camera = StereoCamera(
        left_camera_num=args.left_sensor,
        right_camera_num=args.right_sensor,
        capture_width=args.width,
        capture_height=args.height,
        framerate=args.fps,
        flip_method=args.flip_method,
    )
    left_writer = make_writer(left_path, frame_size, args.fps)
    right_writer = make_writer(right_path, frame_size, args.fps)

    frame_count = 0
    last_sequence = None

    try:
        if not camera.open() or not camera.start():
            raise RuntimeError("Could not start the stereo cameras.")

        print(f"Left video : {left_path}")
        print(f"Right video: {right_path}")
        print("Recording. Press Q or ESC to stop.")

        while True:
            success, stereo_frame = camera.read(
                last_sequence=last_sequence,
                copy_frames=False,
            )
            if not success or stereo_frame is None:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                continue

            last_sequence = stereo_frame.sequence
            left_writer.write(stereo_frame.left)
            right_writer.write(stereo_frame.right)
            frame_count += 1

            if not args.no_preview:
                preview = cv2.hconcat([stereo_frame.left, stereo_frame.right])
                cv2.putText(
                    preview,
                    f"Frames: {frame_count}  Pair delta: {stereo_frame.delta_ms:.2f} ms",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Stereo recording: left | right", preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
    finally:
        camera.release()
        left_writer.release()
        right_writer.release()
        cv2.destroyAllWindows()

    print(f"Saved {frame_count} synchronized frame pairs.")
    print(f"Left : {left_path}")
    print(f"Right: {right_path}")


if __name__ == "__main__":
    main()