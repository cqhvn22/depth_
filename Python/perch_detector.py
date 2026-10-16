"""Raspberry Pi 5 stereo perch detector, NCNN FP16, without PyTorch.

python3 Python/perch_detector.py
T: next 7 fresh stereo pairs -> median -> frozen landing assessment.
LIVE runs segmentation/alignment only; stereo depth runs only during the T burst.
R: live/reset, SPACE: pause, D: depth, S: snapshot, Q/ESC: quit.
See perch_detector_pi5_README.md. Defaults are in CONFIG below.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

# Set limits before NumPy/OpenCV are imported. NCNN has its own thread pool.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import cv2
from perch_ncnn import NCNNSegmenter
from perch_video_seg import (
    MaskTracks, PerchWorkflow, SegmentDepth, VideoPair,
    process_stereo_pair, records, refresh_view, save_current,
)

ROOT = Path(__file__).resolve().parent.parent

# CONFIG - chạy không tham số dùng các giá trị này.
CALIB_FILE = ROOT / "Calibration" / "stereo_calibration.npz"
MODEL_FILE = ROOT / "models" / "perch_pi5_640_fp16_ncnn_model"
LEFT_CAM_NUM, RIGHT_CAM_NUM = 0, 1
CAPTURE_WIDTH, CAPTURE_HEIGHT, CAMERA_FPS = 960, 540, 15
DEPTH_DOWNSCALE, CALIB_UNIT = 0.5, "mm"
YOLO_CONF, N_BURST, MIN_VALID_FRAMES = 0.35, 7, 3
NCNN_THREADS, OPENCV_THREADS = 3, 1
SNAPSHOT_DIR = ROOT / "Python" / "perch_snapshots"


class PicameraPairSource:
    """Latest fresh camera pair; never count one sequence twice in a burst.

    camera.py configures Picamera2 BGR888, whose byte order is RGB. Convert
    it here to OpenCV BGR. The software retrieval gap is not exposure sync.
    """
    total = "LIVE"

    def __init__(self, args, camera_factory=None):
        if camera_factory is None:
            from camera import StereoCamera
            camera_factory = StereoCamera
        self.camera = camera_factory(left_camera_num=args.left_camera, right_camera_num=args.right_camera,
                                     capture_width=args.width, capture_height=args.height,
                                     framerate=args.camera_fps)
        self.size, self.fps = (args.width, args.height), args.camera_fps
        self.index, self.last_sequence = 0, None
        self.timeout, self.color = args.camera_timeout, args.camera_color
        self.max_gap, self.last_gap_ms = args.max_pair_gap_ms, None
        self.sequence, self.rejected = None, 0
        try:
            if not self.camera.open() or not self.camera.start():
                raise RuntimeError("Cannot start both Picamera2 cameras")
        except Exception:
            self.release()
            raise

    def discard_buffered(self):
        """Discard a pair already retrieved before T/reset, then wait for a new one."""
        ok, frame = self.camera.read(copy_frames=False)
        if ok and frame is not None:
            self.last_sequence = frame.sequence

    def read(self):
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            ok, frame = self.camera.read(last_sequence=self.last_sequence, copy_frames=False)
            if not ok or frame is None or (self.last_sequence is not None and frame.sequence <= self.last_sequence):
                time.sleep(0.003)
                continue
            self.last_sequence = frame.sequence
            if self.max_gap > 0 and frame.delta_ms > self.max_gap:
                self.rejected += 1
                continue
            left, right = frame.left, frame.right
            if self.color == "rgb":
                left = cv2.cvtColor(left, cv2.COLOR_RGB2BGR)
                right = cv2.cvtColor(right, cv2.COLOR_RGB2BGR)
            self.last_gap_ms, self.sequence = float(frame.delta_ms), int(frame.sequence)
            index = self.index
            self.index += 1
            return index, left, right
        raise TimeoutError(f"No fresh accepted camera pair for {self.timeout:g}s; "
                           f"check cameras/gap limit (rejected={self.rejected})")

    def release(self):
        self.camera.release()


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--source", choices=("camera", "video"), default="camera")
    p.add_argument("--profile", choices=("fast", "detail"), default="fast")
    p.add_argument("--model", type=Path, help="Custom NCNN model directory; overrides profile")
    p.add_argument("--calib", type=Path, default=CALIB_FILE)
    p.add_argument("--left-camera", type=int, default=LEFT_CAM_NUM)
    p.add_argument("--right-camera", type=int, default=RIGHT_CAM_NUM)
    p.add_argument("--width", type=int, default=CAPTURE_WIDTH)
    p.add_argument("--height", type=int, default=CAPTURE_HEIGHT)
    p.add_argument("--camera-fps", type=int, default=CAMERA_FPS)
    p.add_argument("--camera-timeout", type=float, default=3.0)
    p.add_argument("--camera-color", choices=("rgb", "bgr"), default="rgb",
                   help="Byte order from camera.py; its Picamera2 BGR888 outputs RGB bytes")
    p.add_argument("--max-pair-gap-ms", type=float, default=0,
                   help="Optional software retrieval-gap rejection; 0 disables (not exposure timestamps)")
    p.add_argument("--threads", type=int, default=NCNN_THREADS)
    p.add_argument("--cv-threads", type=int, default=OPENCV_THREADS)
    p.add_argument("--fp32-arithmetic", action="store_true", help="Use FP32 arithmetic with FP16 stored weights")
    p.add_argument("--right-mask-every", type=int, default=1,
                   help="Run right-eye segmentation every N pairs; both images always current")
    p.add_argument("--unit", choices=("mm", "cm", "m"), default=CALIB_UNIT)
    p.add_argument("--downscale", type=float, default=DEPTH_DOWNSCALE)
    p.add_argument("--conf", type=float, default=YOLO_CONF)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--classes", type=int, nargs="+")
    p.add_argument("--seg-space", choices=("original", "rectified"), default="original")
    p.add_argument("--burst-frames", type=int, default=N_BURST)
    p.add_argument("--min-samples", type=int, default=MIN_VALID_FRAMES)
    p.add_argument("--align-angle", type=float, default=5)
    p.add_argument("--align-center", type=float, default=0.12)
    p.add_argument("--min-mask-area", type=int, default=16)
    p.add_argument("--min-depth-pixels", type=int, default=20)
    p.add_argument("--min-depth-fraction", type=float, default=0.15)
    p.add_argument("--max-depth", type=float, default=5)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--trigger-frame", type=int, help="Auto T after N processed pairs (zero based)")
    p.add_argument("--max-frames", type=int)
    p.add_argument("--snapshot-dir", type=Path, default=SNAPSHOT_DIR)
    p.add_argument("--save-last", action="store_true")
    p.add_argument("--output", type=Path, help="Save 2x2 preview as MP4")
    p.add_argument("--left", type=Path, default=ROOT / "Python/l1.mp4", help="Video debug only")
    p.add_argument("--right", type=Path, default=ROOT / "Python/r1.mp4", help="Video debug only")
    p.add_argument("--left-start", type=int, default=0)
    p.add_argument("--right-start", type=int, default=0)
    return p


def validate_args(args, parser):
    if not args.calib.is_file():
        parser.error(f"Calibration missing: {args.calib}")
    if min(args.width, args.height, args.camera_fps, args.threads, args.cv_threads, args.right_mask_every) < 1:
        parser.error("Resolution, FPS, threads and right-mask-every must be positive")
    if args.left_camera == args.right_camera and args.source == "camera":
        parser.error("Left and right must use different camera indices")
    if not (0 < args.downscale <= 1 and 0 < args.conf <= 1 and 0 < args.iou <= 1):
        parser.error("downscale/conf/iou must be in (0, 1]")
    if not (0 < args.align_angle < 90 and 0 < args.align_center < .5):
        parser.error("Invalid alignment tolerances")
    if not 1 <= args.min_samples <= args.burst_frames:
        parser.error("Require 1 <= min-samples <= burst-frames")
    if min(args.min_mask_area, args.min_depth_pixels) < 1 or not 0 < args.min_depth_fraction <= 1:
        parser.error("Invalid mask/depth sample thresholds")
    if args.camera_timeout <= 0 or args.max_depth <= 0 or args.max_pair_gap_ms < 0:
        parser.error("Invalid camera timeout, depth maximum or pair-gap threshold")
    if min(args.left_start, args.right_start) < 0 or (args.trigger_frame is not None and args.trigger_frame < 0):
        parser.error("Frame indices must be nonnegative")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("max-frames must be positive")
    if args.headless and args.source == "camera" and args.max_frames is None and args.trigger_frame is None:
        parser.error("Headless camera mode needs max-frames or trigger-frame")
    if args.output:
        if args.output.suffix.lower() != ".mp4":
            parser.error("output must be .mp4")
        if args.output.resolve() in {args.left.resolve(), args.right.resolve(), args.calib.resolve()}:
            parser.error("output cannot overwrite an input")


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)
    cv2.setNumThreads(args.cv_threads)
    model_path = args.model or (MODEL_FILE if args.profile == "fast" else
                               ROOT / "models/perch_pi5_960_fp16_ncnn_model")
    model = NCNNSegmenter(model_path, args.threads, fp16_arithmetic=not args.fp32_arithmetic)
    args.imgsz, args.device = max(model.imgsz), "cpu"
    estimator = SegmentDepth(args)
    print(f"[MODEL] NCNN FP16 {model_path} | {model.imgsz} | threads={args.threads}; no PyTorch")
    source = writer = last = None
    workflow = PerchWorkflow()
    tracks = MaskTracks(max(7, args.burst_frames), args.min_samples)
    processed, auto_triggered, show_depth = 0, False, True
    window = "Pi 5 Perch - NCNN - T to measure"
    started = time.perf_counter()
    try:
        source = (PicameraPairSource(args) if args.source == "camera" else
                  VideoPair(args.left, args.right, (args.left_start, args.right_start)))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"),
                                     source.fps, (estimator.size[0]*2, estimator.size[1]*2))
            if not writer.isOpened():
                raise RuntimeError(f"Cannot create output: {args.output}")
        if not args.headless:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window, 1280, 720)
        print("[CONTROLS] T=burst, R=live/reset, SPACE=pause, D=depth, S=save, Q=quit")
        print("[DEPTH] LIVE=off; T=reference SGBM/WLS for the next burst only", flush=True)
        while True:
            tick = time.perf_counter()
            if not workflow.paused:
                pair = source.read()
                if pair is None:
                    if workflow.state == "CAPTURING":
                        workflow.finish(last["detections"] if last else [])
                        if last:
                            refresh_view(last, workflow, args, source, show_depth)
                        if not args.headless:
                            continue
                    break
                index, left, right = pair
                last = process_stereo_pair(model, estimator, tracks, workflow, args, index, left, right,
                                           right_mask=index % args.right_mask_every == 0)
                processed += 1
                limit = args.max_frames is not None and processed >= args.max_frames
                if workflow.state == "CAPTURING" and (limit or (isinstance(source, VideoPair) and source.index == source.total)):
                    workflow.finish(last["detections"])
                refresh_view(last, workflow, args, source, show_depth)
                last["summary"].update(source=args.source, backend="ncnn", model=str(model_path),
                                       camera_sequence=getattr(source, "sequence", None),
                                       retrieval_gap_ms=getattr(source, "last_gap_ms", None))
                if writer is not None:
                    writer.write(last["view"])
                if processed == 1 or processed % 30 == 0:
                    print(json.dumps(dict(pair=index, state=workflow.state, inference_ms=model.speed,
                                          detections=records(last["detections"])), allow_nan=False), flush=True)
                if workflow.state == "RESULT":
                    print("[RESULT] " + json.dumps(workflow.result, allow_nan=False), flush=True)
                if limit or (args.headless and workflow.state == "RESULT"):
                    break
            if args.headless:
                key = -1
            else:
                refresh_view(last, workflow, args, source, show_depth)
                cv2.imshow(window, last["view"])
                delay = max(1, round((1 / source.fps - (time.perf_counter()-tick))*1000))
                key = cv2.waitKey(30 if workflow.paused else delay) & 0xff
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
            if key in (ord("q"), 27):
                break
            if args.trigger_frame is not None and not auto_triggered and last["index"] >= args.trigger_frame and workflow.state == "LIVE":
                key, auto_triggered = ord("t"), True
            if key == ord("t"):
                if workflow.trigger(args.burst_frames, args.min_samples):
                    tracks.reset_measurements()
                    if isinstance(source, PicameraPairSource):
                        source.discard_buffered()
                    print(f"[BURST] Target #{workflow.target_id}: next {args.burst_frames} fresh pairs", flush=True)
            elif key == ord("r") or (key == ord(" ") and workflow.state == "RESULT"):
                workflow.reset()
                tracks.reset_measurements()
                if isinstance(source, PicameraPairSource):
                    source.discard_buffered()
            elif key == ord(" "):
                workflow.paused = not workflow.paused
            elif key == ord("d"):
                show_depth = not show_depth
            elif key == ord("b") and isinstance(source, VideoPair):
                source.rewind()
                tracks = MaskTracks(max(7, args.burst_frames), args.min_samples)
                workflow.reset()
            elif key == ord("s"):
                save_current(args.snapshot_dir, last)
        if last is None:
            raise RuntimeError("No stereo pairs received")
        if args.save_last:
            refresh_view(last, workflow, args, source, show_depth)
            save_current(args.snapshot_dir, last)
        print(f"[DONE] {processed} pairs; {time.perf_counter()-started:.1f}s; torch imported={'torch' in sys.modules}")
    finally:
        if source is not None:
            source.release()
        if writer is not None:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, KeyError, OSError, ImportError, cv2.error) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
