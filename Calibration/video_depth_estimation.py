"""
video_depth_estimation.py

Stereo depth estimation from pre-recorded left/right video files.
Uses the same StereoSGBM + WLS algorithm as the live depth_test.py pipeline.

Usage:
    python3 video_depth_estimation.py \\
        --left  recordings/left_20260928_134412.mp4 \\
        --right recordings/right_20260928_134412.mp4 \\
        --calib stereo_calibration.npz

Controls (preview window):
    Q / ESC  : Quit
    SPACE    : Pause / Resume
    S        : Save current frame set as PNG
    [ / ]    : Decrease / Increase playback speed
    R        : Restart from the beginning

Optional – export to video:
    python3 video_depth_estimation.py --left L.mp4 --right R.mp4 \\
        --calib stereo_calibration.npz --save-video output.mp4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# StereoDepth  (identical algorithm to Python/stereo_depth.py)
# ---------------------------------------------------------------------------

class StereoDepth:
    """SGBM stereo matcher with WLS filter – same parameters as the live
    pipeline so depth values are directly comparable."""

    def __init__(self, calibration_file: str, downscale: float = 0.5) -> None:
        data = np.load(calibration_file)

        self.left_map_x  = data["left_map_x"]
        self.left_map_y  = data["left_map_y"]
        self.right_map_x = data["right_map_x"]
        self.right_map_y = data["right_map_y"]
        self.Q           = data["Q"].copy()

        # Resolution the calibration was computed at
        self._calib_w = int(self.left_map_x.shape[1])
        self._calib_h = int(self.left_map_x.shape[0])
        self._video_w: int | None = None  # set on first process() call
        self._video_h: int | None = None

        print(f"Calibration map size : {self._calib_w}x{self._calib_h}")

        self.downscale = downscale

        block_size = 7
        num_disp   = 128 if downscale >= 1.0 else 96

        self.matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disp,
            blockSize=block_size,
            P1=8  * block_size ** 2,
            P2=32 * block_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

        self.right_matcher = cv2.ximgproc.createRightMatcher(self.matcher)
        self.wls_filter    = cv2.ximgproc.createDisparityWLSFilter(
            matcher_left=self.matcher
        )
        self.wls_filter.setLambda(8000)
        self.wls_filter.setSigmaColor(1.5)

    def process(
        self, left: np.ndarray, right: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process a stereo pair and return rectified image, disparity, and depth.

        If the input frames differ from the calibration resolution (including
        a different aspect ratio), they are resized to the calibration size
        before remapping. This is the only geometrically correct approach when
        the aspect ratios do not match (e.g. 640x480 video vs 960x540 calib).

        Returns:
            left_rectified  – BGR image at calibration resolution
            disparity       – float32 disparity map
            depth           – float32 depth map in the calibration unit (mm)
        """
        h, w = left.shape[:2]

        # Resize to calibration resolution if needed (handles both same-ratio
        # scale differences and different-aspect-ratio recordings).
        if w != self._calib_w or h != self._calib_h:
            if self._video_w != w or self._video_h != h:
                # Log once per unique video resolution
                print(f"Video size           : {w}x{h}")
                print(f"Calibration size     : {self._calib_w}x{self._calib_h}")
                if abs(w / h - self._calib_w / self._calib_h) > 0.01:
                    print("WARNING: Different aspect ratios detected "
                          f"({w/h:.3f} vs {self._calib_w/self._calib_h:.3f}). "
                          "Frames will be stretched to calibration resolution.\n"
                          "For accurate depth, re-record videos at "
                          f"{self._calib_w}x{self._calib_h} or re-calibrate at "
                          f"{w}x{h}.\n")
                else:
                    print(f"Resizing frames {w}x{h} -> "
                          f"{self._calib_w}x{self._calib_h} (same aspect ratio).\n")
                self._video_w = w
                self._video_h = h

            left  = cv2.resize(left,  (self._calib_w, self._calib_h),
                                interpolation=cv2.INTER_LINEAR)
            right = cv2.resize(right, (self._calib_w, self._calib_h),
                                interpolation=cv2.INTER_LINEAR)

        left_rectified  = cv2.remap(left,  self.left_map_x,  self.left_map_y,  cv2.INTER_LINEAR)
        right_rectified = cv2.remap(right, self.right_map_x, self.right_map_y, cv2.INTER_LINEAR)

        left_gray  = cv2.cvtColor(left_rectified,  cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)

        if self.downscale != 1.0:
            small_left  = cv2.resize(left_gray,  None,
                                     fx=self.downscale, fy=self.downscale,
                                     interpolation=cv2.INTER_AREA)
            small_right = cv2.resize(right_gray, None,
                                     fx=self.downscale, fy=self.downscale,
                                     interpolation=cv2.INTER_AREA)
        else:
            small_left, small_right = left_gray, right_gray

        disp_left  = self.matcher.compute(small_left, small_right)
        disp_right = self.right_matcher.compute(small_right, small_left)

        filtered = self.wls_filter.filter(
            disp_left, small_left, disparity_map_right=disp_right
        )

        disparity = filtered.astype(np.float32) / 16.0

        if self.downscale != 1.0:
            disparity = cv2.resize(
                disparity,
                (left_gray.shape[1], left_gray.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            ) / self.downscale

        points_3d = cv2.reprojectImageTo3D(disparity, self.Q)
        depth = points_3d[:, :, 2]

        valid = np.isfinite(depth) & (disparity > 0) & (depth > 0)
        depth[~valid] = 0

        return left_rectified, disparity, depth


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def disparity_to_color(disparity: np.ndarray) -> np.ndarray:
    """Render a float32 disparity map as a colour image (TURBO colormap)."""
    valid   = disparity > 0
    display = np.zeros(disparity.shape, dtype=np.uint8)

    if np.any(valid):
        v = disparity[valid]
        lo, hi = np.percentile(v, 2), np.percentile(v, 98)

        if hi > lo:
            normalized = np.clip((disparity - lo) / (hi - lo), 0, 1)
            display    = (normalized * 255).astype(np.uint8)

    display[~valid] = 0
    return cv2.applyColorMap(display, cv2.COLORMAP_TURBO)


def depth_to_color(depth: np.ndarray,
                   max_depth_m: float = 5.0,
                   calibration_unit: str = "mm") -> np.ndarray:
    """Render a float32 depth map as a colour image."""
    depth_m = depth.copy()
    if calibration_unit == "mm":
        depth_m /= 1000.0
    elif calibration_unit == "cm":
        depth_m /= 100.0

    valid   = (depth_m > 0) & np.isfinite(depth_m)
    display = np.zeros(depth_m.shape, dtype=np.uint8)

    if np.any(valid):
        normalized       = np.clip(depth_m / max_depth_m, 0, 1)
        display          = (normalized * 255).astype(np.uint8)
        display[~valid]  = 0

    return cv2.applyColorMap(display, cv2.COLORMAP_PLASMA)


def get_center_depth(depth: np.ndarray,
                     calibration_unit: str = "mm") -> float | None:
    """Median depth in a 11x11 region around the image centre."""
    h, w = depth.shape
    cx, cy = w // 2, h // 2
    region = depth[cy - 5 : cy + 6, cx - 5 : cx + 6]
    valid  = region[np.isfinite(region) & (region > 0)]

    if valid.size == 0:
        return None

    dist = float(np.median(valid))
    if calibration_unit == "mm":
        dist /= 1000.0
    elif calibration_unit == "cm":
        dist /= 100.0
    return dist


def overlay_hud(
    img: np.ndarray,
    frame_idx: int,
    total_frames: int,
    fps_display: float,
    center_depth_m: float | None,
    delta_ms: float,
    paused: bool,
    speed: float,
) -> np.ndarray:
    """Draw HUD text onto a copy of img."""
    out = img.copy()
    h, w = out.shape[:2]

    # Centre crosshair
    cx, cy = w // 2, h // 2
    cv2.drawMarker(out, (cx, cy), (0, 255, 0),
                   cv2.MARKER_CROSS, 20, 2)

    # Depth at centre
    if center_depth_m is None:
        depth_text = "Depth: unavailable"
    else:
        depth_text = f"Depth: {center_depth_m:.2f} m"

    cv2.putText(out, depth_text,
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.putText(out, f"Frame delta: {delta_ms:.1f} ms",
                (20, 75), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(out, f"FPS: {fps_display:.1f}  Speed: {speed:.2f}x",
                (20, 108), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (200, 200, 200), 1, cv2.LINE_AA)

    # Progress bar
    bar_w  = w - 40
    bar_h  = 10
    bar_x  = 20
    bar_y  = h - 20
    filled = int(bar_w * frame_idx / max(total_frames - 1, 1))
    cv2.rectangle(out, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (80, 80, 80), -1)
    cv2.rectangle(out, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h),
                  (0, 200, 100), -1)

    cv2.putText(out,
                f"{'[PAUSED]' if paused else ''}  {frame_idx}/{total_frames}",
                (bar_x, bar_y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    return out


# ---------------------------------------------------------------------------
# Video writer helper
# ---------------------------------------------------------------------------

def make_video_writer(
    path: str,
    width: int,
    height: int,
    fps: float,
) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for '{path}'")
    return writer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stereo depth estimation from recorded left/right video files."
    )
    parser.add_argument("--left",  required=True,
                        help="Path to the left video file")
    parser.add_argument("--right", required=True,
                        help="Path to the right video file")
    parser.add_argument("--calib",
                        default="stereo_calibration.npz",
                        help="Path to stereo_calibration.npz (default: stereo_calibration.npz)")
    parser.add_argument("--downscale", type=float, default=0.5,
                        help="Downscale factor for disparity computation (default: 0.5). "
                             "Lower -> faster, higher -> more detail.")
    parser.add_argument("--calibration-unit", default="mm",
                        choices=["mm", "cm", "m"],
                        help="Unit used during calibration (default: mm)")
    parser.add_argument("--max-depth", type=float, default=5.0,
                        help="Maximum depth for colour scale in metres (default: 5.0)")
    parser.add_argument("--save-video", default="",
                        help="If set, export the result to this MP4 file (no preview needed)")
    parser.add_argument("--no-preview", action="store_true",
                        help="Skip the live preview window (useful when --save-video is set)")
    parser.add_argument("--frame-skip", type=int, default=0,
                        help="Process every Nth frame (0 = process all frames)")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Open video captures
    # ------------------------------------------------------------------
    left_path  = args.left
    right_path = args.right

    cap_l = cv2.VideoCapture(left_path)
    cap_r = cv2.VideoCapture(right_path)

    if not cap_l.isOpened():
        sys.exit(f"[ERROR] Cannot open left video: {left_path}")
    if not cap_r.isOpened():
        sys.exit(f"[ERROR] Cannot open right video: {right_path}")

    total_frames = int(min(cap_l.get(cv2.CAP_PROP_FRAME_COUNT),
                           cap_r.get(cv2.CAP_PROP_FRAME_COUNT)))
    src_fps      = cap_l.get(cv2.CAP_PROP_FPS) or 30.0
    frame_width  = int(cap_l.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap_l.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Left  video : {left_path}")
    print(f"Right video : {right_path}")
    print(f"Resolution  : {frame_width}x{frame_height}  FPS: {src_fps:.1f}")
    print(f"Total frames: {total_frames}")

    # ------------------------------------------------------------------
    # Load calibration + build depth estimator
    # ------------------------------------------------------------------
    if not Path(args.calib).exists():
        sys.exit(f"[ERROR] Calibration file not found: {args.calib}")

    print(f"\nLoading calibration from: {args.calib}")
    depth_estimator = StereoDepth(args.calib, downscale=args.downscale)
    print("Calibration loaded.\n")

    # ------------------------------------------------------------------
    # Video writer (optional)
    # ------------------------------------------------------------------
    writer: cv2.VideoWriter | None = None
    if args.save_video:
        # Output frame: [left_rectified | disparity_color] stacked side by side
        out_w = frame_width * 2
        out_h = frame_height
        writer = make_video_writer(args.save_video, out_w, out_h, src_fps)
        print(f"Saving output to: {args.save_video}  ({out_w}x{out_h} @ {src_fps} fps)\n")

    # ------------------------------------------------------------------
    # Playback state
    # ------------------------------------------------------------------
    paused        = False
    speed         = 1.0         # playback speed multiplier
    frame_idx     = 0
    fps_display   = 0.0
    t_last        = time.monotonic()
    save_snapshot = False

    snapshot_dir = Path("depth_snapshots")

    skip_counter = 0            # for --frame-skip

    print("Controls: Q/ESC=quit  SPACE=pause  S=snapshot  [=slower  ]=faster  R=restart\n")

    while True:
        if not paused:
            ok_l, frame_l = cap_l.read()
            ok_r, frame_r = cap_r.read()

            if not ok_l or not ok_r:
                print("End of video.")
                break

            frame_idx += 1

            # Optional frame skipping for speed
            if args.frame_skip > 0:
                skip_counter += 1
                if skip_counter % (args.frame_skip + 1) != 0:
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    continue

            # ----------------------------------------------------------
            # Depth estimation
            # ----------------------------------------------------------
            t0 = time.monotonic()

            left_rectified, disparity, depth = depth_estimator.process(
                frame_l, frame_r
            )

            t1 = time.monotonic()
            process_ms = (t1 - t0) * 1000.0

            # Timestamps from video positions (nanoseconds equivalent)
            ts_l = cap_l.get(cv2.CAP_PROP_POS_MSEC)
            ts_r = cap_r.get(cv2.CAP_PROP_POS_MSEC)
            delta_ms = abs(ts_l - ts_r)

            # ----------------------------------------------------------
            # Centre depth
            # ----------------------------------------------------------
            center_depth_m = get_center_depth(depth, args.calibration_unit)

            # ----------------------------------------------------------
            # Colour maps
            # ----------------------------------------------------------
            disp_color  = disparity_to_color(disparity)
            depth_color = depth_to_color(depth, args.max_depth, args.calibration_unit)

            # ----------------------------------------------------------
            # FPS counter
            # ----------------------------------------------------------
            t_now = time.monotonic()
            fps_display = 1.0 / max(t_now - t_last, 1e-6)
            t_last = t_now

            # Print occasionally
            if frame_idx % 30 == 0:
                depth_str = (f"{center_depth_m:.3f} m"
                             if center_depth_m is not None else "N/A")
                print(f"Frame {frame_idx:5d}/{total_frames}  "
                      f"depth={depth_str:>10s}  "
                      f"process={process_ms:6.1f} ms  "
                      f"delta={delta_ms:5.1f} ms")

            # ----------------------------------------------------------
            # Export frame to video file
            # ----------------------------------------------------------
            if writer is not None:
                combined = np.hstack((left_rectified, disp_color))
                writer.write(combined)

            # ----------------------------------------------------------
            # Preview windows
            # ----------------------------------------------------------
            if not args.no_preview:
                hud = overlay_hud(
                    left_rectified,
                    frame_idx, total_frames,
                    fps_display, center_depth_m,
                    delta_ms, paused, speed,
                )

                cv2.imshow("Rectified Left  (Q=quit  SPACE=pause  S=save)", hud)
                cv2.imshow("Disparity Map", disp_color)
                cv2.imshow("Depth Map (Plasma)", depth_color)

            # ----------------------------------------------------------
            # Snapshot
            # ----------------------------------------------------------
            if save_snapshot:
                snapshot_dir.mkdir(parents=True, exist_ok=True)
                tag = f"frame{frame_idx:05d}"
                cv2.imwrite(str(snapshot_dir / f"{tag}_left_rect.png"), left_rectified)
                cv2.imwrite(str(snapshot_dir / f"{tag}_disparity.png"), disp_color)
                cv2.imwrite(str(snapshot_dir / f"{tag}_depth.png"),     depth_color)
                np.save(str(snapshot_dir / f"{tag}_depth_raw.npy"), depth)
                print(f"  -> Snapshot saved to {snapshot_dir}/{tag}_*.png")
                save_snapshot = False

            # ----------------------------------------------------------
            # Throttle playback speed
            # ----------------------------------------------------------
            delay_ms = max(1, int(1000.0 / (src_fps * speed)))

        else:
            # Paused: just wait for a key
            delay_ms = 50

        # ------------------------------------------------------------------
        # Key handling
        # ------------------------------------------------------------------
        key = cv2.waitKey(delay_ms) & 0xFF

        if key in (ord("q"), 27):          # Q / ESC -> quit
            break

        elif key == ord(" "):              # SPACE -> pause/resume
            paused = not paused
            print("Paused." if paused else "Resumed.")

        elif key == ord("s"):              # S -> snapshot
            save_snapshot = True

        elif key == ord("]"):              # ] -> faster
            speed = min(speed * 2.0, 16.0)
            print(f"Speed: {speed:.2f}x")

        elif key == ord("["):              # [ -> slower
            speed = max(speed / 2.0, 0.125)
            print(f"Speed: {speed:.2f}x")

        elif key == ord("r"):              # R -> restart
            cap_l.set(cv2.CAP_PROP_POS_FRAMES, 0)
            cap_r.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_idx = 0
            print("Restarted.")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    cap_l.release()
    cap_r.release()

    if writer is not None:
        writer.release()
        print(f"\nOutput video saved to: {args.save_video}")

    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
