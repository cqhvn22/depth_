"""
video_yolo_depth.py

YOLOv8n-seg object segmentation integrated with stereo depth estimation on
pre-recorded left/right video files.

For every detected object the pipeline reports:
  - Distance  : robust median depth inside the segmentation mask (metres)
  - Physical width  : estimated from pixel width  * depth / focal-length
  - Physical height : estimated from pixel height * depth / focal-length

Physical-size formula (thin-lens / pinhole model):
    W_real = (box_w_px * Z) / f_px
    H_real = (box_h_px * Z) / f_px

where Z is the median depth in the mask ROI and f_px is the focal length
extracted from the Q reprojection matrix (Q[2, 3]).

Requires a YOLO segmentation model (e.g. yolov8n-seg.pt) to draw true
object outlines. Falls back to bounding-box rectangles if no mask is
available.

Usage:
    python video_yolo_depth.py \\
        --left  recordings/left_20260930_164731.mp4 \\
        --right recordings/right_20260930_164731.mp4 \\
        --calib stereo_calibration.npz

    # Save output:
    python video_yolo_depth.py --left L.mp4 --right R.mp4 \\
        --calib stereo_calibration.npz --save-video output.mp4

Controls (preview window):
    Q / ESC  : Quit
    SPACE    : Pause / Resume
    S        : Save snapshot of current frame set
    [ / ]    : Decrease / Increase playback speed
    R        : Restart from the beginning
    D        : Toggle depth-map window
    T        : Cycle confidence threshold  (0.25 -> 0.35 -> 0.50 -> 0.25)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# Re-use the StereoDepth class from the existing pipeline
sys.path.insert(0, str(Path(__file__).parent))
from video_depth_estimation import (
    StereoDepth,
    disparity_to_color,
    depth_to_color,
)


# ---------------------------------------------------------------------------
# Colour palette – one colour per COCO class id (80 classes)
# ---------------------------------------------------------------------------
_RNG = np.random.default_rng(42)
_CLASS_COLOURS: list[tuple[int, int, int]] = [
    tuple(int(c) for c in _RNG.integers(80, 230, 3))
    for _ in range(80)
]


def class_colour(class_id: int) -> tuple[int, int, int]:
    return _CLASS_COLOURS[int(class_id) % len(_CLASS_COLOURS)]


# ---------------------------------------------------------------------------
# Physical-size estimation
# ---------------------------------------------------------------------------

def focal_length_from_Q(Q: np.ndarray) -> float:
    """Extract pixel focal length from the 4x4 Q reprojection matrix.

    Standard Q layout (cv2.stereoRectify):
        [[1,  0,   0,  -cx ],
         [0,  1,   0,  -cy ],
         [0,  0,   0,   f  ],
         [0,  0, -1/Tx, dX ]]
    So Q[2, 3] == f  (focal length in pixels, at calibration resolution).
    """
    return float(Q[2, 3])


def box_physical_size(
    depth_map: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    focal_px: float,
    calib_unit: str = "mm",
) -> tuple[float | None, float | None, float | None]:
    """Return (distance_m, width_m, height_m) for the bounding box.

    - distance : robust median of valid depths inside the box
    - width_m  : physical width  at that distance
    - height_m : physical height at that distance

    Returns (None, None, None) when no valid depth is available.
    """
    # Clamp to image bounds
    h_img, w_img = depth_map.shape[:2]
    x1c = max(0, x1);  y1c = max(0, y1)
    x2c = min(w_img - 1, x2);  y2c = min(h_img - 1, y2)

    roi = depth_map[y1c:y2c, x1c:x2c]
    valid = roi[np.isfinite(roi) & (roi > 0)]

    if valid.size < 5:          # too few valid pixels
        return None, None, None

    # Use the 25th–75th percentile median to exclude background leakage
    lo, hi = np.percentile(valid, 25), np.percentile(valid, 75)
    core = valid[(valid >= lo) & (valid <= hi)]
    if core.size == 0:
        core = valid

    depth_raw = float(np.median(core))   # in calibration unit

    # Convert to metres
    if calib_unit == "mm":
        depth_m = depth_raw / 1000.0
    elif calib_unit == "cm":
        depth_m = depth_raw / 100.0
    else:
        depth_m = depth_raw

    if depth_m <= 0 or not np.isfinite(depth_m):
        return None, None, None

    box_w_px = x2c - x1c
    box_h_px = y2c - y1c

    if focal_px <= 0:
        return depth_m, None, None

    width_m  = (box_w_px * depth_m) / focal_px
    height_m = (box_h_px * depth_m) / focal_px

    return depth_m, width_m, height_m


# ---------------------------------------------------------------------------
# Overlay helpers
# ---------------------------------------------------------------------------

# Confidence threshold cycle values
_CONF_CYCLE = [0.25, 0.35, 0.50]


def draw_detection(
    img: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    label: str,
    distance_m: float | None,
    width_m: float | None,
    height_m: float | None,
    colour: tuple[int, int, int],
    depth_map: np.ndarray,
    mask_polygon: np.ndarray | None = None,
) -> None:
    """Draw a single detection with true object contour (mask) overlay.

    If ``mask_polygon`` is provided (Nx2 int32 array of pixel coordinates),
    the object's actual shape is rendered with a semi-transparent fill and
    a solid outline.  Otherwise a plain bounding-box rectangle is used as
    fallback.
    """
    overlay = img.copy()
    alpha   = 0.25

    if mask_polygon is not None and len(mask_polygon) >= 3:
        pts = mask_polygon.reshape((-1, 1, 2)).astype(np.int32)

        # ---- Semi-transparent mask fill ----
        cv2.fillPoly(overlay, [pts], colour)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        # ---- Solid outline (2-px glow + 1-px sharp) ----
        cv2.polylines(img, [pts], isClosed=True, color=colour, thickness=3,
                      lineType=cv2.LINE_AA)
        bright = tuple(min(255, int(c * 1.5)) for c in colour)
        cv2.polylines(img, [pts], isClosed=True, color=bright, thickness=1,
                      lineType=cv2.LINE_AA)

        # Centre from mask centroid
        M = cv2.moments(pts)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    else:
        # ---- Fallback: bounding-box rectangle ----
        cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    # ---- Distance cross-hair at the object centre ----
    arm = 12
    cv2.line(img, (cx - arm, cy), (cx + arm, cy), (0, 255, 0), 2)
    cv2.line(img, (cx, cy - arm), (cx, cy + arm), (0, 255, 0), 2)
    cv2.circle(img, (cx, cy), 4, (0, 255, 0), -1)

    # ---- Build label lines ----
    lines: list[str] = [label]
    if distance_m is not None:
        lines.append(f"Dist : {distance_m:.2f} m")
    else:
        lines.append("Dist : N/A")

    if width_m is not None and height_m is not None:
        # Choose cm for small objects, m for large
        if max(width_m, height_m) < 1.0:
            lines.append(f"Size : {width_m * 100:.1f} x {height_m * 100:.1f} cm")
        else:
            lines.append(f"Size : {width_m:.2f} x {height_m:.2f} m")
    else:
        lines.append("Size : N/A")

    # ---- Draw label background + text ----
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.52
    thickness  = 1
    pad        = 4
    line_h     = 18

    text_w = max(
        cv2.getTextSize(ln, font, font_scale, thickness)[0][0]
        for ln in lines
    )
    text_h_total = line_h * len(lines) + pad

    # Try to place label above the box; fall back to inside
    label_y0 = y1 - text_h_total - pad
    if label_y0 < 0:
        label_y0 = y1 + pad

    cv2.rectangle(
        img,
        (x1, label_y0),
        (x1 + text_w + pad * 2, label_y0 + text_h_total),
        colour, -1,
    )
    for i, ln in enumerate(lines):
        ty = label_y0 + pad + line_h * (i + 1) - 2
        cv2.putText(img, ln, (x1 + pad, ty),
                    font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def overlay_hud(
    img: np.ndarray,
    frame_idx: int,
    total_frames: int,
    fps_display: float,
    n_det: int,
    conf_thresh: float,
    paused: bool,
    speed: float,
) -> np.ndarray:
    """Draw global HUD (FPS, frame counter, progress bar)."""
    out = img.copy()
    h, w = out.shape[:2]

    # Top-left info block
    info_lines = [
        f"FPS: {fps_display:.1f}   Speed: {speed:.2f}x",
        f"Detections: {n_det}   Conf: {conf_thresh:.2f}  (T to cycle)",
        f"{'[PAUSED]' if paused else ''}",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, ln in enumerate(info_lines):
        cv2.putText(out, ln, (14, 28 + i * 26),
                    font, 0.65, (200, 255, 200), 2, cv2.LINE_AA)
        cv2.putText(out, ln, (14, 28 + i * 26),
                    font, 0.65, (20, 20, 20), 1, cv2.LINE_AA)

    # Bottom progress bar
    bar_w  = w - 40
    bar_h  = 10
    bar_x  = 20
    bar_y  = h - 20
    filled = int(bar_w * frame_idx / max(total_frames - 1, 1))
    cv2.rectangle(out, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    cv2.rectangle(out, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h), (0, 200, 100), -1)
    cv2.putText(out, f"{frame_idx}/{total_frames}",
                (bar_x, bar_y - 6), font, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

    return out


# ---------------------------------------------------------------------------
# Video writer helper
# ---------------------------------------------------------------------------

def make_video_writer(path: str, width: int, height: int, fps: float) -> cv2.VideoWriter:
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
        description="YOLOv8n detection + stereo depth: distance and physical size."
    )
    parser.add_argument("--left",  default=r"D:\Lenna-Stereo-Camera\Calibration\recordings\left_20260930_164731.mp4",
                        help="Path to left video file (default: left_20260930_164731.mp4)")
    parser.add_argument("--right", default=r"D:\Lenna-Stereo-Camera\Calibration\recordings\right_20260930_164731.mp4",
                        help="Path to right video file (default: right_20260930_164731.mp4)")
    parser.add_argument("--calib", default="stereo_calibration.npz",
                        help="Path to stereo_calibration.npz (default: stereo_calibration.npz)")
    parser.add_argument("--model", default="yolov8n-seg.pt",
                        help="YOLO segmentation model weights (default: yolov8n-seg.pt)")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="YOLO confidence threshold (default: 0.35)")
    parser.add_argument("--downscale", type=float, default=0.5,
                        help="Downscale factor for disparity computation (default: 0.5)")
    parser.add_argument("--calibration-unit", default="mm",
                        choices=["mm", "cm", "m"],
                        help="Unit used during calibration (default: mm)")
    parser.add_argument("--max-depth", type=float, default=5.0,
                        help="Max depth for colour scale in metres (default: 5.0)")
    parser.add_argument("--save-video", default="",
                        help="Export annotated result to this MP4 file")
    parser.add_argument("--no-preview", action="store_true",
                        help="Disable preview windows (use with --save-video)")
    parser.add_argument("--frame-skip", type=int, default=0,
                        help="Skip every N frames for speed (0 = process all)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Open video captures
    # ------------------------------------------------------------------
    cap_l = cv2.VideoCapture(args.left)
    cap_r = cv2.VideoCapture(args.right)

    if not cap_l.isOpened():
        sys.exit(f"[ERROR] Cannot open left video: {args.left}")
    if not cap_r.isOpened():
        sys.exit(f"[ERROR] Cannot open right video: {args.right}")

    total_frames = int(min(cap_l.get(cv2.CAP_PROP_FRAME_COUNT),
                           cap_r.get(cv2.CAP_PROP_FRAME_COUNT)))
    src_fps      = cap_l.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w      = int(cap_l.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h      = int(cap_l.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Left  video : {args.left}")
    print(f"Right video : {args.right}")
    print(f"Resolution  : {frame_w}x{frame_h}  FPS: {src_fps:.1f}")
    print(f"Total frames: {total_frames}")

    # ------------------------------------------------------------------
    # Load calibration + build depth estimator
    # ------------------------------------------------------------------
    if not Path(args.calib).exists():
        sys.exit(f"[ERROR] Calibration file not found: {args.calib}")

    print(f"\nLoading calibration from: {args.calib}")
    depth_est = StereoDepth(args.calib, downscale=args.downscale)

    # Extract focal length from Q matrix for physical-size calculations
    focal_px = focal_length_from_Q(depth_est.Q)
    print(f"Focal length (Q[2,3]) : {focal_px:.2f} px  (at calibration resolution)")
    print("Calibration loaded.\n")

    # ------------------------------------------------------------------
    # Load YOLO model
    # ------------------------------------------------------------------
    print(f"Loading YOLO model : {args.model}")
    yolo = YOLO(args.model)
    print("YOLO model loaded.\n")

    # ------------------------------------------------------------------
    # Optional video writer
    # ------------------------------------------------------------------
    writer: cv2.VideoWriter | None = None
    if args.save_video:
        out_w = depth_est._calib_w * 2
        out_h = depth_est._calib_h
        writer = make_video_writer(args.save_video, out_w, out_h, src_fps)
        print(f"Saving output to: {args.save_video}  ({out_w}x{out_h} @ {src_fps} fps)\n")

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    paused        = False
    speed         = 1.0
    frame_idx     = 0
    fps_display   = 0.0
    t_last        = time.monotonic()
    save_snapshot = False
    show_depth    = True
    skip_counter  = 0
    conf_thresh   = args.conf
    conf_cycle_i  = _CONF_CYCLE.index(min(_CONF_CYCLE, key=lambda x: abs(x - conf_thresh)))

    snapshot_dir = Path("yolo_depth_snapshots")

    print("Controls: Q/ESC=quit  SPACE=pause  S=snapshot  [=slower  ]=faster")
    print("          R=restart  D=toggle-depth-window  T=cycle-confidence\n")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    while True:
        if not paused:
            ok_l, frame_l = cap_l.read()
            ok_r, frame_r = cap_r.read()

            if not ok_l or not ok_r:
                print("End of video.")
                break

            frame_idx += 1

            # Frame skipping
            if args.frame_skip > 0:
                skip_counter += 1
                if skip_counter % (args.frame_skip + 1) != 0:
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    continue

            # --------------------------------------------------------------
            # 1. Stereo depth estimation
            # --------------------------------------------------------------
            t0 = time.monotonic()
            left_rect, disparity, depth_map = depth_est.process(frame_l, frame_r)
            t1 = time.monotonic()
            depth_ms = (t1 - t0) * 1000.0

            # --------------------------------------------------------------
            # 2. YOLOv8n inference on the rectified left image
            # --------------------------------------------------------------
            t2 = time.monotonic()
            results = yolo(left_rect, conf=conf_thresh, verbose=False)
            t3 = time.monotonic()
            yolo_ms = (t3 - t2) * 1000.0

            detections = results[0].boxes  # ultralytics Boxes object

            # --------------------------------------------------------------
            # 3. Draw detections with distance + physical size
            # --------------------------------------------------------------
            annotated = left_rect.copy()
            n_det = len(detections) if detections is not None else 0

            # Extract segmentation masks (None for non-seg models)
            seg_masks = results[0].masks   # ultralytics Masks or None

            if detections is not None and n_det > 0:
                for i, box in enumerate(detections):
                    x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                    cls_id  = int(box.cls[0])
                    conf    = float(box.conf[0])
                    name    = yolo.names[cls_id]
                    colour  = class_colour(cls_id)

                    # ---- Extract mask polygon for this detection ----
                    mask_polygon: np.ndarray | None = None
                    if seg_masks is not None and i < len(seg_masks):
                        # xy: list of (N,2) float arrays in pixel coordinates
                        xy = seg_masks[i].xy
                        if xy is not None and len(xy) > 0 and len(xy[0]) >= 3:
                            mask_polygon = xy[0].astype(np.int32)

                    dist_m, w_m, h_m = box_physical_size(
                        depth_map, x1, y1, x2, y2, focal_px, args.calibration_unit
                    )

                    draw_detection(
                        annotated,
                        x1, y1, x2, y2,
                        f"{name} {conf:.0%}",
                        dist_m, w_m, h_m,
                        colour,
                        depth_map,
                        mask_polygon,
                    )

                    # Console log every 30 frames
                    if frame_idx % 30 == 0:
                        dist_str = f"{dist_m:.2f} m" if dist_m else "N/A"
                        size_str = (f"{w_m*100:.1f}x{h_m*100:.1f} cm"
                                    if w_m and h_m else "N/A")
                        print(f"  [{name:15s}] dist={dist_str:>8s}  "
                              f"size={size_str}  conf={conf:.2f}")

            # HUD overlay
            t_now = time.monotonic()
            fps_display = 1.0 / max(t_now - t_last, 1e-6)
            t_last = t_now

            annotated = overlay_hud(
                annotated, frame_idx, total_frames,
                fps_display, n_det, conf_thresh, paused, speed,
            )

            if frame_idx % 30 == 0:
                print(f"Frame {frame_idx:5d}/{total_frames}  "
                      f"depth={depth_ms:5.1f} ms  yolo={yolo_ms:5.1f} ms  "
                      f"det={n_det}")

            # --------------------------------------------------------------
            # 4. Colour maps for visualisation
            # --------------------------------------------------------------
            disp_color  = disparity_to_color(disparity)
            depth_color = depth_to_color(depth_map, args.max_depth, args.calibration_unit)

            # --------------------------------------------------------------
            # 5. Export frame
            # --------------------------------------------------------------
            if writer is not None:
                combined = np.hstack((annotated, disp_color))
                # Resize to expected output size if needed
                exp_w = depth_est._calib_w * 2
                exp_h = depth_est._calib_h
                if combined.shape[1] != exp_w or combined.shape[0] != exp_h:
                    combined = cv2.resize(combined, (exp_w, exp_h))
                writer.write(combined)

            # --------------------------------------------------------------
            # 6. Preview windows
            # --------------------------------------------------------------
            if not args.no_preview:
                cv2.imshow("YOLOv8 + Stereo Depth  [Q=quit SPACE=pause S=snap T=conf]",
                           annotated)
                if show_depth:
                    cv2.imshow("Depth Map (Plasma)", depth_color)
                    cv2.imshow("Disparity Map", disp_color)
                else:
                    cv2.destroyWindow("Depth Map (Plasma)")
                    cv2.destroyWindow("Disparity Map")

            # --------------------------------------------------------------
            # 7. Snapshot
            # --------------------------------------------------------------
            if save_snapshot:
                snapshot_dir.mkdir(parents=True, exist_ok=True)
                tag = f"frame{frame_idx:05d}"
                cv2.imwrite(str(snapshot_dir / f"{tag}_annotated.png"), annotated)
                cv2.imwrite(str(snapshot_dir / f"{tag}_disparity.png"), disp_color)
                cv2.imwrite(str(snapshot_dir / f"{tag}_depth.png"), depth_color)
                np.save(str(snapshot_dir / f"{tag}_depth_raw.npy"), depth_map)
                print(f"  Snapshot -> {snapshot_dir}/{tag}_*.png")
                save_snapshot = False

            # Throttle
            delay_ms = max(1, int(1000.0 / (src_fps * speed)))

        else:
            delay_ms = 50

        # ------------------------------------------------------------------
        # Key handling
        # ------------------------------------------------------------------
        key = cv2.waitKey(delay_ms) & 0xFF

        if key in (ord("q"), 27):           # Q / ESC
            break
        elif key == ord(" "):               # SPACE – pause/resume
            paused = not paused
            print("Paused." if paused else "Resumed.")
        elif key == ord("s"):               # S – snapshot
            save_snapshot = True
        elif key == ord("]"):               # ] – faster
            speed = min(speed * 2.0, 16.0)
            print(f"Speed: {speed:.2f}x")
        elif key == ord("["):               # [ – slower
            speed = max(speed / 2.0, 0.125)
            print(f"Speed: {speed:.2f}x")
        elif key == ord("r"):               # R – restart
            cap_l.set(cv2.CAP_PROP_POS_FRAMES, 0)
            cap_r.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_idx = 0
            print("Restarted.")
        elif key == ord("d"):               # D – toggle depth windows
            show_depth = not show_depth
            print(f"Depth windows: {'ON' if show_depth else 'OFF'}")
        elif key == ord("t"):               # T – cycle confidence
            conf_cycle_i = (conf_cycle_i + 1) % len(_CONF_CYCLE)
            conf_thresh  = _CONF_CYCLE[conf_cycle_i]
            print(f"Confidence threshold: {conf_thresh:.2f}")

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
