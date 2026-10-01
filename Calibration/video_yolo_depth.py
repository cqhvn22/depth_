"""
video_yolo_depth.py
===================

YOLOv8-seg (best.pt) + stereo depth – đầu vào là 2 video quay từ 2 cam.

Chạy:
    python video_yolo_depth.py
    python video_yolo_depth.py --left L.mp4 --right R.mp4 --calib calib.npz

Controls:
    T        – Burst 10 frame liên tiếp từ vị trí hiện tại → phân tích
               chi tiết + hiển thị trung vị. Có thể bấm nhiều lần.
    Q / ESC  – Thoát
    SPACE    – Pause / Resume
    D        – Bật / Tắt cửa sổ depth map
    R        – Restart video từ đầu
    S        – Lưu snapshot frame hiện tại
    +  / -   – Tăng / Giảm confidence ±0.05
    ]  / [   – Tăng / Giảm tốc độ phát

Thông số mỗi detection (sau burst):
    - Tên lớp + confidence (trung vị 10 frame)
    - Khoảng cách đến vật (m)
    - Kích thước vật: Rộng × Cao (m/cm)
    - Tọa độ tâm vật trong ảnh (px)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Import StereoDepth từ cùng thư mục Calibration/
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from video_depth_estimation import (
    StereoDepth,
    disparity_to_color,
    depth_to_color,
)

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("[ERROR] ultralytics chưa được cài. Chạy: pip install ultralytics")


# ===========================================================================
#  C O N F I G  – chỉnh tại đây, không cần truyền argument
# ===========================================================================

_SCRIPT_DIR = Path(__file__).resolve().parent

# Đường dẫn mặc định (có thể override bằng --left, --right, --calib, --model)
DEFAULT_LEFT:  str = str(_SCRIPT_DIR / "recordings" / "left_20260930_164731.mp4")
DEFAULT_RIGHT: str = str(_SCRIPT_DIR / "recordings" / "right_20260930_164731.mp4")
DEFAULT_CALIB: str = str(_SCRIPT_DIR / "stereo_calibration.npz")
DEFAULT_MODEL: str = str(_SCRIPT_DIR / "best.pt")
SNAPSHOT_DIR:  str = "yolo_depth_snapshots"

DEPTH_DOWNSCALE: float = 0.5
CALIB_UNIT:      str   = "mm"
MAX_DEPTH_M:     float = 5.0

N_BURST:   int   = 10    # số frame liên tiếp mỗi lần bấm T
CONF_INIT: float = 0.35


# ===========================================================================
#  Màu sắc
# ===========================================================================

_RNG = np.random.default_rng(42)
_CLASS_COLOURS: list[tuple[int, int, int]] = [
    tuple(int(c) for c in _RNG.integers(80, 230, 3))
    for _ in range(200)
]


def class_colour(cls_id: int) -> tuple[int, int, int]:
    return _CLASS_COLOURS[int(cls_id) % len(_CLASS_COLOURS)]


# ===========================================================================
#  Tiện ích vật lý
# ===========================================================================

def focal_length_from_Q(Q: np.ndarray) -> float:
    return float(Q[2, 3])


def box_physical_size(
    depth_map: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    focal_px: float,
    calib_unit: str = "mm",
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """(khoảng cách m, rộng m, cao m) từ bounding box + depth map."""
    h_img, w_img = depth_map.shape[:2]
    x1c = max(0, x1);    y1c = max(0, y1)
    x2c = min(w_img-1, x2);  y2c = min(h_img-1, y2)

    roi   = depth_map[y1c:y2c, x1c:x2c]
    valid = roi[np.isfinite(roi) & (roi > 0)]
    if valid.size < 5:
        return None, None, None

    lo, hi = np.percentile(valid, 25), np.percentile(valid, 75)
    core   = valid[(valid >= lo) & (valid <= hi)]
    if core.size == 0:
        core = valid

    depth_raw = float(np.median(core))
    if calib_unit == "mm":
        depth_m = depth_raw / 1000.0
    elif calib_unit == "cm":
        depth_m = depth_raw / 100.0
    else:
        depth_m = depth_raw

    if depth_m <= 0 or not np.isfinite(depth_m):
        return None, None, None
    if focal_px <= 0:
        return depth_m, None, None

    w_m = ((x2c - x1c) * depth_m) / focal_px
    h_m = ((y2c - y1c) * depth_m) / focal_px
    return depth_m, w_m, h_m


def fmt_m(v: Optional[float]) -> str:
    if v is None:
        return "N/A"
    if v < 1.0:
        return f"{v * 100:.1f} cm"
    return f"{v:.3f} m"


# ===========================================================================
#  Vẽ overlay
# ===========================================================================

def draw_detection(
    img: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    label: str,
    dist_m: Optional[float],
    width_m: Optional[float],
    height_m: Optional[float],
    colour: tuple[int, int, int],
    mask_polygon: Optional[np.ndarray] = None,
) -> None:
    overlay = img.copy()
    alpha   = 0.25

    if mask_polygon is not None and len(mask_polygon) >= 3:
        pts = mask_polygon.reshape((-1, 1, 2)).astype(np.int32)
        cv2.fillPoly(overlay, [pts], colour)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        cv2.polylines(img, [pts], True, colour, 3, cv2.LINE_AA)
        bright = tuple(min(255, int(c * 1.5)) for c in colour)
        cv2.polylines(img, [pts], True, bright, 1, cv2.LINE_AA)
        M = cv2.moments(pts)
        cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else (x1 + x2) // 2
        cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else (y1 + y2) // 2
    else:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    arm = 12
    cv2.line(img, (cx - arm, cy), (cx + arm, cy), (0, 255, 0), 2)
    cv2.line(img, (cx, cy - arm), (cx, cy + arm), (0, 255, 0), 2)
    cv2.circle(img, (cx, cy), 4, (0, 255, 0), -1)

    lines = [label, f"Dist : {fmt_m(dist_m)}"]
    if width_m is not None and height_m is not None:
        lines.append(f"Size : {fmt_m(width_m)} x {fmt_m(height_m)}")
    else:
        lines.append("Size : N/A")

    font = cv2.FONT_HERSHEY_SIMPLEX
    fscale, thick, pad, lh = 0.50, 1, 4, 18
    tw = max(cv2.getTextSize(ln, font, fscale, thick)[0][0] for ln in lines)
    th = lh * len(lines) + pad
    ly0 = y1 - th - pad if y1 - th - pad >= 0 else y1 + pad
    cv2.rectangle(img, (x1, ly0), (x1 + tw + pad * 2, ly0 + th),
                  tuple(max(0, c - 50) for c in colour), -1)
    for i, ln in enumerate(lines):
        cv2.putText(img, ln, (x1 + pad, ly0 + pad + lh * (i + 1) - 2),
                    font, fscale, (255, 255, 255), thick, cv2.LINE_AA)


def draw_hud(
    img: np.ndarray,
    frame_idx: int,
    total_frames: int,
    fps: float,
    n_det: int,
    conf: float,
    paused: bool,
    speed: float,
    state: str,
) -> None:
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    lines = [
        f"{'[PAUSED]  ' if paused else ''}State: {state}   FPS: {fps:.1f}   Speed: {speed:.2f}x",
        f"Det: {n_det}   Conf: {conf:.2f}   Frame: {frame_idx}/{total_frames}",
        "T=Burst(10)  SPACE=Pause  Q=Quit  D=Depth  R=Restart  S=Save  +/-=Conf  ]/[=Speed",
    ]
    for i, ln in enumerate(lines):
        cv2.putText(img, ln, (14, 26 + i * 24), font, 0.50, (0, 0, 0),       3, cv2.LINE_AA)
        cv2.putText(img, ln, (14, 26 + i * 24), font, 0.50, (200, 240, 200), 1, cv2.LINE_AA)

    # Progress bar
    bar_x, bar_y = 14, h - 14
    bar_w = w - 28
    filled = int(bar_w * frame_idx / max(total_frames - 1, 1))
    cv2.rectangle(img, (bar_x, bar_y - 6), (bar_x + bar_w, bar_y + 4), (60, 60, 60), -1)
    cv2.rectangle(img, (bar_x, bar_y - 6), (bar_x + filled, bar_y + 4), (0, 180, 100), -1)


def draw_burst_banner(
    img: np.ndarray,
    n_frames: int,
    detections_summary: list[dict],
) -> None:
    """Banner kết quả burst ở phía dưới ảnh."""
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    n_rows = max(1, len(detections_summary))
    banner_h = 34 + 22 * n_rows
    y0 = h - banner_h

    overlay = img.copy()
    cv2.rectangle(overlay, (0, y0), (w, h), (18, 18, 45), -1)
    cv2.addWeighted(overlay, 0.80, img, 0.20, 0, img)

    header = (f"[BURST  T=10 frames | trung vị]  "
              f"{len(detections_summary)} đối tượng phát hiện")
    cv2.putText(img, header, (12, y0 + 20), font, 0.54,
                (100, 220, 255), 1, cv2.LINE_AA)

    if not detections_summary:
        cv2.putText(img, "  Không phát hiện đối tượng.", (12, y0 + 42),
                    font, 0.50, (180, 180, 180), 1, cv2.LINE_AA)
        return

    for i, det in enumerate(detections_summary):
        ty   = y0 + 42 + i * 22
        col  = class_colour(det.get("cls_id", i))
        bright = tuple(min(255, int(c * 1.35)) for c in col)
        text = (
            f"  [{i+1}] {det['name']:<18s}  conf={det['conf']:.2f}  |  "
            f"Dist={fmt_m(det['dist_m'])}  |  "
            f"W={fmt_m(det['width_m'])}  H={fmt_m(det['height_m'])}  |  "
            f"Cx={det.get('cx_px','?')}px  Cy={det.get('cy_px','?')}px"
        )
        cv2.putText(img, text, (12, ty), font, 0.46, bright, 1, cv2.LINE_AA)


# ===========================================================================
#  Burst – đọc N frame liên tiếp từ video
# ===========================================================================

class BurstResult:
    __slots__ = ["annotated", "depth_color", "detections_summary", "frame_end"]

    def __init__(self) -> None:
        self.annotated:          Optional[np.ndarray] = None
        self.depth_color:        Optional[np.ndarray] = None
        self.detections_summary: list[dict]            = []
        self.frame_end:          int                   = 0


def run_burst(
    cap_l: cv2.VideoCapture,
    cap_r: cv2.VideoCapture,
    depth_est: StereoDepth,
    yolo: YOLO,
    focal_px: float,
    frame_idx: int,
    n_burst: int = N_BURST,
    conf: float  = CONF_INIT,
) -> Optional[BurstResult]:
    """
    Đọc n_burst frame liên tiếp từ vị trí hiện tại trong video.
    Tổng hợp kết quả bằng trung vị theo từng class.
    """
    result = BurstResult()

    # cls_id → gom danh sách giá trị qua các frame
    agg: dict[int, dict] = {}

    last_left_rect:   Optional[np.ndarray] = None
    last_depth_map:   Optional[np.ndarray] = None
    last_det_raw:     list[dict]            = []
    frames_ok = 0

    print(f"\n[BURST] Chụp {n_burst} frame từ vị trí {frame_idx} …")

    for fi in range(n_burst):
        ok_l, frame_l = cap_l.read()
        ok_r, frame_r = cap_r.read()
        if not ok_l or not ok_r:
            print(f"  [BURST] Frame {fi+1}: hết video, dừng burst.")
            break

        frames_ok += 1
        left_rect, _, depth_map = depth_est.process(frame_l, frame_r)

        res   = yolo(left_rect, conf=conf, verbose=False)
        boxes = res[0].boxes
        masks = res[0].masks
        frame_dets: list[dict] = []

        if boxes is not None and len(boxes) > 0:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                cls_id = int(box.cls[0])
                conf_v = float(box.conf[0])
                name   = yolo.names[cls_id]
                cx_px  = (x1 + x2) // 2
                cy_px  = (y1 + y2) // 2

                mask_poly: Optional[np.ndarray] = None
                if masks is not None and i < len(masks):
                    xy = masks[i].xy
                    if xy is not None and len(xy) > 0 and len(xy[0]) >= 3:
                        mask_poly = xy[0].astype(np.int32)

                dist_m, w_m, h_m = box_physical_size(
                    depth_map, x1, y1, x2, y2, focal_px, CALIB_UNIT
                )

                frame_dets.append({
                    "cls_id": cls_id, "name": name, "conf": conf_v,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "cx_px": cx_px, "cy_px": cy_px,
                    "dist_m": dist_m, "width_m": w_m, "height_m": h_m,
                    "mask_poly": mask_poly,
                })

                if cls_id not in agg:
                    agg[cls_id] = {
                        "name": name,
                        "confs": [], "dists": [],
                        "widths": [], "heights": [],
                        "cxs": [], "cys": [],
                    }
                agg[cls_id]["confs"].append(conf_v)
                if dist_m  is not None: agg[cls_id]["dists"].append(dist_m)
                if w_m     is not None: agg[cls_id]["widths"].append(w_m)
                if h_m     is not None: agg[cls_id]["heights"].append(h_m)
                agg[cls_id]["cxs"].append(cx_px)
                agg[cls_id]["cys"].append(cy_px)

        last_left_rect = left_rect
        last_depth_map = depth_map
        last_det_raw   = frame_dets
        print(f"  [BURST] Frame {fi+1}/{n_burst}: {len(frame_dets)} detection(s)")

    result.frame_end = frame_idx + frames_ok

    # ── Tổng hợp trung vị ────────────────────────────────────────────
    summary: list[dict] = []
    for cls_id, data in agg.items():
        summary.append({
            "cls_id":   cls_id,
            "name":     data["name"],
            "conf":     float(np.median(data["confs"]))   if data["confs"]   else 0.0,
            "dist_m":   float(np.median(data["dists"]))   if data["dists"]   else None,
            "width_m":  float(np.median(data["widths"]))  if data["widths"]  else None,
            "height_m": float(np.median(data["heights"])) if data["heights"] else None,
            "cx_px":    int(np.median(data["cxs"]))       if data["cxs"]     else None,
            "cy_px":    int(np.median(data["cys"]))       if data["cys"]     else None,
        })

    result.detections_summary = summary

    # Console log
    print(f"\n{'='*64}")
    print(f"  BURST RESULT  ({frames_ok}/{n_burst} frames đọc được)")
    print(f"{'='*64}")
    if not summary:
        print("  [!] Không phát hiện đối tượng nào trong burst.")
    for i, det in enumerate(summary):
        print(f"  [{i+1}] {det['name']:<22s}  conf={det['conf']:.2f}")
        print(f"       Khoảng cách  : {fmt_m(det['dist_m'])}")
        print(f"       Rộng × Cao   : {fmt_m(det['width_m'])} × {fmt_m(det['height_m'])}")
        print(f"       Tâm (px)     : Cx={det['cx_px']}  Cy={det['cy_px']}")
    print(f"{'='*64}\n")

    # ── Vẽ frame cuối ────────────────────────────────────────────────
    if last_left_rect is not None:
        annotated = last_left_rect.copy()
        for det in last_det_raw:
            draw_detection(
                annotated,
                det["x1"], det["y1"], det["x2"], det["y2"],
                f"{det['name']} {det['conf']:.0%}",
                det["dist_m"], det["width_m"], det["height_m"],
                class_colour(det["cls_id"]),
                det["mask_poly"],
            )
        draw_burst_banner(annotated, n_burst, summary)
        result.annotated = annotated

    if last_depth_map is not None:
        result.depth_color = depth_to_color(last_depth_map, MAX_DEPTH_M, CALIB_UNIT)

    return result


# ===========================================================================
#  Main
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="YOLO (best.pt) + stereo depth từ 2 video file."
    )
    parser.add_argument("--left",  default=DEFAULT_LEFT,
                        help="Video trái (default: recordings/left_*.mp4)")
    parser.add_argument("--right", default=DEFAULT_RIGHT,
                        help="Video phải (default: recordings/right_*.mp4)")
    parser.add_argument("--calib", default=DEFAULT_CALIB,
                        help="stereo_calibration.npz")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="YOLO model (default: best.pt)")
    parser.add_argument("--conf", type=float, default=CONF_INIT,
                        help="Confidence threshold (default: 0.35)")
    parser.add_argument("--downscale", type=float, default=DEPTH_DOWNSCALE)
    parser.add_argument("--calibration-unit", default=CALIB_UNIT,
                        choices=["mm", "cm", "m"])
    parser.add_argument("--max-depth", type=float, default=MAX_DEPTH_M)
    parser.add_argument("--n-burst", type=int, default=N_BURST,
                        help="Số frame mỗi lần bấm T (default: 10)")
    args = parser.parse_args()

    # ── Mở video ──────────────────────────────────────────────────────
    cap_l = cv2.VideoCapture(args.left)
    cap_r = cv2.VideoCapture(args.right)

    if not cap_l.isOpened():
        sys.exit(f"[ERROR] Không mở được video trái: {args.left}")
    if not cap_r.isOpened():
        sys.exit(f"[ERROR] Không mở được video phải: {args.right}")

    total_frames = int(min(cap_l.get(cv2.CAP_PROP_FRAME_COUNT),
                           cap_r.get(cv2.CAP_PROP_FRAME_COUNT)))
    src_fps  = cap_l.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w  = int(cap_l.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h  = int(cap_l.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Left  : {args.left}")
    print(f"Right : {args.right}")
    print(f"Res   : {frame_w}x{frame_h}  FPS: {src_fps:.1f}  Frames: {total_frames}")

    # ── Load calibration + depth ───────────────────────────────────────
    if not Path(args.calib).exists():
        sys.exit(f"[ERROR] Không tìm thấy calibration: {args.calib}")
    print(f"\n[INFO] Load calibration: {args.calib}")
    depth_est = StereoDepth(args.calib, downscale=args.downscale)
    focal_px  = focal_length_from_Q(depth_est.Q)
    print(f"[INFO] Focal length Q[2,3] = {focal_px:.2f} px")

    # ── Load YOLO ─────────────────────────────────────────────────────
    if not Path(args.model).exists():
        sys.exit(f"[ERROR] Không tìm thấy model: {args.model}")
    print(f"[INFO] Load model: {args.model}")
    yolo = YOLO(args.model)
    print("[INFO] Sẵn sàng.\n")

    # ── State ─────────────────────────────────────────────────────────
    WIN_MAIN  = "YOLO + Depth  [T=Burst  SPACE=Pause  Q=Quit  D=Depth  R=Restart  S=Save]"
    WIN_DEPTH = "Depth Map"

    cv2.namedWindow(WIN_MAIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_MAIN, max(frame_w, 960), max(frame_h, 540))

    paused       = False
    speed        = 1.0
    frame_idx    = 0
    fps_display  = 0.0
    t_last       = time.monotonic()
    show_depth   = True
    conf_thresh  = args.conf
    state        = "STANDBY"
    snapshot_dir = Path(SNAPSHOT_DIR)
    n_burst      = args.n_burst
    calib_unit   = args.calibration_unit

    burst_result: Optional[BurstResult] = None

    # Giữ frame hiển thị cuối cùng khi paused
    last_display:      Optional[np.ndarray] = None
    last_depth_display: Optional[np.ndarray] = None

    print("=" * 66)
    print("  YOLO + STEREO DEPTH  –  Video file mode")
    print("=" * 66)
    print(f"  Model  : {Path(args.model).name}")
    print(f"  Burst  : {n_burst} frames  |  Tổng hợp: trung vị")
    print(f"  Conf   : {conf_thresh:.2f}")
    print("=" * 66)
    print("  T=Burst  SPACE=Pause  Q=Quit  D=Depth  R=Restart")
    print("  S=Save   +/-=Conf   ]/[=Speed")
    print("=" * 66 + "\n")

    while True:
        if not paused:
            ok_l, frame_l = cap_l.read()
            ok_r, frame_r = cap_r.read()

            if not ok_l or not ok_r:
                print("[INFO] Hết video.")
                paused = True
                key = cv2.waitKey(50) & 0xFF
                if key in (ord("q"), 27):
                    break
                continue

            frame_idx += 1

            # ── Depth + YOLO ──────────────────────────────────────────
            left_rect, disparity, depth_map = depth_est.process(frame_l, frame_r)
            results = yolo(left_rect, conf=conf_thresh, verbose=False)
            boxes   = results[0].boxes
            masks   = results[0].masks
            n_det   = len(boxes) if boxes is not None else 0

            # ── Annotate ──────────────────────────────────────────────
            if burst_result is not None and burst_result.annotated is not None:
                # Đóng băng frame burst
                display       = burst_result.annotated.copy()
                depth_display = (burst_result.depth_color
                                 if burst_result.depth_color is not None
                                 else depth_to_color(depth_map, args.max_depth, calib_unit))
            else:
                display = left_rect.copy()
                if boxes is not None and n_det > 0:
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                        cls_id = int(box.cls[0])
                        conf_v = float(box.conf[0])
                        name   = yolo.names[cls_id]

                        mask_poly = None
                        if masks is not None and i < len(masks):
                            xy = masks[i].xy
                            if xy is not None and len(xy) > 0 and len(xy[0]) >= 3:
                                mask_poly = xy[0].astype(np.int32)

                        dist_m, w_m, h_m = box_physical_size(
                            depth_map, x1, y1, x2, y2, focal_px, calib_unit
                        )
                        draw_detection(display, x1, y1, x2, y2,
                                       f"{name} {conf_v:.0%}",
                                       dist_m, w_m, h_m,
                                       class_colour(cls_id), mask_poly)
                depth_display = depth_to_color(depth_map, args.max_depth, calib_unit)

            t_now = time.monotonic()
            fps_display = 1.0 / max(t_now - t_last, 1e-6)
            t_last = t_now

            draw_hud(display, frame_idx, total_frames, fps_display,
                     n_det, conf_thresh, paused, speed, state)

            last_display       = display
            last_depth_display = depth_display

            cv2.imshow(WIN_MAIN, display)
            if show_depth:
                cv2.imshow(WIN_DEPTH, depth_display)
            else:
                cv2.destroyWindow(WIN_DEPTH)

            delay_ms = max(1, int(1000.0 / (src_fps * speed)))

        else:
            # Paused – hiển thị lại frame cuối
            if last_display is not None:
                disp = last_display.copy()
                draw_hud(disp, frame_idx, total_frames, fps_display,
                         0, conf_thresh, paused, speed, state)
                cv2.imshow(WIN_MAIN, disp)
                if show_depth and last_depth_display is not None:
                    cv2.imshow(WIN_DEPTH, last_depth_display)
            delay_ms = 50

        # ── Key handling ──────────────────────────────────────────────
        key = cv2.waitKey(delay_ms) & 0xFF

        if key in (ord("q"), 27):
            break

        elif key == ord(" "):
            paused = not paused
            print("Paused." if paused else "Resumed.")

        elif key in (ord("t"), ord("T")):
            # Burst: đọc n_burst frame từ vị trí hiện tại
            # Lưu vị trí để sau burst có thể tiếp tục
            state = "CAPTURING"
            if last_display is not None:
                cv2.putText(last_display, f"CAPTURING {n_burst} frames…",
                            (14, 100), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 200, 255), 2, cv2.LINE_AA)
                cv2.imshow(WIN_MAIN, last_display)
                cv2.waitKey(1)

            burst_result = run_burst(
                cap_l, cap_r, depth_est, yolo, focal_px,
                frame_idx, n_burst, conf_thresh,
            )
            if burst_result is not None:
                frame_idx = burst_result.frame_end
            state  = "RESULT"
            paused = True   # Đóng băng để xem kết quả

        elif key in (ord("d"), ord("D")):
            show_depth = not show_depth

        elif key in (ord("r"), ord("R")):
            cap_l.set(cv2.CAP_PROP_POS_FRAMES, 0)
            cap_r.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_idx    = 0
            burst_result = None
            state        = "STANDBY"
            paused       = False
            print("[INFO] Restart.")

        elif key in (ord("s"), ord("S")):
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            if last_display is not None:
                cv2.imwrite(str(snapshot_dir / f"yolo_{ts}.jpg"), last_display)
            if burst_result and burst_result.depth_color is not None:
                cv2.imwrite(str(snapshot_dir / f"yolo_{ts}_depth.jpg"),
                            burst_result.depth_color)
            print(f"[SAVE] {snapshot_dir}/yolo_{ts}*.jpg")

        elif key in (ord("+"), ord("=")):
            conf_thresh  = min(conf_thresh + 0.05, 0.95)
            burst_result = None
            state        = "STANDBY"
            print(f"[CONF] → {conf_thresh:.2f}")

        elif key in (ord("-"), ord("_")):
            conf_thresh  = max(conf_thresh - 0.05, 0.05)
            burst_result = None
            state        = "STANDBY"
            print(f"[CONF] → {conf_thresh:.2f}")

        elif key == ord("]"):
            speed = min(speed * 2.0, 16.0)
            print(f"[SPEED] {speed:.2f}x")

        elif key == ord("["):
            speed = max(speed / 2.0, 0.125)
            print(f"[SPEED] {speed:.2f}x")

        # Nếu đang hiển thị burst result và tiếp tục chạy: bấm phím bất kỳ để resume
        elif key != 255 and state == "RESULT":
            burst_result = None
            state        = "STANDBY"
            paused       = False

    # ── Cleanup ───────────────────────────────────────────────────────
    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()
    print("[INFO] Đã thoát.")


if __name__ == "__main__":
    main()
