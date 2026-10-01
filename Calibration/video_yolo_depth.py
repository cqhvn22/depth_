"""
video_yolo_depth.py
===================

YOLO (best.pt) + stereo depth – luồng stream trực tiếp từ 2 camera.

Pipeline:
  • Live preview: nhận frame liên tục, chạy YOLO + depth, hiển thị kết quả
    trực quan theo thời gian thực.
  • Khi bấm T: chụp liên tiếp 10 khung hình, tổng hợp kết quả bằng TRUNG VỊ,
    rồi hiển thị bảng thông số chi tiết. Có thể bấm T nhiều lần.

Thông số mỗi detection:
    - Tên lớp + confidence
    - Khoảng cách đến vật (m)
    - Kích thước vật (rộng × cao, cm hoặc m)
    - Tọa độ tâm vật trong ảnh (px)
    - Thông tin burst: N frame, median vs mean

Controls:
    T        – Trigger burst 10 frame → phân tích + hiển thị chi tiết
    Q / ESC  – Thoát
    D        – Bật / Tắt cửa sổ depth map
    R        – Reset về live preview
    S        – Lưu snapshot kết quả hiện tại
    +/-      – Tăng / Giảm confidence threshold (±0.05)

Chạy:
    python video_yolo_depth.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Thêm thư mục Calibration vào sys.path để import StereoDepth
# ---------------------------------------------------------------------------
_CALIB_DIR = Path(__file__).resolve().parent
if str(_CALIB_DIR) not in sys.path:
    sys.path.insert(0, str(_CALIB_DIR))

_PYTHON_DIR = _CALIB_DIR.parent / "Python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from video_depth_estimation import (
    StereoDepth,
    disparity_to_color,
    depth_to_color,
)

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("[ERROR] ultralytics chưa được cài. Chạy: pip install ultralytics")

try:
    from camera import StereoCamera
except ImportError:
    sys.exit("[ERROR] Không tìm thấy camera.py trong thư mục Python/")


# ===========================================================================
#  C O N F I G
# ===========================================================================

CALIB_FILE: str = str(_CALIB_DIR / "stereo_calibration.npz")
MODEL_FILE:  str = str(_CALIB_DIR / "best.pt")
SNAPSHOT_DIR: str = "yolo_depth_snapshots"

LEFT_CAM_NUM:   int   = 0
RIGHT_CAM_NUM:  int   = 1
CAPTURE_WIDTH:  int   = 960
CAPTURE_HEIGHT: int   = 540
FRAMERATE:      int   = 30

DEPTH_DOWNSCALE: float = 0.5
CALIB_UNIT:      str   = "mm"
MAX_DEPTH_M:     float = 5.0

N_BURST: int   = 10      # số frame mỗi lần bấm T
CONF_INIT: float = 0.35  # confidence ban đầu


# ===========================================================================
#  Màu sắc
# ===========================================================================

_RNG = np.random.default_rng(42)
_CLASS_COLOURS: list[tuple[int, int, int]] = [
    tuple(int(c) for c in _RNG.integers(80, 230, 3))
    for _ in range(200)
]


def class_colour(class_id: int) -> tuple[int, int, int]:
    return _CLASS_COLOURS[int(class_id) % len(_CLASS_COLOURS)]


# ===========================================================================
#  Tiện ích vật lý
# ===========================================================================

def focal_length_from_Q(Q: np.ndarray) -> float:
    """Focal length (px) từ ma trận Q của stereoRectify."""
    return float(Q[2, 3])


def box_physical_size(
    depth_map: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    focal_px: float,
    calib_unit: str = "mm",
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Tính (khoảng cách m, chiều rộng m, chiều cao m) từ bounding box.
    Dùng IQR-median để lọc noise.
    """
    h_img, w_img = depth_map.shape[:2]
    x1c = max(0, x1);   y1c = max(0, y1)
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

    width_m  = ((x2c - x1c) * depth_m) / focal_px
    height_m = ((y2c - y1c) * depth_m) / focal_px

    return depth_m, width_m, height_m


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
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    else:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    # Crosshair
    arm = 12
    cv2.line(img, (cx - arm, cy), (cx + arm, cy), (0, 255, 0), 2)
    cv2.line(img, (cx, cy - arm), (cx, cy + arm), (0, 255, 0), 2)
    cv2.circle(img, (cx, cy), 4, (0, 255, 0), -1)

    # Label
    lines: list[str] = [label]
    lines.append(f"Dist : {fmt_m(dist_m)}")
    if width_m is not None and height_m is not None:
        lines.append(f"Size : {fmt_m(width_m)} x {fmt_m(height_m)}")
    else:
        lines.append("Size : N/A")

    font   = cv2.FONT_HERSHEY_SIMPLEX
    fscale = 0.50
    thick  = 1
    pad    = 4
    lh     = 18

    tw = max(cv2.getTextSize(ln, font, fscale, thick)[0][0] for ln in lines)
    th = lh * len(lines) + pad

    ly0 = y1 - th - pad
    if ly0 < 0:
        ly0 = y1 + pad

    cv2.rectangle(img, (x1, ly0), (x1 + tw + pad * 2, ly0 + th),
                  tuple(max(0, c - 50) for c in colour), -1)
    for i, ln in enumerate(lines):
        ty = ly0 + pad + lh * (i + 1) - 2
        cv2.putText(img, ln, (x1 + pad, ty),
                    font, fscale, (255, 255, 255), thick, cv2.LINE_AA)


def draw_hud(
    img: np.ndarray,
    state: str,
    n_det: int,
    fps: float,
    conf: float,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = [
        f"State: {state}   FPS: {fps:.1f}",
        f"Detections: {n_det}   Conf: {conf:.2f}",
        "T=Burst(10)  Q=Quit  D=Depth  R=Reset  S=Save  +/-=Conf",
    ]
    for i, ln in enumerate(lines):
        cv2.putText(img, ln, (14, 26 + i * 24), font, 0.54, (0, 0, 0),      3, cv2.LINE_AA)
        cv2.putText(img, ln, (14, 26 + i * 24), font, 0.54, (200, 240, 200), 1, cv2.LINE_AA)


def draw_burst_banner(
    img: np.ndarray,
    n_frames: int,
    n_valid: int,
    detections_summary: list[dict],
) -> None:
    """
    Vẽ banner kết quả burst ở phía dưới ảnh.
    detections_summary: list of {name, conf, dist_m, width_m, height_m}
    """
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    banner_h = 30 + 22 * max(1, len(detections_summary))
    y0 = h - banner_h
    overlay = img.copy()
    cv2.rectangle(overlay, (0, y0), (w, h), (20, 20, 50), -1)
    cv2.addWeighted(overlay, 0.78, img, 0.22, 0, img)

    # Tiêu đề
    header = (f"[BURST RESULT]  {n_valid}/{n_frames} frames hợp lệ  "
              f"– Trung vị {n_frames} frames")
    cv2.putText(img, header, (12, y0 + 20), font, 0.55,
                (100, 220, 255), 1, cv2.LINE_AA)

    if not detections_summary:
        cv2.putText(img, "Không phát hiện đối tượng.", (12, y0 + 42),
                    font, 0.50, (180, 180, 180), 1, cv2.LINE_AA)
        return

    for i, det in enumerate(detections_summary):
        ty = y0 + 42 + i * 22
        colour = class_colour(det.get("cls_id", i))
        text = (
            f"  [{i+1}] {det['name']} {det['conf']:.0%}  |  "
            f"Dist={fmt_m(det['dist_m'])}  |  "
            f"W={fmt_m(det['width_m'])}  H={fmt_m(det['height_m'])}  |  "
            f"Cx={det.get('cx_px', 'N/A')}px Cy={det.get('cy_px', 'N/A')}px"
        )
        cv2.putText(img, text, (12, ty), font, 0.46,
                    tuple(min(255, int(c * 1.4)) for c in colour),
                    1, cv2.LINE_AA)


# ===========================================================================
#  Burst capture
# ===========================================================================

class BurstResult:
    __slots__ = [
        "annotated", "depth_color",
        "detections_summary", "n_frames", "n_valid",
    ]

    def __init__(self) -> None:
        self.annotated:          Optional[np.ndarray] = None
        self.depth_color:        Optional[np.ndarray] = None
        self.detections_summary: list[dict]            = []
        self.n_frames:           int                   = 0
        self.n_valid:            int                   = 0


def run_burst(
    camera: StereoCamera,
    depth_est: StereoDepth,
    yolo: YOLO,
    focal_px: float,
    n_burst: int = N_BURST,
    conf: float  = CONF_INIT,
) -> BurstResult:
    """
    Chụp n_burst frame, chạy YOLO + depth trên từng frame,
    gom nhóm detection theo cls_id, tổng hợp bằng trung vị.
    """
    result   = BurstResult()
    result.n_frames = n_burst

    # cls_id → lists of (dist, width, height, cx, cy, conf)
    agg: dict[int, dict] = {}

    last_left_rect:   Optional[np.ndarray] = None
    last_depth_map:   Optional[np.ndarray] = None
    last_det_raw:     list[dict]            = []

    last_sequence: Optional[int] = None
    n_valid = 0

    print(f"\n[BURST] Bắt đầu chụp {n_burst} frame …")

    for frame_i in range(n_burst):
        # Chờ frame mới
        ok = False
        for _ in range(60):
            ok_, sf = camera.read(last_sequence=last_sequence, copy_frames=True)
            if ok_ and sf is not None:
                ok = True
                last_sequence = sf.sequence
                left_raw, right_raw = sf.left, sf.right
                break
            time.sleep(0.008)

        if not ok:
            print(f"  [BURST] Frame {frame_i+1}: timeout, bỏ qua.")
            continue

        n_valid += 1
        left_rect, _, depth_map = depth_est.process(left_raw, right_raw)

        results = yolo(left_rect, conf=conf, verbose=False)
        boxes   = results[0].boxes
        masks   = results[0].masks

        frame_dets: list[dict] = []

        if boxes is not None and len(boxes) > 0:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                cls_id  = int(box.cls[0])
                conf_v  = float(box.conf[0])
                name    = yolo.names[cls_id]
                cx_px   = (x1 + x2) // 2
                cy_px   = (y1 + y2) // 2

                mask_poly: Optional[np.ndarray] = None
                if masks is not None and i < len(masks):
                    xy = masks[i].xy
                    if xy is not None and len(xy) > 0 and len(xy[0]) >= 3:
                        mask_poly = xy[0].astype(np.int32)

                dist_m, w_m, h_m = box_physical_size(
                    depth_map, x1, y1, x2, y2, focal_px, CALIB_UNIT
                )

                frame_dets.append({
                    "cls_id": cls_id, "name": name,
                    "conf": conf_v,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "cx_px": cx_px, "cy_px": cy_px,
                    "dist_m": dist_m, "width_m": w_m, "height_m": h_m,
                    "mask_poly": mask_poly,
                })

                # Tích luỹ theo cls_id
                if cls_id not in agg:
                    agg[cls_id] = {
                        "name": name,
                        "confs": [], "dists": [],
                        "widths": [], "heights": [],
                        "cxs": [], "cys": [],
                    }
                agg[cls_id]["confs"].append(conf_v)
                if dist_m is not None:
                    agg[cls_id]["dists"].append(dist_m)
                if w_m is not None:
                    agg[cls_id]["widths"].append(w_m)
                if h_m is not None:
                    agg[cls_id]["heights"].append(h_m)
                agg[cls_id]["cxs"].append(cx_px)
                agg[cls_id]["cys"].append(cy_px)

        last_left_rect = left_rect
        last_depth_map = depth_map
        last_det_raw   = frame_dets
        print(f"  [BURST] Frame {frame_i+1}/{n_burst}: {len(frame_dets)} detection(s)")

    result.n_valid = n_valid

    # ── Tổng hợp bằng trung vị ────────────────────────────────────────
    summary: list[dict] = []
    for cls_id, data in agg.items():
        med_dist  = float(np.median(data["dists"]))   if data["dists"]   else None
        med_w     = float(np.median(data["widths"]))  if data["widths"]  else None
        med_h     = float(np.median(data["heights"])) if data["heights"] else None
        med_conf  = float(np.median(data["confs"]))   if data["confs"]   else 0.0
        med_cx    = int(np.median(data["cxs"]))       if data["cxs"]     else 0
        med_cy    = int(np.median(data["cys"]))       if data["cys"]     else 0
        summary.append({
            "cls_id": cls_id,
            "name":   data["name"],
            "conf":   med_conf,
            "dist_m":  med_dist,
            "width_m": med_w,
            "height_m": med_h,
            "cx_px": med_cx,
            "cy_px": med_cy,
        })

    result.detections_summary = summary

    # Console log chi tiết
    print(f"\n{'='*62}")
    print(f"  BURST RESULT  ({n_valid}/{n_burst} frames hợp lệ)")
    print(f"{'='*62}")
    if not summary:
        print("  [!] Không phát hiện đối tượng nào.")
    for i, det in enumerate(summary):
        print(f"  [{i+1}] {det['name']:<20s}  conf={det['conf']:.2f}")
        print(f"       Khoảng cách : {fmt_m(det['dist_m'])}")
        print(f"       Rộng × Cao  : {fmt_m(det['width_m'])} × {fmt_m(det['height_m'])}")
        print(f"       Tâm (px)    : Cx={det['cx_px']}  Cy={det['cy_px']}")
    print(f"{'='*62}\n")

    # ── Vẽ annotated frame cuối ──────────────────────────────────────
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

        draw_burst_banner(annotated, n_burst, n_valid, summary)
        result.annotated = annotated

    if last_depth_map is not None:
        result.depth_color = depth_to_color(last_depth_map, MAX_DEPTH_M, CALIB_UNIT)

    return result


# ===========================================================================
#  Main loop
# ===========================================================================

def main() -> None:
    # ── Kiểm tra file ──────────────────────────────────────────────────
    calib_path = Path(CALIB_FILE)
    model_path = Path(MODEL_FILE)

    if not calib_path.exists():
        sys.exit(f"[ERROR] Không tìm thấy calibration: {calib_path}")
    if not model_path.exists():
        sys.exit(f"[ERROR] Không tìm thấy model: {model_path}")

    # ── Mở camera ──────────────────────────────────────────────────────
    print("[INFO] Khởi động stereo camera …")
    camera = StereoCamera(
        left_camera_num=LEFT_CAM_NUM,
        right_camera_num=RIGHT_CAM_NUM,
        capture_width=CAPTURE_WIDTH,
        capture_height=CAPTURE_HEIGHT,
        framerate=FRAMERATE,
    )
    if not camera.open():
        sys.exit("[ERROR] Không mở được camera.")
    if not camera.start():
        camera.release()
        sys.exit("[ERROR] Không khởi động được luồng chụp.")

    # ── Load depth + YOLO ──────────────────────────────────────────────
    print(f"[INFO] Load calibration: {calib_path}")
    depth_est = StereoDepth(str(calib_path), downscale=DEPTH_DOWNSCALE)
    focal_px  = focal_length_from_Q(depth_est.Q)
    print(f"[INFO] Focal length Q[2,3] = {focal_px:.2f} px")

    print(f"[INFO] Load YOLO model: {model_path}")
    yolo = YOLO(str(model_path))
    print("[INFO] YOLO sẵn sàng.\n")

    # ── State ──────────────────────────────────────────────────────────
    WIN_MAIN  = "YOLO + Depth  [T=Burst  Q=Quit  D=Depth  R=Reset  S=Save  +/-=Conf]"
    WIN_DEPTH = "Depth Map"

    cv2.namedWindow(WIN_MAIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_MAIN, CAPTURE_WIDTH, CAPTURE_HEIGHT)

    show_depth    = True
    burst_result: Optional[BurstResult] = None
    state         = "STANDBY"
    fps_display   = 0.0
    t_last        = time.monotonic()
    last_sequence: Optional[int] = None
    n_det_live    = 0
    conf_thresh   = CONF_INIT
    snapshot_dir  = Path(SNAPSHOT_DIR)

    print("=" * 62)
    print("  YOLO + STEREO DEPTH – LIVE STREAM")
    print("=" * 62)
    print(f"  Model   : {model_path.name}")
    print(f"  Burst   : {N_BURST} frames  |  Tổng hợp: trung vị")
    print(f"  Conf    : {conf_thresh:.2f}")
    print("=" * 62)
    print("  T = Burst  |  Q = Thoát  |  D = Depth  |  R = Reset  |  S = Save")
    print("  + = Tăng conf  |  - = Giảm conf")
    print("=" * 62 + "\n")

    try:
        while True:
            t_now = time.monotonic()
            fps_display = 1.0 / max(t_now - t_last, 1e-6)
            t_last = t_now

            # ── Đọc frame ──────────────────────────────────────────────
            ok, stereo_frame = camera.read(
                last_sequence=last_sequence, copy_frames=False
            )

            if not ok or stereo_frame is None:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key in (ord("t"), ord("T")):
                    state = "CAPTURING"
                    burst_result = run_burst(
                        camera, depth_est, yolo, focal_px, N_BURST, conf_thresh
                    )
                    state = "RESULT"
                continue

            last_sequence = stereo_frame.sequence
            left_rect, _, depth_map = depth_est.process(
                stereo_frame.left, stereo_frame.right
            )

            # ── YOLO live ──────────────────────────────────────────────
            results_live = yolo(left_rect, conf=conf_thresh, verbose=False)
            boxes_live   = results_live[0].boxes
            masks_live   = results_live[0].masks
            n_det_live   = len(boxes_live) if boxes_live is not None else 0

            depth_color_live = depth_to_color(depth_map, MAX_DEPTH_M, CALIB_UNIT)

            # ── Chọn display ───────────────────────────────────────────
            if burst_result is not None and burst_result.annotated is not None:
                display       = burst_result.annotated.copy()
                depth_display = (burst_result.depth_color
                                 if burst_result.depth_color is not None
                                 else depth_color_live)
            else:
                display = left_rect.copy()

                if boxes_live is not None and n_det_live > 0:
                    for i, box in enumerate(boxes_live):
                        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                        cls_id = int(box.cls[0])
                        conf_v = float(box.conf[0])
                        name   = yolo.names[cls_id]

                        mask_poly = None
                        if masks_live is not None and i < len(masks_live):
                            xy = masks_live[i].xy
                            if xy is not None and len(xy) > 0 and len(xy[0]) >= 3:
                                mask_poly = xy[0].astype(np.int32)

                        dist_m, w_m, h_m = box_physical_size(
                            depth_map, x1, y1, x2, y2, focal_px, CALIB_UNIT
                        )
                        draw_detection(
                            display, x1, y1, x2, y2,
                            f"{name} {conf_v:.0%}",
                            dist_m, w_m, h_m,
                            class_colour(cls_id),
                            mask_poly,
                        )

                depth_display = depth_color_live

            draw_hud(display, state, n_det_live, fps_display, conf_thresh)

            cv2.imshow(WIN_MAIN, display)
            if show_depth:
                cv2.imshow(WIN_DEPTH, depth_display)
            else:
                cv2.destroyWindow(WIN_DEPTH)

            # ── Key handling ───────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            elif key in (ord("t"), ord("T")):
                state = "CAPTURING"
                cv2.waitKey(1)
                burst_result = run_burst(
                    camera, depth_est, yolo, focal_px, N_BURST, conf_thresh
                )
                state = "RESULT"

            elif key in (ord("d"), ord("D")):
                show_depth = not show_depth

            elif key in (ord("r"), ord("R")):
                burst_result = None
                state = "STANDBY"

            elif key in (ord("s"), ord("S")):
                snapshot_dir.mkdir(parents=True, exist_ok=True)
                ts = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(str(snapshot_dir / f"yolo_depth_{ts}.jpg"), display)
                if burst_result and burst_result.depth_color is not None:
                    cv2.imwrite(
                        str(snapshot_dir / f"yolo_depth_{ts}_depthmap.jpg"),
                        burst_result.depth_color,
                    )
                print(f"[SAVE] Đã lưu vào: {snapshot_dir}/yolo_depth_{ts}*.jpg")

            elif key == ord("+") or key == ord("="):
                conf_thresh = min(conf_thresh + 0.05, 0.95)
                burst_result = None
                state = "STANDBY"
                print(f"[CONF] Confidence → {conf_thresh:.2f}")

            elif key == ord("-") or key == ord("_"):
                conf_thresh = max(conf_thresh - 0.05, 0.05)
                burst_result = None
                state = "STANDBY"
                print(f"[CONF] Confidence → {conf_thresh:.2f}")

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("[INFO] Đã thoát.")


if __name__ == "__main__":
    main()
