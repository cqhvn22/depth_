"""
perch_detector.py
=================

Drone Perch Landing Feasibility Detector – Raspberry Pi 5 / Picamera2
-----------------------------------------------------------------------
Chạy thẳng bằng:
    python3 perch_detector.py

Toàn bộ tham số được cài sẵn trong khối CONFIG bên dưới.
Không cần truyền tham số từ terminal.

Controls:
    T        – Trigger burst capture + phân tích
    Q / ESC  – Thoát
    D        – Bật / Tắt cửa sổ depth map
    R        – Reset về live preview
    S        – Lưu snapshot kết quả hiện tại
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
_REPO_ROOT = Path(__file__).resolve().parent.parent   # Lenna-Stereo-Camera/
_CALIB_DIR = _REPO_ROOT / "Calibration"
if str(_CALIB_DIR) not in sys.path:
    sys.path.insert(0, str(_CALIB_DIR))

from video_depth_estimation import (
    StereoDepth,
    depth_to_color,
)

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("[ERROR] ultralytics not installed. Run: pip install ultralytics")


# ===========================================================================
# ██████████████████████   C O N F I G   ████████████████████████████████
# Chỉnh tất cả tham số TẠI ĐÂY – không cần truyền gì từ terminal
# ===========================================================================

# ---------- Đường dẫn file -----------------------------------------------
CALIB_FILE: str = str(_CALIB_DIR / "stereo_calibration.npz")  # file calibration
MODEL_FILE: str = str(_CALIB_DIR / "best.pt")                  # model YOLOv8n
SNAPSHOT_DIR: str = "perch_snapshots"                           # thư mục lưu ảnh

# ---------- Camera (Picamera2 trên Raspberry Pi 5) -----------------------
LEFT_CAM_NUM:    int = 0     # cổng CAM0 → camera trái
RIGHT_CAM_NUM:   int = 1     # cổng CAM1 → camera phải
CAPTURE_WIDTH:   int = 960   # độ phân giải chụp (px)
CAPTURE_HEIGHT:  int = 540

# ---------- Stereo depth -------------------------------------------------
DEPTH_DOWNSCALE: float = 0.5   # hệ số thu nhỏ ảnh khi tính disparity (0.5 = nhanh)
CALIB_UNIT:      str   = "mm"  # đơn vị của file calibration (mm / cm / m)
MAX_DEPTH_M:     float = 5.0   # giới hạn hiển thị depth colormap (mét)

# ---------- YOLO ---------------------------------------------------------
YOLO_CONF:    float = 0.35   # ngưỡng confidence

# ---------- Burst (số frame thu thập mỗi lần bấm T) ---------------------
N_BURST: int = 7

# ---------- Ngưỡng đánh giá thanh ngang (perch bar) ---------------------
# Kích thước hợp lệ (đơn vị: mét)
MIN_LENGTH_M: float = 0.10   # chiều dài tối thiểu (dimension lớn hơn của box)
MAX_LENGTH_M: float = 2.00   # chiều dài tối đa
MIN_WIDTH_M:  float = 0.02   # chiều rộng tối thiểu (dimension nhỏ hơn)
MAX_WIDTH_M:  float = 0.20   # chiều rộng tối đa
# Khoảng cách đến thanh hợp lệ (mét)
MIN_DIST_M:   float = 0.20
MAX_DIST_M:   float = 5.00

# ---------- Hiển thị -----------------------------------------------------
SHOW_DEPTH_ON_START: bool = True   # True = mở cửa sổ depth map ngay khi khởi động
WINDOW_WIDTH:        int  = 960
WINDOW_HEIGHT:       int  = 540


# ===========================================================================
# Màu sắc
# ===========================================================================

COLOR_OK     = (0, 220, 60)     # xanh lá → đáp được
COLOR_FAIL   = (0, 60, 240)     # đỏ      → không đáp được
COLOR_DETECT = (255, 180, 0)    # vàng    → phát hiện nhưng chưa có depth
COLOR_HUD_BG = (15, 15, 30)     # nền HUD tối


# ===========================================================================
# Hàm tiện ích
# ===========================================================================

def focal_length_from_Q(Q: np.ndarray) -> float:
    """Lấy focal length (px) từ ma trận Q của stereoRectify."""
    return float(Q[2, 3])


def box_physical_size(
    depth_map: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    focal_px: float,
    calib_unit: str = "mm",
) -> tuple[float | None, float | None, float | None]:
    """
    Tính (khoảng cách, chiều rộng, chiều cao) từ bounding box và depth map.

    Trả về (None, None, None) nếu không đủ điểm depth hợp lệ.
    """
    h_img, w_img = depth_map.shape[:2]
    x1c = max(0, x1);  y1c = max(0, y1)
    x2c = min(w_img - 1, x2);  y2c = min(h_img - 1, y2)

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

    box_w_px = x2c - x1c
    box_h_px = y2c - y1c

    if focal_px <= 0:
        return depth_m, None, None

    width_m  = (box_w_px * depth_m) / focal_px
    height_m = (box_h_px * depth_m) / focal_px

    return depth_m, width_m, height_m


def assess_perch(
    dist_m:   float | None,
    width_m:  float | None,
    height_m: float | None,
) -> tuple[bool | None, str]:
    """
    Đánh giá xem thanh ngang có đáp được không.

    Returns:
        (True = đáp được, False = không, None = thiếu dữ liệu)
        Chuỗi lý do/thông báo.
    """
    if dist_m is None or width_m is None or height_m is None:
        return None, "Không đủ dữ liệu depth"

    length_m = max(width_m, height_m)   # chiều dài thanh = dimension lớn hơn
    bar_w_m  = min(width_m, height_m)   # chiều rộng thanh = dimension nhỏ hơn

    reasons = []
    ok = True

    if not (MIN_DIST_M <= dist_m <= MAX_DIST_M):
        ok = False
        reasons.append(f"Khoảng cách {dist_m:.2f}m ngoài [{MIN_DIST_M:.2f}, {MAX_DIST_M:.2f}]m")

    if not (MIN_LENGTH_M <= length_m <= MAX_LENGTH_M):
        ok = False
        reasons.append(f"Chiều dài {length_m*100:.1f}cm ngoài [{MIN_LENGTH_M*100:.0f}, {MAX_LENGTH_M*100:.0f}]cm")

    if not (MIN_WIDTH_M <= bar_w_m <= MAX_WIDTH_M):
        ok = False
        reasons.append(f"Chiều rộng {bar_w_m*100:.1f}cm ngoài [{MIN_WIDTH_M*100:.0f}, {MAX_WIDTH_M*100:.0f}]cm")

    if ok:
        return True, "✓ ĐÁP ĐƯỢC"
    else:
        return False, "✗ KHÔNG ĐÁP ĐƯỢC: " + "; ".join(reasons)


def fmt_m(val: float | None) -> str:
    """Định dạng giá trị mét thành chuỗi dễ đọc."""
    if val is None:
        return "N/A"
    if val < 1.0:
        return f"{val * 100:.1f} cm"
    return f"{val:.3f} m"


# ===========================================================================
# Vẽ overlay lên ảnh
# ===========================================================================

def draw_perch_detection(
    img: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    label: str,
    dist_m: float | None,
    width_m: float | None,
    height_m: float | None,
    verdict: bool | None,
    mask_polygon: np.ndarray | None = None,
) -> None:
    """Vẽ detection box/mask, nhãn kích thước, verdict lên ảnh."""
    if verdict is True:
        colour = COLOR_OK
    elif verdict is False:
        colour = COLOR_FAIL
    else:
        colour = COLOR_DETECT

    overlay = img.copy()
    alpha   = 0.28

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
    arm = 14
    cv2.line(img, (cx - arm, cy), (cx + arm, cy), (0, 255, 180), 2)
    cv2.line(img, (cx, cy - arm), (cx, cy + arm), (0, 255, 180), 2)
    cv2.circle(img, (cx, cy), 5, (0, 255, 180), -1)

    # Thông tin nhãn
    length_m = max(width_m, height_m) if (width_m and height_m) else None
    bar_w_m  = min(width_m, height_m) if (width_m and height_m) else None

    lines = [
        label,
        f"Dist  : {fmt_m(dist_m)}",
        f"Length: {fmt_m(length_m)}",
        f"Width : {fmt_m(bar_w_m)}",
    ]

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.52
    thickness  = 1
    pad        = 4
    line_h     = 19

    text_w = max(cv2.getTextSize(ln, font, font_scale, thickness)[0][0] for ln in lines)
    text_h_total = line_h * len(lines) + pad

    label_y0 = y1 - text_h_total - pad
    if label_y0 < 0:
        label_y0 = y1 + pad

    cv2.rectangle(
        img,
        (x1, label_y0),
        (x1 + text_w + pad * 2, label_y0 + text_h_total),
        tuple(max(0, c - 60) for c in colour), -1,
    )
    for i, ln in enumerate(lines):
        ty = label_y0 + pad + line_h * (i + 1) - 2
        cv2.putText(img, ln, (x1 + pad, ty), font, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)


def draw_verdict_banner(
    img: np.ndarray,
    verdict: bool | None,
    reason: str,
    dist_m: float | None,
    length_m: float | None,
    bar_w_m: float | None,
) -> None:
    """Vẽ banner verdict + tóm tắt số liệu ở phía dưới ảnh."""
    h, w = img.shape[:2]
    banner_h = 90
    y0 = h - banner_h

    # Nền mờ
    overlay = img.copy()
    if verdict is True:
        bg = (10, 80, 10)
    elif verdict is False:
        bg = (60, 10, 10)
    else:
        bg = (30, 30, 60)
    cv2.rectangle(overlay, (0, y0), (w, h), bg, -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Dòng verdict lớn
    v_colour = COLOR_OK if verdict is True else (COLOR_FAIL if verdict is False else (200, 200, 200))
    cv2.putText(img, reason, (12, y0 + 30), font, 0.75, v_colour, 2, cv2.LINE_AA)

    # Dòng số liệu
    stats = (
        f"Dist={fmt_m(dist_m)}  "
        f"Length={fmt_m(length_m)}  "
        f"Width={fmt_m(bar_w_m)}"
    )
    cv2.putText(img, stats, (12, y0 + 58), font, 0.58, (210, 220, 255), 1, cv2.LINE_AA)

    # Thresholds reminder
    thr = (
        f"Thresholds — Dist:[{MIN_DIST_M*100:.0f},{MAX_DIST_M*100:.0f}]cm  "
        f"Len:[{MIN_LENGTH_M*100:.0f},{MAX_LENGTH_M*100:.0f}]cm  "
        f"W:[{MIN_WIDTH_M*100:.0f},{MAX_WIDTH_M*100:.0f}]cm"
    )
    cv2.putText(img, thr, (12, y0 + 80), font, 0.42, (150, 160, 180), 1, cv2.LINE_AA)


def draw_depth_annotation(
    depth_color: np.ndarray,
    detections_info: list[dict],
) -> np.ndarray:
    """Vẽ nhãn khoảng cách của từng detection lên depth map."""
    out = depth_color.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for info in detections_info:
        x1, y1, x2, y2 = info["box"]
        dist_m   = info.get("dist_m")
        verdict  = info.get("verdict")
        length_m = info.get("length_m")
        bar_w_m  = info.get("bar_w_m")

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        colour = COLOR_OK if verdict is True else (COLOR_FAIL if verdict is False else COLOR_DETECT)

        # Bounding box viền
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)

        lines = [
            f"D:{fmt_m(dist_m)}",
            f"L:{fmt_m(length_m)}",
            f"W:{fmt_m(bar_w_m)}",
        ]
        for i, ln in enumerate(lines):
            cv2.putText(out, ln, (x1 + 4, y1 + 16 + i * 17), font, 0.45,
                        colour, 1, cv2.LINE_AA)

        cv2.circle(out, (cx, cy), 5, (0, 255, 180), -1)

    return out


def draw_hud_top(
    img: np.ndarray,
    state: str,
    n_det: int,
    fps: float,
    conf: float,
) -> None:
    """HUD ở góc trên-trái: trạng thái, FPS, số detection."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = [
        f"State: {state}  FPS: {fps:.1f}",
        f"Detections: {n_det}  Conf: {conf:.2f}",
        "T=Trigger  Q=Quit  D=Depth  R=Reset  S=Save",
    ]
    for i, ln in enumerate(lines):
        # Shadow
        cv2.putText(img, ln, (14, 26 + i * 24), font, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        # Text
        cv2.putText(img, ln, (14, 26 + i * 24), font, 0.55, (200, 240, 200), 1, cv2.LINE_AA)


# ===========================================================================
# Camera source abstraction (Picamera2 / OpenCV webcam / video file)
# ===========================================================================

class _VideoFileSource:
    """Giả lập live camera từ hai video file (dùng khi debug trên PC)."""

    def __init__(self, left_path: str, right_path: str):
        self._cap_l = cv2.VideoCapture(left_path)
        self._cap_r = cv2.VideoCapture(right_path)
        if not self._cap_l.isOpened():
            raise RuntimeError(f"Cannot open left video: {left_path}")
        if not self._cap_r.isOpened():
            raise RuntimeError(f"Cannot open right video: {right_path}")

    def read(self) -> tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        ok_l, fl = self._cap_l.read()
        ok_r, fr = self._cap_r.read()
        if not ok_l or not ok_r:
            # Loop video
            self._cap_l.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._cap_r.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok_l, fl = self._cap_l.read()
            ok_r, fr = self._cap_r.read()
        return (ok_l and ok_r), fl, fr

    def release(self):
        self._cap_l.release()
        self._cap_r.release()


class _WebcamSource:
    """Hai webcam OpenCV (dùng khi debug trên PC không có Picamera2)."""

    def __init__(self, left_idx: int = 0, right_idx: int = 1):
        self._cap_l = cv2.VideoCapture(left_idx)
        self._cap_r = cv2.VideoCapture(right_idx)
        if not self._cap_l.isOpened():
            raise RuntimeError(f"Cannot open left cam (index {left_idx})")
        if not self._cap_r.isOpened():
            raise RuntimeError(f"Cannot open right cam (index {right_idx})")

    def read(self) -> tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        ok_l, fl = self._cap_l.read()
        ok_r, fr = self._cap_r.read()
        return (ok_l and ok_r), fl, fr

    def release(self):
        self._cap_l.release()
        self._cap_r.release()


class _Picamera2Source:
    """Wrapper quanh StereoCamera (picamera2) từ Python/camera.py."""

    def __init__(self, left_num: int = 0, right_num: int = 1,
                 width: int = 960, height: int = 540):
        # Import lazily để tránh lỗi trên PC không có picamera2
        _python_dir = Path(__file__).resolve().parent
        if str(_python_dir) not in sys.path:
            sys.path.insert(0, str(_python_dir))
        from camera import StereoCamera
        self._cam = StereoCamera(
            left_camera_num=left_num,
            right_camera_num=right_num,
            capture_width=width,
            capture_height=height,
        )
        if not self._cam.open():
            raise RuntimeError("Cannot open Picamera2 stereo cameras.")
        if not self._cam.start():
            self._cam.release()
            raise RuntimeError("Cannot start Picamera2 capture thread.")

    def read(self) -> tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        ok, frame = self._cam.read()
        if not ok or frame is None:
            return False, None, None
        return True, frame.left, frame.right

    def release(self):
        self._cam.release()


# ===========================================================================
# Burst capture + analysis
# ===========================================================================

class BurstResult:
    """Lưu kết quả tổng hợp từ N burst frames."""
    __slots__ = [
        "annotated", "depth_color", "detections_info",
        "best_dist_m", "best_length_m", "best_bar_w_m",
        "verdict", "reason", "n_frames",
    ]

    def __init__(self):
        self.annotated:        Optional[np.ndarray] = None
        self.depth_color:      Optional[np.ndarray] = None
        self.detections_info:  list[dict]           = []
        self.best_dist_m:      Optional[float]      = None
        self.best_length_m:    Optional[float]      = None
        self.best_bar_w_m:     Optional[float]      = None
        self.verdict:          Optional[bool]       = None
        self.reason:           str                  = ""
        self.n_frames:         int                  = 0


def run_burst(
    source,
    depth_est: StereoDepth,
    yolo: YOLO,
    focal_px: float,
    n_burst: int = N_BURST,
    conf: float = 0.35,
) -> BurstResult:
    """
    Thu thập n_burst frame, chạy depth + YOLO trên từng frame,
    tổng hợp kết quả bằng trung vị.
    """
    result = BurstResult()
    result.n_frames = n_burst

    # Mỗi detection (theo class+vị trí tương đối) → list giá trị qua các frames
    # Đơn giản hóa: gom tất cả detections của "class cylinder/perch" qua N frames
    # rồi lấy trung vị.

    all_dist:   list[float] = []
    all_width:  list[float] = []
    all_height: list[float] = []

    last_annotated:   Optional[np.ndarray] = None
    last_depth_color: Optional[np.ndarray] = None
    last_detinfo:     list[dict]           = []

    print(f"[BURST] Capturing {n_burst} frames …")

    for frame_i in range(n_burst):
        ok, fl, fr = source.read()
        if not ok or fl is None or fr is None:
            print(f"  [BURST] Frame {frame_i+1}: read failed, skipping.")
            continue

        # Depth
        left_rect, disparity, depth_map = depth_est.process(fl, fr)

        # YOLO
        results = yolo(left_rect, conf=conf, verbose=False)
        boxes   = results[0].boxes
        masks   = results[0].masks

        frame_detinfo: list[dict] = []

        if boxes is not None and len(boxes) > 0:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                cls_id = int(box.cls[0])
                conf_v = float(box.conf[0])
                name   = yolo.names[cls_id]

                mask_poly: Optional[np.ndarray] = None
                if masks is not None and i < len(masks):
                    xy = masks[i].xy
                    if xy is not None and len(xy) > 0 and len(xy[0]) >= 3:
                        mask_poly = xy[0].astype(np.int32)

                dist_m, width_m, height_m = box_physical_size(
                    depth_map, x1, y1, x2, y2, focal_px, CALIB_UNIT
                )

                if dist_m is not None:
                    all_dist.append(dist_m)
                if width_m is not None:
                    all_width.append(width_m)
                if height_m is not None:
                    all_height.append(height_m)

                frame_detinfo.append({
                    "box":      (x1, y1, x2, y2),
                    "name":     name,
                    "conf":     conf_v,
                    "dist_m":   dist_m,
                    "width_m":  width_m,
                    "height_m": height_m,
                    "mask_poly": mask_poly,
                })

        # Giữ lại frame cuối để hiển thị
        last_annotated   = left_rect
        last_depth_color = depth_to_color(depth_map, 5.0, CALIB_UNIT)
        last_detinfo     = frame_detinfo

        print(f"  [BURST] Frame {frame_i+1}/{n_burst}: {len(frame_detinfo)} detection(s)")

    # -----------------------------------------------------------------------
    # Tổng hợp: lấy trung vị
    # -----------------------------------------------------------------------
    med_dist   = float(np.median(all_dist))   if all_dist   else None
    med_width  = float(np.median(all_width))  if all_width  else None
    med_height = float(np.median(all_height)) if all_height else None

    verdict, reason = assess_perch(med_dist, med_width, med_height)

    med_length = max(med_width, med_height)  if (med_width and med_height) else None
    med_bar_w  = min(med_width, med_height)  if (med_width and med_height) else None

    result.best_dist_m   = med_dist
    result.best_length_m = med_length
    result.best_bar_w_m  = med_bar_w
    result.verdict       = verdict
    result.reason        = reason

    print(f"\n[RESULT] dist={fmt_m(med_dist)}  length={fmt_m(med_length)}  "
          f"width={fmt_m(med_bar_w)}")
    print(f"[RESULT] {reason}\n")

    # -----------------------------------------------------------------------
    # Vẽ annotated frame cuối
    # -----------------------------------------------------------------------
    if last_annotated is not None:
        annotated = last_annotated.copy()

        for info in last_detinfo:
            x1, y1, x2, y2 = info["box"]
            v, r = assess_perch(info["dist_m"], info["width_m"], info["height_m"])
            info["verdict"]  = v
            lv  = max(info["width_m"] or 0, info["height_m"] or 0) or None
            bwv = min(info["width_m"] or 0, info["height_m"] or 0) or None
            info["length_m"] = lv
            info["bar_w_m"]  = bwv

            draw_perch_detection(
                annotated,
                x1, y1, x2, y2,
                f"{info['name']} {info['conf']:.0%}",
                info["dist_m"], info["width_m"], info["height_m"],
                v,
                info.get("mask_poly"),
            )

        # Verdict banner tổng hợp (trung vị)
        draw_verdict_banner(
            annotated,
            verdict, reason,
            med_dist, med_length, med_bar_w,
        )

        result.annotated = annotated

    if last_depth_color is not None:
        result.depth_color    = draw_depth_annotation(last_depth_color, last_detinfo)
        result.detections_info = last_detinfo

    return result


# ===========================================================================
# Main loop
# ===========================================================================

def main() -> None:
    # ------------------------------------------------------------------
    # Mở camera source – Picamera2 trên Raspberry Pi 5
    # (chỉnh LEFT_CAM_NUM / RIGHT_CAM_NUM trong khối CONFIG ở trên)
    # ------------------------------------------------------------------
    print("[INFO] Khởi động Picamera2 stereo cameras …")
    print(f"       CAM{LEFT_CAM_NUM} → trái   |   CAM{RIGHT_CAM_NUM} → phải")
    print(f"       Độ phân giải: {CAPTURE_WIDTH}x{CAPTURE_HEIGHT}")
    source = _Picamera2Source(
        left_num=LEFT_CAM_NUM,
        right_num=RIGHT_CAM_NUM,
        width=CAPTURE_WIDTH,
        height=CAPTURE_HEIGHT,
    )

    # ------------------------------------------------------------------
    # Load calibration + depth estimator
    # ------------------------------------------------------------------
    calib_path = Path(CALIB_FILE)
    if not calib_path.exists():
        sys.exit(f"[ERROR] Calibration file not found: {calib_path}")

    print(f"[INFO] Loading calibration: {calib_path}")
    depth_est = StereoDepth(str(calib_path), downscale=DEPTH_DOWNSCALE)
    focal_px  = focal_length_from_Q(depth_est.Q)
    print(f"[INFO] Focal length Q[2,3] = {focal_px:.2f} px")

    # ------------------------------------------------------------------
    # Load YOLO
    # ------------------------------------------------------------------
    model_path = Path(MODEL_FILE)
    if not model_path.exists():
        sys.exit(f"[ERROR] YOLO model not found: {model_path}")

    print(f"[INFO] Loading YOLO model: {model_path}")
    yolo = YOLO(str(model_path))
    print("[INFO] YOLO ready.\n")

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    show_depth   = SHOW_DEPTH_ON_START
    burst_result: Optional[BurstResult] = None
    state        = "STANDBY"   # STANDBY | CAPTURING | RESULT
    fps_display  = 0.0
    t_last       = time.monotonic()
    snapshot_dir = Path(SNAPSHOT_DIR)
    n_det_live   = 0
    conf_thresh  = YOLO_CONF

    WIN_MAIN  = "Perch Detector  [T=Trigger  Q=Quit  D=Depth  R=Reset  S=Save]"
    WIN_DEPTH = "Depth Map"

    cv2.namedWindow(WIN_MAIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_MAIN, WINDOW_WIDTH, WINDOW_HEIGHT)

    print("=" * 62)
    print("  DRONE PERCH FEASIBILITY DETECTOR")
    print("=" * 62)
    print(f"  Ngưỡng đánh giá:")
    print(f"    Khoảng cách : [{MIN_DIST_M*100:.0f}, {MAX_DIST_M*100:.0f}] cm")
    print(f"    Chiều dài   : [{MIN_LENGTH_M*100:.0f}, {MAX_LENGTH_M*100:.0f}] cm")
    print(f"    Chiều rộng  : [{MIN_WIDTH_M*100:.0f}, {MAX_WIDTH_M*100:.0f}] cm")
    print(f"  Burst size    : {N_BURST} frames")
    print(f"  YOLO conf     : {YOLO_CONF}")
    print("=" * 62)
    print("  T = Trigger  |  Q = Thoát  |  D = Depth  |  R = Reset  |  S = Save")
    print("=" * 62 + "\n")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    while True:
        t_now       = time.monotonic()
        fps_display = 1.0 / max(t_now - t_last, 1e-6)
        t_last      = t_now

        ok, fl, fr = source.read()
        if not ok or fl is None or fr is None:
            time.sleep(0.01)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            continue

        # ----------------------------------------------------------------
        # Xử lý frame live (depth + YOLO để preview)
        # ----------------------------------------------------------------
        left_rect, _, depth_map = depth_est.process(fl, fr)
        results_live = yolo(left_rect, conf=conf_thresh, verbose=False)
        boxes_live   = results_live[0].boxes
        n_det_live   = len(boxes_live) if boxes_live is not None else 0

        depth_color_live = depth_to_color(depth_map, MAX_DEPTH_M, CALIB_UNIT)

        # Tạo frame hiển thị
        if burst_result is not None and burst_result.annotated is not None:
            # Hiển thị kết quả burst (đóng băng)
            display       = burst_result.annotated.copy()
            depth_display = (burst_result.depth_color
                             if burst_result.depth_color is not None
                             else depth_color_live)
        else:
            # Live preview
            display = left_rect.copy()

            if boxes_live is not None:
                masks_live = results_live[0].masks
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

                    dist_m, width_m, height_m = box_physical_size(
                        depth_map, x1, y1, x2, y2, focal_px, CALIB_UNIT
                    )
                    v, _ = assess_perch(dist_m, width_m, height_m)

                    draw_perch_detection(
                        display, x1, y1, x2, y2,
                        f"{name} {conf_v:.0%}",
                        dist_m, width_m, height_m, v, mask_poly,
                    )

            depth_display = depth_color_live

        # HUD góc trên-trái
        draw_hud_top(display, state, n_det_live, fps_display, conf_thresh)

        # Hiển thị
        cv2.imshow(WIN_MAIN, display)
        if show_depth:
            cv2.imshow(WIN_DEPTH, depth_display)
        else:
            cv2.destroyWindow(WIN_DEPTH)

        # ----------------------------------------------------------------
        # Xử lý phím bấm
        # ----------------------------------------------------------------
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):       # Q / ESC → thoát
            break

        elif key == ord("t"):           # T → Trigger burst
            state = "CAPTURING"
            draw_hud_top(display, state, n_det_live, fps_display, conf_thresh)
            cv2.imshow(WIN_MAIN, display)
            cv2.waitKey(1)

            burst_result = run_burst(
                source, depth_est, yolo, focal_px,
                n_burst=N_BURST,
                conf=conf_thresh,
            )
            state = "RESULT"

        elif key == ord("d"):           # D → toggle depth map
            show_depth = not show_depth
            print(f"[INFO] Depth window: {'ON' if show_depth else 'OFF'}")

        elif key == ord("r"):           # R → reset
            burst_result = None
            state = "STANDBY"
            print("[INFO] Reset – quay về live preview.")

        elif key == ord("s"):           # S → save snapshot
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            tag = time.strftime("%Y%m%d_%H%M%S")
            if burst_result is not None and burst_result.annotated is not None:
                cv2.imwrite(
                    str(snapshot_dir / f"{tag}_annotated.png"),
                    burst_result.annotated,
                )
                cv2.imwrite(
                    str(snapshot_dir / f"{tag}_depth.png"),
                    burst_result.depth_color if burst_result.depth_color is not None
                    else depth_display,
                )
            else:
                cv2.imwrite(str(snapshot_dir / f"{tag}_live.png"),       display)
                cv2.imwrite(str(snapshot_dir / f"{tag}_depth_live.png"), depth_display)
            print(f"[INFO] Snapshot saved → {snapshot_dir}/{tag}_*.png")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    source.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
