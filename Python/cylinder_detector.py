"""
cylinder_detector.py
====================

Phát hiện vật thể hình trụ nằm ngang (thanh sắt, thanh nhựa, cẳng tay, ...)
từ ảnh độ sâu stereo — KHÔNG dùng model object detection.

Thuật toán thuần OpenCV / NumPy:
  1. Tính depth map từ stereo camera.
  2. Phát hiện cạnh trong ảnh trái đã rectify (Canny).
  3. Phát hiện đường thẳng bằng HoughLinesP → chọn các đoạn gần ngang
     (góc lệch < MAX_TILT_DEG so với phương ngang).
  4. Gom nhóm (cluster) các đường thẳng gần nhau → mỗi cluster là 1 ứng viên
     hình trụ (cylinder segment).
  5. Với mỗi ứng viên, xây dựng ROI dày (dày theo chiều dọc) để đo khoảng cách
     và chiều rộng vật thể từ depth map bằng phép chiếu ngược.
  6. Khi bấm T: chụp N_BURST frame, tổng hợp kết quả bằng trung vị.

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
# sys.path – thêm thư mục Python/ vào path để import camera / stereo_depth
# ---------------------------------------------------------------------------
_PYTHON_DIR = Path(__file__).resolve().parent
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from camera import StereoCamera
from stereo_depth import StereoDepth


# ===========================================================================
#  C O N F I G  – chỉnh tất cả tham số TẠI ĐÂY
# ===========================================================================

# ── Đường dẫn ──────────────────────────────────────────────────────────────
CALIB_FILE: str  = str(_PYTHON_DIR.parent / "stereo_calibration.npz")
SNAPSHOT_DIR: str = "cylinder_snapshots"

# ── Camera ─────────────────────────────────────────────────────────────────
LEFT_CAM_NUM:   int = 0
RIGHT_CAM_NUM:  int = 1
CAPTURE_WIDTH:  int = 960
CAPTURE_HEIGHT: int = 540
FRAMERATE:      int = 30

# ── Stereo depth ────────────────────────────────────────────────────────────
DEPTH_DOWNSCALE: float = 0.5
CALIB_UNIT:      str   = "mm"   # "mm" | "cm" | "m"
MAX_DEPTH_M:     float = 5.0

# ── Burst ───────────────────────────────────────────────────────────────────
N_BURST: int = 7   # số frame chụp mỗi lần bấm T

# ── Phát hiện đường thẳng (Hough) ──────────────────────────────────────────
CANNY_LOW:    int   = 40
CANNY_HIGH:   int   = 120
HOUGH_THRESH: int   = 40        # số điểm tối thiểu trên đường Hough
HOUGH_MIN_LEN: int  = 50        # chiều dài tối thiểu của đoạn thẳng (px)
HOUGH_GAP:    int   = 15        # gap tối đa được phép trong đoạn thẳng (px)
MAX_TILT_DEG: float = 20.0      # góc lệch tối đa so với phương ngang (độ)

# ── Gom nhóm đường thẳng ────────────────────────────────────────────────────
CLUSTER_Y_TOL:    int = 25      # hai đường cách nhau ≤ px theo Y → cùng nhóm
CLUSTER_ANGLE_TOL: float = 10.0 # góc lệch tối đa giữa hai đường cùng nhóm (độ)

# ── ROI depth (dày bao nhiêu px theo chiều dọc để đo depth) ─────────────────
ROI_HALF_H: int = 12   # lấy ROI dày ±ROI_HALF_H px quanh trục trung tâm

# ── Ngưỡng kích thước vật thể (mét) ─────────────────────────────────────────
MIN_LENGTH_M: float = 0.05    # chiều dài tối thiểu của cylinder (m)
MAX_LENGTH_M: float = 3.00
MIN_DIAM_M:   float = 0.01    # đường kính / chiều cao bounding tối thiểu (m)
MAX_DIAM_M:   float = 0.30
MIN_DIST_M:   float = 0.15    # khoảng cách tối thiểu đến vật (m)
MAX_DIST_M:   float = 5.00

# ── Hiển thị ────────────────────────────────────────────────────────────────
SHOW_DEPTH_ON_START: bool = True
WINDOW_WIDTH:        int  = 960
WINDOW_HEIGHT:       int  = 540

# ── Màu sắc ─────────────────────────────────────────────────────────────────
COLOR_OK      = (0, 220, 60)    # xanh lá  → đạt ngưỡng kích thước / khoảng cách
COLOR_FAIL    = (0, 60, 240)    # đỏ       → không đạt
COLOR_DETECT  = (255, 180, 0)   # vàng     → phát hiện, chưa đủ depth


# ===========================================================================
#  Tiện ích
# ===========================================================================

def depth_unit_scale(calib_unit: str) -> float:
    """Trả về hệ số chuyển đổi sang mét."""
    if calib_unit == "mm":
        return 1.0 / 1000.0
    if calib_unit == "cm":
        return 1.0 / 100.0
    return 1.0


def disparity_to_color(disparity: np.ndarray) -> np.ndarray:
    """Chuyển disparity thành ảnh màu TURBO để hiển thị."""
    valid = disparity > 0
    display = np.zeros(disparity.shape, dtype=np.uint8)
    if np.any(valid):
        vd = disparity[valid]
        lo, hi = np.percentile(vd, 2), np.percentile(vd, 98)
        if hi > lo:
            norm = np.clip((disparity - lo) / (hi - lo), 0, 1)
            display = (norm * 255).astype(np.uint8)
    display[~valid] = 0
    return cv2.applyColorMap(display, cv2.COLORMAP_TURBO)


def depth_to_color(
    depth: np.ndarray,
    max_m: float = MAX_DEPTH_M,
    calib_unit: str = CALIB_UNIT,
) -> np.ndarray:
    """Chuyển depth map thành ảnh màu TURBO."""
    scale = depth_unit_scale(calib_unit)
    depth_m = depth * scale
    valid = (depth > 0) & np.isfinite(depth)
    display = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        clipped = np.clip(depth_m, 0, max_m)
        display = ((1.0 - clipped / max_m) * 255).astype(np.uint8)
    display[~valid] = 0
    return cv2.applyColorMap(display, cv2.COLORMAP_TURBO)


def fmt_m(val: Optional[float]) -> str:
    if val is None:
        return "N/A"
    if val < 1.0:
        return f"{val * 100:.1f} cm"
    return f"{val:.3f} m"


def focal_length_from_Q(Q: np.ndarray) -> float:
    """Focal length (px) từ ma trận Q của stereoRectify."""
    return float(Q[2, 3])


# ===========================================================================
#  Phát hiện cylinder từ depth + ảnh màu
# ===========================================================================

def detect_cylinders(
    color_img: np.ndarray,
    depth_map: np.ndarray,
    focal_px: float,
    calib_unit: str = CALIB_UNIT,
) -> list[dict]:
    """
    Phát hiện các cylinder segment nằm ngang trong ảnh.

    Trả về danh sách dict, mỗi phần tử gồm:
        "cx_px", "cy_px"    – tâm segment (pixel)
        "x1_px", "x2_px"    – span ngang (pixel)
        "angle_deg"         – góc nghiêng (độ, <0 = nghiêng trái)
        "dist_m"            – khoảng cách (m), None nếu không đủ depth
        "length_m"          – chiều dài ước tính (m)
        "diam_m"            – chiều rộng/đường kính ước tính (m)
        "verdict"           – True / False / None
        "line_pts"          – [(x1,y1),(x2,y2)] trục trung tâm đường thẳng
        "roi_box"           – (rx1,ry1,rx2,ry2) vùng ROI depth đã dùng
    """
    h, w = color_img.shape[:2]
    scale = depth_unit_scale(calib_unit)

    # 1. Canny trên ảnh xám
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)

    # 2. HoughLinesP
    lines_raw = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESH,
        minLineLength=HOUGH_MIN_LEN,
        maxLineGap=HOUGH_GAP,
    )

    if lines_raw is None:
        return []

    # 3. Lọc đường gần ngang (|angle| <= MAX_TILT_DEG)
    horizontal: list[dict] = []
    for seg in lines_raw:
        x1, y1, x2, y2 = seg[0]
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            continue
        angle = float(np.degrees(np.arctan2(dy, dx)))  # -180..180
        # Chuẩn hoá về -90..90
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180
        if abs(angle) > MAX_TILT_DEG:
            continue
        length_px = float(np.hypot(dx, dy))
        cy_line = (y1 + y2) / 2.0
        cx_line = (x1 + x2) / 2.0
        horizontal.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "angle": angle,
            "length_px": length_px,
            "cy": cy_line,
            "cx": cx_line,
        })

    if not horizontal:
        return []

    # 4. Gom nhóm (cluster) theo cy và angle
    used = [False] * len(horizontal)
    clusters: list[list[dict]] = []

    for i, li in enumerate(horizontal):
        if used[i]:
            continue
        cluster = [li]
        used[i] = True
        for j in range(i + 1, len(horizontal)):
            if used[j]:
                continue
            lj = horizontal[j]
            if (abs(li["cy"] - lj["cy"]) <= CLUSTER_Y_TOL and
                    abs(li["angle"] - lj["angle"]) <= CLUSTER_ANGLE_TOL):
                cluster.append(lj)
                used[j] = True
        clusters.append(cluster)

    # 5. Tổng hợp mỗi cluster → 1 ứng viên
    results: list[dict] = []

    for cluster in clusters:
        # Tìm span ngang rộng nhất trong cluster
        all_x = [seg["x1"] for seg in cluster] + [seg["x2"] for seg in cluster]
        all_y = [seg["y1"] for seg in cluster] + [seg["y2"] for seg in cluster]
        x_left  = int(min(all_x))
        x_right = int(max(all_x))
        cy_mean = float(np.mean([seg["cy"] for seg in cluster]))
        angle_mean = float(np.mean([seg["angle"] for seg in cluster]))

        # Góc cuối dùng để tính điểm endpoints sau khi merge
        cy_int = int(round(cy_mean))

        # 6. ROI depth: dải ngang từ x_left → x_right, dày ±ROI_HALF_H
        ry1 = max(0, cy_int - ROI_HALF_H)
        ry2 = min(h - 1, cy_int + ROI_HALF_H)
        rx1, rx2 = max(0, x_left), min(w - 1, x_right)

        roi_depth = depth_map[ry1:ry2 + 1, rx1:rx2 + 1]
        valid_mask = np.isfinite(roi_depth) & (roi_depth > 0)
        valid_depths = roi_depth[valid_mask]

        dist_m: Optional[float] = None
        length_m: Optional[float] = None
        diam_m: Optional[float] = None

        if valid_depths.size >= 5:
            # Lấy trung vị của IQR để bền vững với noise
            lo25 = np.percentile(valid_depths, 25)
            hi75 = np.percentile(valid_depths, 75)
            core = valid_depths[(valid_depths >= lo25) & (valid_depths <= hi75)]
            depth_raw = float(np.median(core if core.size > 0 else valid_depths))
            dist_m = depth_raw * scale

            if focal_px > 0 and dist_m > 0:
                span_px  = x_right - x_left
                length_m = (span_px * dist_m) / focal_px

                # Chiều cao bounding box (theo phương đứng) ≈ đường kính
                height_px = ry2 - ry1
                diam_m = (height_px * dist_m) / focal_px

        # Đánh giá
        verdict = _assess(dist_m, length_m, diam_m)

        results.append({
            "cx_px":    (x_left + x_right) // 2,
            "cy_px":    cy_int,
            "x1_px":    x_left,
            "x2_px":    x_right,
            "angle_deg": angle_mean,
            "dist_m":   dist_m,
            "length_m": length_m,
            "diam_m":   diam_m,
            "verdict":  verdict,
            "line_pts": ((x_left, cy_int), (x_right, cy_int)),
            "roi_box":  (rx1, ry1, rx2, ry2),
        })

    # Sắp xếp theo khoảng cách gần nhất trước
    results.sort(key=lambda d: d["dist_m"] if d["dist_m"] is not None else 1e9)
    return results


def _assess(
    dist_m: Optional[float],
    length_m: Optional[float],
    diam_m: Optional[float],
) -> Optional[bool]:
    """True = đạt, False = không đạt, None = thiếu data."""
    if dist_m is None or length_m is None or diam_m is None:
        return None
    ok = (
        MIN_DIST_M   <= dist_m   <= MAX_DIST_M   and
        MIN_LENGTH_M <= length_m <= MAX_LENGTH_M and
        MIN_DIAM_M   <= diam_m   <= MAX_DIAM_M
    )
    return ok


# ===========================================================================
#  Burst capture + tổng hợp
# ===========================================================================

class BurstResult:
    """Kết quả sau N_BURST frame."""
    __slots__ = [
        "annotated", "depth_color",
        "cylinders",
        "med_dist_m", "med_length_m", "med_diam_m",
        "verdict", "n_frames",
    ]

    def __init__(self) -> None:
        self.annotated:   Optional[np.ndarray] = None
        self.depth_color: Optional[np.ndarray] = None
        self.cylinders:   list[dict]            = []
        self.med_dist_m:   Optional[float]      = None
        self.med_length_m: Optional[float]      = None
        self.med_diam_m:   Optional[float]      = None
        self.verdict:      Optional[bool]       = None
        self.n_frames:     int                  = 0


def run_burst(
    camera: StereoCamera,
    depth_est: StereoDepth,
    focal_px: float,
    n_burst: int = N_BURST,
) -> BurstResult:
    """
    Chụp n_burst frame, chạy detect_cylinders trên từng frame,
    tổng hợp kết quả bằng trung vị.
    """
    result = BurstResult()
    result.n_frames = n_burst

    all_dist:   list[float] = []
    all_length: list[float] = []
    all_diam:   list[float] = []

    last_left_rect:   Optional[np.ndarray] = None
    last_depth_map:   Optional[np.ndarray] = None
    last_cylinders:   list[dict]           = []

    last_sequence: Optional[int] = None

    print(f"[BURST] Đang chụp {n_burst} frame …")

    for frame_i in range(n_burst):
        # Chờ frame mới
        for _ in range(30):
            ok, stereo_frame = camera.read(
                last_sequence=last_sequence, copy_frames=True
            )
            if ok and stereo_frame is not None:
                break
            time.sleep(0.01)
        else:
            print(f"  [BURST] Frame {frame_i + 1}: không lấy được frame, bỏ qua.")
            continue

        last_sequence = stereo_frame.sequence
        left_rect, _, depth_map = depth_est.process(
            stereo_frame.left, stereo_frame.right
        )

        cylinders = detect_cylinders(left_rect, depth_map, focal_px, CALIB_UNIT)

        # Thu thập giá trị đo được từ cylinder đầu tiên (gần nhất, đã sort)
        for cyl in cylinders:
            if cyl["dist_m"] is not None:
                all_dist.append(cyl["dist_m"])
            if cyl["length_m"] is not None:
                all_length.append(cyl["length_m"])
            if cyl["diam_m"] is not None:
                all_diam.append(cyl["diam_m"])

        last_left_rect   = left_rect
        last_depth_map   = depth_map
        last_cylinders   = cylinders

        print(f"  [BURST] Frame {frame_i + 1}/{n_burst}: "
              f"{len(cylinders)} cylinder(s) phát hiện")

    # ── Tổng hợp bằng trung vị ───────────────────────────────────────────
    med_dist   = float(np.median(all_dist))   if all_dist   else None
    med_length = float(np.median(all_length)) if all_length else None
    med_diam   = float(np.median(all_diam))   if all_diam   else None

    result.med_dist_m   = med_dist
    result.med_length_m = med_length
    result.med_diam_m   = med_diam
    result.verdict      = _assess(med_dist, med_length, med_diam)

    print(f"\n[RESULT] dist={fmt_m(med_dist)}  "
          f"length={fmt_m(med_length)}  diam={fmt_m(med_diam)}")
    ok_str = {True: "✓ ĐẠT", False: "✗ KHÔNG ĐẠT", None: "? THIẾU DỮ LIỆU"}
    print(f"[RESULT] {ok_str[result.verdict]}\n")

    # ── Vẽ annotated frame cuối ──────────────────────────────────────────
    if last_left_rect is not None:
        annotated = last_left_rect.copy()
        for cyl in last_cylinders:
            _draw_cylinder(annotated, cyl)

        _draw_verdict_banner(
            annotated,
            result.verdict,
            med_dist, med_length, med_diam,
            n_frames=n_burst,
        )
        result.annotated = annotated

    if last_depth_map is not None:
        dc = depth_to_color(last_depth_map, MAX_DEPTH_M, CALIB_UNIT)
        _draw_depth_annotations(dc, last_cylinders)
        result.depth_color = dc

    result.cylinders = last_cylinders
    return result


# ===========================================================================
#  Vẽ overlay
# ===========================================================================

def _draw_cylinder(img: np.ndarray, cyl: dict) -> None:
    """Vẽ 1 cylinder detection lên ảnh."""
    verdict = cyl["verdict"]
    if verdict is True:
        colour = COLOR_OK
    elif verdict is False:
        colour = COLOR_FAIL
    else:
        colour = COLOR_DETECT

    rx1, ry1, rx2, ry2 = cyl["roi_box"]
    cx, cy = cyl["cx_px"], cyl["cy_px"]

    # Vùng ROI mờ
    overlay = img.copy()
    cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), colour, -1)
    cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)

    # Viền ROI
    cv2.rectangle(img, (rx1, ry1), (rx2, ry2), colour, 2)

    # Đường trục trung tâm
    (lx1, ly1), (lx2, ly2) = cyl["line_pts"]
    cv2.line(img, (lx1, ly1), (lx2, ly2), colour, 2, cv2.LINE_AA)

    # Crosshair tại tâm
    arm = 12
    cv2.line(img, (cx - arm, cy), (cx + arm, cy), (0, 255, 180), 2)
    cv2.line(img, (cx, cy - arm), (cx, cy + arm), (0, 255, 180), 2)
    cv2.circle(img, (cx, cy), 4, (0, 255, 180), -1)

    # Nhãn thông tin
    lines_text = [
        f"Cylinder  {cyl['angle_deg']:+.1f}deg",
        f"Dist  : {fmt_m(cyl['dist_m'])}",
        f"Length: {fmt_m(cyl['length_m'])}",
        f"Diam  : {fmt_m(cyl['diam_m'])}",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    fscale, thick, pad, lh = 0.50, 1, 4, 18

    tw = max(cv2.getTextSize(ln, font, fscale, thick)[0][0] for ln in lines_text)
    th_total = lh * len(lines_text) + pad

    label_y0 = ry1 - th_total - pad
    if label_y0 < 0:
        label_y0 = ry2 + pad

    cv2.rectangle(
        img,
        (rx1, label_y0),
        (rx1 + tw + pad * 2, label_y0 + th_total),
        tuple(max(0, c - 50) for c in colour),
        -1,
    )
    for i, ln in enumerate(lines_text):
        ty = label_y0 + pad + lh * (i + 1) - 2
        cv2.putText(img, ln, (rx1 + pad, ty), font, fscale,
                    (255, 255, 255), thick, cv2.LINE_AA)


def _draw_depth_annotations(
    depth_color: np.ndarray,
    cylinders: list[dict],
) -> None:
    """Vẽ nhãn khoảng cách ngắn gọn lên ảnh depth colormap."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    for cyl in cylinders:
        rx1, ry1, rx2, ry2 = cyl["roi_box"]
        verdict = cyl["verdict"]
        colour = (
            COLOR_OK if verdict is True
            else COLOR_FAIL if verdict is False
            else COLOR_DETECT
        )
        cv2.rectangle(depth_color, (rx1, ry1), (rx2, ry2), colour, 2)
        info_lines = [
            f"D:{fmt_m(cyl['dist_m'])}",
            f"L:{fmt_m(cyl['length_m'])}",
            f"d:{fmt_m(cyl['diam_m'])}",
        ]
        for i, ln in enumerate(info_lines):
            cv2.putText(depth_color, ln, (rx1 + 3, ry1 + 15 + i * 16),
                        font, 0.42, colour, 1, cv2.LINE_AA)


def _draw_verdict_banner(
    img: np.ndarray,
    verdict: Optional[bool],
    dist_m: Optional[float],
    length_m: Optional[float],
    diam_m: Optional[float],
    n_frames: int = N_BURST,
) -> None:
    """Banner tóm tắt kết quả burst ở phía dưới ảnh."""
    h, w = img.shape[:2]
    banner_h = 90
    y0 = h - banner_h

    overlay = img.copy()
    bg = (10, 80, 10) if verdict is True else (60, 10, 10) if verdict is False else (30, 30, 60)
    cv2.rectangle(overlay, (0, y0), (w, h), bg, -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

    font = cv2.FONT_HERSHEY_SIMPLEX
    verdict_str = {True: "✓ CYLINDER – ĐẠT", False: "✗ CYLINDER – KHÔNG ĐẠT", None: "? THIẾU DỮ LIỆU"}
    v_colour = COLOR_OK if verdict is True else COLOR_FAIL if verdict is False else (200, 200, 200)
    cv2.putText(img, verdict_str[verdict], (12, y0 + 28), font, 0.72, v_colour, 2, cv2.LINE_AA)

    stats = (
        f"Dist={fmt_m(dist_m)}  "
        f"Length={fmt_m(length_m)}  "
        f"Diam={fmt_m(diam_m)}  "
        f"[{n_frames} frames median]"
    )
    cv2.putText(img, stats, (12, y0 + 55), font, 0.55, (210, 220, 255), 1, cv2.LINE_AA)

    thr = (
        f"Thr — Dist:[{MIN_DIST_M*100:.0f},{MAX_DIST_M*100:.0f}]cm  "
        f"Len:[{MIN_LENGTH_M*100:.0f},{MAX_LENGTH_M*100:.0f}]cm  "
        f"Diam:[{MIN_DIAM_M*100:.0f},{MAX_DIAM_M*100:.0f}]cm  "
        f"MaxTilt:{MAX_TILT_DEG:.0f}deg"
    )
    cv2.putText(img, thr, (12, y0 + 78), font, 0.40, (150, 160, 180), 1, cv2.LINE_AA)


def draw_hud(
    img: np.ndarray,
    state: str,
    n_det: int,
    fps: float,
) -> None:
    """HUD góc trên-trái."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = [
        f"State: {state}   FPS: {fps:.1f}",
        f"Cylinders: {n_det}",
        "T=Trigger  Q=Quit  D=Depth  R=Reset  S=Save",
    ]
    for i, ln in enumerate(lines):
        cv2.putText(img, ln, (14, 26 + i * 24), font, 0.54, (0, 0, 0),     3, cv2.LINE_AA)
        cv2.putText(img, ln, (14, 26 + i * 24), font, 0.54, (200, 240, 200), 1, cv2.LINE_AA)


# ===========================================================================
#  Main loop
# ===========================================================================

def main() -> None:
    # ── Kiểm tra calibration ────────────────────────────────────────────
    calib_path = Path(CALIB_FILE)
    if not calib_path.exists():
        sys.exit(f"[ERROR] Không tìm thấy file calibration: {calib_path}")

    # ── Mở camera ───────────────────────────────────────────────────────
    print("[INFO] Khởi động Picamera2 stereo cameras …")
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

    # ── Load depth estimator ────────────────────────────────────────────
    print(f"[INFO] Load calibration: {calib_path}")
    depth_est = StereoDepth(str(calib_path), downscale=DEPTH_DOWNSCALE)
    focal_px  = focal_length_from_Q(depth_est.Q)
    print(f"[INFO] Focal length Q[2,3] = {focal_px:.2f} px")

    # ── Khởi tạo state ──────────────────────────────────────────────────
    WIN_MAIN  = "Cylinder Detector  [T=Trigger  Q=Quit  D=Depth  R=Reset  S=Save]"
    WIN_DEPTH = "Depth Map"

    cv2.namedWindow(WIN_MAIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_MAIN, WINDOW_WIDTH, WINDOW_HEIGHT)

    show_depth    = SHOW_DEPTH_ON_START
    burst_result: Optional[BurstResult] = None
    state         = "STANDBY"
    fps_display   = 0.0
    t_last        = time.monotonic()
    last_sequence: Optional[int] = None
    n_det_live    = 0
    snapshot_dir  = Path(SNAPSHOT_DIR)

    print("=" * 60)
    print("  CYLINDER SEGMENT DETECTOR (không dùng YOLO)")
    print("=" * 60)
    print(f"  Burst       : {N_BURST} frames  |  Tổng hợp: trung vị")
    print(f"  Góc lệch tối đa : {MAX_TILT_DEG} độ")
    print(f"  Khoảng cách : [{MIN_DIST_M*100:.0f}, {MAX_DIST_M*100:.0f}] cm")
    print(f"  Chiều dài   : [{MIN_LENGTH_M*100:.0f}, {MAX_LENGTH_M*100:.0f}] cm")
    print(f"  Đường kính  : [{MIN_DIAM_M*100:.0f}, {MAX_DIAM_M*100:.0f}] cm")
    print("=" * 60)
    print("  T = Trigger  |  Q = Thoát  |  D = Depth  |  R = Reset  |  S = Save")
    print("=" * 60 + "\n")

    try:
        while True:
            t_now = time.monotonic()
            fps_display = 1.0 / max(t_now - t_last, 1e-6)
            t_last = t_now

            # ── Đọc frame mới ──────────────────────────────────────────
            ok, stereo_frame = camera.read(
                last_sequence=last_sequence, copy_frames=False
            )

            if not ok or stereo_frame is None:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("t") or key == ord("T"):
                    # Burst
                    state = "CAPTURING"
                    cv2.waitKey(1)
                    burst_result = run_burst(camera, depth_est, focal_px, N_BURST)
                    state = "RESULT"
                continue

            last_sequence = stereo_frame.sequence
            left_rect, _, depth_map = depth_est.process(
                stereo_frame.left, stereo_frame.right
            )
            depth_color_live = depth_to_color(depth_map, MAX_DEPTH_M, CALIB_UNIT)

            # ── Detect live ────────────────────────────────────────────
            cylinders_live = detect_cylinders(left_rect, depth_map, focal_px, CALIB_UNIT)
            n_det_live = len(cylinders_live)

            # ── Chọn frame hiển thị ────────────────────────────────────
            if burst_result is not None and burst_result.annotated is not None:
                display       = burst_result.annotated.copy()
                depth_display = (
                    burst_result.depth_color
                    if burst_result.depth_color is not None
                    else depth_color_live
                )
            else:
                display = left_rect.copy()
                for cyl in cylinders_live:
                    _draw_cylinder(display, cyl)
                depth_display = depth_color_live.copy()
                _draw_depth_annotations(depth_display, cylinders_live)

            draw_hud(display, state, n_det_live, fps_display)

            cv2.imshow(WIN_MAIN, display)
            if show_depth:
                cv2.imshow(WIN_DEPTH, depth_display)
            else:
                cv2.destroyWindow(WIN_DEPTH)

            # ── Phím bấm ──────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            elif key == ord("t") or key == ord("T"):
                state = "CAPTURING"
                cv2.waitKey(1)
                burst_result = run_burst(camera, depth_est, focal_px, N_BURST)
                state = "RESULT"

            elif key == ord("d") or key == ord("D"):
                show_depth = not show_depth

            elif key == ord("r") or key == ord("R"):
                burst_result = None
                state = "STANDBY"

            elif key == ord("s") or key == ord("S"):
                snapshot_dir.mkdir(parents=True, exist_ok=True)
                ts = time.strftime("%Y%m%d_%H%M%S")
                fname = snapshot_dir / f"cylinder_{ts}.jpg"
                cv2.imwrite(str(fname), display)
                print(f"[SAVE] Đã lưu: {fname}")

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("[INFO] Đã thoát.")


if __name__ == "__main__":
    main()
