"""
perch_detector.py
=================

Drone Perch Landing Feasibility Detector – Raspberry Pi 5 / Picamera2
-----------------------------------------------------------------------
Chạy thẳng bằng:
    python3 perch_detector.py

Toàn bộ tham số được cài sẵn trong khối CONFIG bên dưới.
Không cần truyền tham số từ terminal.

Luồng hoạt động:
    1. SEARCHING (realtime) – chỉ rectify ảnh trái + YOLO tìm thanh đáp.
       KHÔNG tính depth để giữ FPS cao. Hiển thị 2 trục dọc/ngang ở tâm
       ảnh để căn, góc nghiêng của thanh trong ảnh và độ lệch tâm.
    2. Bấm T – burst N frame, tính depth (stereo SGBM+WLS), kích thước
       thanh, khoảng cách, và GÓC YAW drone đang lệch so với thanh
       (dựa trên 3D) → thông báo cần xoay drone TRÁI/PHẢI bao nhiêu độ
       và đánh giá có đáp được hay không. Màn hình đóng băng kết quả.
    3. Bấm K – tiếp tục quá trình tìm (quay lại SEARCHING).

Controls:
    T        – Phân tích (burst capture + depth + góc yaw + verdict)
    K        – Tiếp tục tìm (quay lại realtime)
    R        – (giống K) Reset về realtime
    D        – Bật / Tắt cửa sổ depth map (chỉ có sau khi bấm T)
    S        – Lưu snapshot màn hình hiện tại
    Q / ESC  – Thoát
"""

from __future__ import annotations

import math
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
MODEL_FILE: str = str(_CALIB_DIR / "best_obb.pt")              # model YOLOv8n-OBB (train bằng train_cylinder_seg_colab.ipynb)
SNAPSHOT_DIR: str = "perch_snapshots"                           # thư mục lưu ảnh

# ---------- Camera (Picamera2 trên Raspberry Pi 5) -----------------------
LEFT_CAM_NUM:    int = 0     # cổng CAM0 → camera trái
RIGHT_CAM_NUM:   int = 1     # cổng CAM1 → camera phải
CAPTURE_WIDTH:   int = 960   # độ phân giải chụp (px)
CAPTURE_HEIGHT:  int = 540

# Hướng lắp camera trên drone:
#   "forward" – camera nhìn thẳng về phía trước (thanh nằm trước mặt drone).
#               Góc yaw được tính từ depth 3D khi bấm T. Góc nghiêng của
#               thanh trong ảnh (tilt) được hiển thị realtime.
#   "down"    – camera nhìn thẳng xuống dưới (đỉnh ảnh = mũi drone).
#               Góc của thanh trong ảnh chính là góc yaw → có realtime.
CAMERA_MOUNT: str = "forward"

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

# ---------- Căn chỉnh drone (alignment) ---------------------------------
YAW_TOL_DEG:       float = 3.0   # |yaw| <= ngưỡng này → coi như song song với thanh
ANGLE_TOL_DEG:     float = 3.0   # |góc nghiêng thanh trong ảnh| <= ngưỡng → OK
CENTER_TOL_PX:     int   = 25    # độ lệch tâm (px) cho phép so với 2 trục căn
MAX_BAR_ANGLE_DEG: float = 25.0  # Góc lệch tối đa so với phương ngang (độ) để nhận diện/vẽ bbox

# ---------- Hiển thị -----------------------------------------------------
SHOW_DEPTH_ON_START: bool = True   # True = mở cửa sổ depth map sau khi bấm T
WINDOW_WIDTH:        int  = 960
WINDOW_HEIGHT:       int  = 540


# ===========================================================================
# Màu sắc
# ===========================================================================

COLOR_OK     = (0, 220, 60)     # xanh lá → đáp được / đã căn
COLOR_FAIL   = (0, 60, 240)     # đỏ      → không đáp được
COLOR_DETECT = (255, 180, 0)    # vàng    → phát hiện nhưng chưa có depth
COLOR_WARN   = (0, 190, 255)    # cam     → cần điều chỉnh
COLOR_AXIS   = (255, 0, 255)    # tím     → trục thanh
COLOR_GUIDE  = (225, 225, 225)  # trắng   → trục căn dọc/ngang
COLOR_MUTED  = (150, 160, 180)  # xám
COLOR_HUD_BG = (15, 15, 30)     # nền HUD tối


# ===========================================================================
# Hàm tiện ích
# ===========================================================================

def focal_length_from_Q(Q: np.ndarray) -> float:
    """Lấy focal length (px) từ ma trận Q của stereoRectify."""
    return float(Q[2, 3])


def principal_point_from_Q(Q: np.ndarray) -> tuple[float, float]:
    """Lấy principal point (cx, cy) của ảnh rectified trái từ ma trận Q."""
    return -float(Q[0, 3]), -float(Q[1, 3])


def to_metres(val: np.ndarray | float, calib_unit: str):
    """Đổi giá trị depth từ đơn vị calibration sang mét."""
    if calib_unit == "mm":
        return val / 1000.0
    if calib_unit == "cm":
        return val / 100.0
    return val


def rectify_left(depth_est: StereoDepth, left: np.ndarray) -> np.ndarray:
    """
    Chỉ rectify ảnh trái (KHÔNG tính disparity/depth) – dùng cho chế độ
    realtime để YOLO chạy trên cùng hệ toạ độ với depth map khi bấm T.
    """
    calib_h, calib_w = depth_est.left_map_x.shape[:2]
    h, w = left.shape[:2]
    if (w, h) != (calib_w, calib_h):
        left = cv2.resize(left, (calib_w, calib_h), interpolation=cv2.INTER_LINEAR)
    return cv2.remap(left, depth_est.left_map_x, depth_est.left_map_y,
                     cv2.INTER_LINEAR)


def extract_detections(yolo_result, names, max_angle_deg: float = MAX_BAR_ANGLE_DEG) -> list[dict]:
    """
    Chuyển kết quả YOLO thành list dict, sắp xếp theo confidence giảm dần.
    Chỉ giữ lại những thanh có góc lệch phương ngang <= max_angle_deg (tối đa 25 độ).
    """
    dets: list[dict] = []
    obb = getattr(yolo_result, "obb", None)
    if obb is None or len(obb) == 0:
        return dets

    # OBB: 4 góc của box xoay (N,4,2) – dùng trực tiếp để tính góc chuẩn
    corners_all = obb.xyxyxyxy.cpu().numpy().reshape(-1, 4, 2)
    cls_all = obb.cls.cpu().numpy().astype(int)
    conf_all = obb.conf.cpu().numpy()

    for i in range(len(obb)):
        corners = corners_all[i]
        x1, y1 = (int(v) for v in np.floor(corners.min(axis=0)))
        x2, y2 = (int(v) for v in np.ceil(corners.max(axis=0)))
        cls_id = int(cls_all[i])
        conf_v = float(conf_all[i])
        mask_poly: Optional[np.ndarray] = corners.astype(np.int32)

        axis = bar_axis_2d((x1, y1, x2, y2), mask_poly)
        if max_angle_deg is not None and abs(axis["angle_deg"]) > max_angle_deg:
            continue  # Bỏ qua những thanh có góc lệch phương ngang > 25 độ

        dets.append({
            "box":       (x1, y1, x2, y2),
            "name":      names[cls_id],
            "conf":      conf_v,
            "mask_poly": mask_poly,
            "axis":      axis,
        })

    dets.sort(key=lambda d: d["conf"], reverse=True)
    return dets


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

    depth_m = float(to_metres(float(np.median(core)), calib_unit))

    if depth_m <= 0 or not np.isfinite(depth_m):
        return None, None, None

    box_w_px = x2c - x1c
    box_h_px = y2c - y1c

    if focal_px <= 0:
        return depth_m, None, None

    width_m  = (box_w_px * depth_m) / focal_px
    height_m = (box_h_px * depth_m) / focal_px

    return depth_m, width_m, height_m


# ===========================================================================
# Hình học thanh đáp: trục 2D trong ảnh + góc yaw 3D từ depth
# ===========================================================================

def bar_axis_2d(
    box: tuple[int, int, int, int],
    mask_poly: np.ndarray | None,
) -> dict:
    """
    Ước lượng trục chính của thanh trong ảnh (không cần depth).

    Dùng minAreaRect trên mask segmentation (nếu có), ngược lại dùng bbox.

    Returns dict:
        center     – (cx, cy) tâm thanh (px)
        dir        – vector đơn vị dọc theo thanh (luôn hướng sang phải)
        angle_deg  – góc của thanh so với trục ngang ảnh.
                     Dương = đầu phải cao hơn (nghiêng ngược chiều kim đồng hồ)
        length_px, thick_px – kích thước dọc/ngang thanh trong ảnh
        p1, p2     – 2 đầu thanh (trái, phải)
        from_mask  – True nếu tính từ mask (góc đáng tin cậy)
    """
    x1, y1, x2, y2 = box
    if mask_poly is not None and len(mask_poly) >= 4:
        pts = mask_poly.reshape(-1, 2).astype(np.float32)
        rect = cv2.minAreaRect(pts)
        (cx, cy) = rect[0]
        corners = cv2.boxPoints(rect)
        e0 = corners[1] - corners[0]
        e1 = corners[2] - corners[1]
        n0, n1 = float(np.linalg.norm(e0)), float(np.linalg.norm(e1))
        if n0 >= n1:
            v, length, thick = e0 / max(n0, 1e-6), n0, n1
        else:
            v, length, thick = e1 / max(n1, 1e-6), n1, n0
        from_mask = True
    else:
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        bw, bh = float(x2 - x1), float(y2 - y1)
        if bw >= bh:
            v, length, thick = np.array([1.0, 0.0]), bw, bh
        else:
            v, length, thick = np.array([0.0, 1.0]), bh, bw
        from_mask = False

    v = np.asarray(v, dtype=np.float64)
    if v[0] < 0 or (abs(v[0]) < 1e-9 and v[1] > 0):
        v = -v

    # Trục y ảnh hướng xuống → đổi dấu để góc dương = ngược chiều kim đồng hồ
    angle = math.degrees(math.atan2(-v[1], v[0]))
    half = length / 2.0
    p1 = (cx - v[0] * half, cy - v[1] * half)
    p2 = (cx + v[0] * half, cy + v[1] * half)

    return {
        "center":    (float(cx), float(cy)),
        "dir":       (float(v[0]), float(v[1])),
        "angle_deg": float(angle),
        "length_px": float(length),
        "thick_px":  float(thick),
        "p1":        p1,
        "p2":        p2,
        "from_mask": from_mask,
    }


def detection_region_mask(
    shape: tuple[int, int],
    box: tuple[int, int, int, int],
    mask_poly: np.ndarray | None,
) -> np.ndarray:
    """Tạo mask bool vùng thanh (ưu tiên polygon segmentation, fallback bbox)."""
    h, w = shape[:2]
    region = np.zeros((h, w), dtype=np.uint8)
    if mask_poly is not None and len(mask_poly) >= 3:
        cv2.fillPoly(region, [mask_poly.reshape((-1, 1, 2)).astype(np.int32)], 1)
    else:
        x1, y1, x2, y2 = box
        region[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 1
    return region.astype(bool)


def bar_yaw_from_depth(
    depth_map: np.ndarray,
    region: np.ndarray,
    focal_px: float,
    cx_pp: float,
    calib_unit: str = "mm",
) -> tuple[float | None, float | None]:
    """
    Tính góc yaw drone cần xoay để song song với thanh (camera nhìn trước).

    Chiếu các pixel thuộc thanh sang 3D (X ngang, Z sâu), fit đường thẳng
    Z = a*X + b trong mặt phẳng ngang. Nếu thanh song song với drone thì
    a ≈ 0 (hai đầu cùng khoảng cách).

    Returns:
        yaw_cmd_deg – góc cần xoay drone. Dương = xoay PHẢI (CW nhìn từ trên),
                      âm = xoay TRÁI. None nếu không đủ dữ liệu.
        length_m    – chiều dài thanh 3D (theo đường fit) hoặc None.
    """
    if focal_px <= 0:
        return None, None

    valid = region & np.isfinite(depth_map) & (depth_map > 0)
    ys, xs = np.nonzero(valid)
    if xs.size < 30:
        return None, None

    Z = to_metres(depth_map[ys, xs].astype(np.float64), calib_unit)
    med = float(np.median(Z))
    keep = np.abs(Z - med) < max(0.25 * med, 0.05)   # loại nền / nhiễu
    if keep.sum() < 30:
        return None, None
    xs_k, Z = xs[keep].astype(np.float64), Z[keep]
    X = (xs_k - cx_pp) * Z / focal_px

    x_lo, x_hi = np.percentile(X, 2), np.percentile(X, 98)
    if (x_hi - x_lo) < 0.03:      # thanh quá ngắn theo phương ngang → không fit
        return None, None

    # Fit 2 vòng: vòng 2 loại outlier residual lớn
    a, b = np.polyfit(X, Z, 1)
    resid = np.abs(Z - (a * X + b))
    thr = max(3.0 * float(np.median(resid)), 0.01)
    inl = resid < thr
    if inl.sum() >= 20:
        a, b = np.polyfit(X[inl], Z[inl], 1)
        X = X[inl]
        x_lo, x_hi = np.percentile(X, 2), np.percentile(X, 98)

    # Thanh có hướng (1, a) trong mặt phẳng XZ. Hướng drone cần quay tới
    # vuông góc với thanh: θ = atan2(-a, 1) → a > 0 (đầu phải xa hơn) → xoay trái.
    yaw_cmd_deg = math.degrees(math.atan(-a))
    length_m = float((x_hi - x_lo) * math.sqrt(1.0 + a * a))
    return float(yaw_cmd_deg), length_m


def compute_guidance(
    angle_deg: float,
    center: tuple[float, float],
    img_w: int,
    img_h: int,
    yaw_cmd_deg: float | None = None,
    dist_m: float | None = None,
    focal_px: float | None = None,
) -> dict:
    """
    Tổng hợp thông tin căn chỉnh drone so với thanh + 2 trục dọc/ngang.

    Returns dict gồm các cờ OK và list (text, colour) để hiển thị.
    """
    cx0, cy0 = img_w / 2.0, img_h / 2.0
    dx = center[0] - cx0      # + = thanh nằm bên phải trục dọc
    dy = center[1] - cy0      # + = thanh nằm dưới trục ngang

    if CAMERA_MOUNT == "down" and yaw_cmd_deg is None:
        # Camera nhìn xuống: góc trong ảnh chính là yaw.
        # Thanh nghiêng CCW (đầu phải cao) → xoay drone TRÁI.
        yaw_cmd_deg = -angle_deg

    angle_ok  = abs(angle_deg) <= ANGLE_TOL_DEG
    center_ok = abs(dx) <= CENTER_TOL_PX and abs(dy) <= CENTER_TOL_PX
    yaw_ok    = None if yaw_cmd_deg is None else abs(yaw_cmd_deg) <= YAW_TOL_DEG

    def off_str(px: float) -> str:
        if dist_m and focal_px:
            return f"{px:+.0f}px ({fmt_m(abs(px) * dist_m / focal_px)})"
        return f"{px:+.0f}px"

    lines: list[tuple[str, tuple]] = []

    # --- Yaw ---------------------------------------------------------
    if yaw_cmd_deg is None:
        lines.append(("Yaw : press T (needs depth)", COLOR_MUTED))
    elif yaw_ok:
        lines.append((f"Yaw : OK ({yaw_cmd_deg:+.1f} deg)", COLOR_OK))
    else:
        side = "RIGHT" if yaw_cmd_deg > 0 else "LEFT"
        lines.append((f"Yaw : ROTATE {side} {abs(yaw_cmd_deg):.1f} deg", COLOR_WARN))

    # --- Tilt trong ảnh (camera nhìn trước) --------------------------
    if CAMERA_MOUNT != "down":
        if angle_ok:
            lines.append((f"Tilt: OK ({angle_deg:+.1f} deg)", COLOR_OK))
        else:
            rot = "CCW" if angle_deg > 0 else "CW"
            lines.append((f"Tilt: {angle_deg:+.1f} deg -> rotate {rot}", COLOR_WARN))

    # --- Lệch theo trục ngang (X) ------------------------------------
    if abs(dx) <= CENTER_TOL_PX:
        lines.append((f"X   : centered ({dx:+.0f}px)", COLOR_OK))
    else:
        lines.append((f"X   : {off_str(dx)} -> move {'RIGHT' if dx > 0 else 'LEFT'}",
                      COLOR_WARN))

    # --- Lệch theo trục dọc (Y) --------------------------------------
    if abs(dy) <= CENTER_TOL_PX:
        lines.append((f"Y   : centered ({dy:+.0f}px)", COLOR_OK))
    else:
        if CAMERA_MOUNT == "down":
            mv = "BACK" if dy > 0 else "FORWARD"
        else:
            mv = "DOWN" if dy > 0 else "UP"
        lines.append((f"Y   : {off_str(dy)} -> move {mv}", COLOR_WARN))

    aligned = bool(center_ok and yaw_ok is True
                   and (angle_ok or CAMERA_MOUNT == "down"))
    if aligned:
        title = ("ALIGNMENT: ALIGNED", COLOR_OK)
    elif yaw_ok is None:
        title = ("ALIGNMENT: SEARCHING", COLOR_DETECT)
    else:
        title = ("ALIGNMENT: ADJUST", COLOR_WARN)

    return {
        "angle_deg":   angle_deg,
        "dx_px":       dx,
        "dy_px":       dy,
        "yaw_cmd_deg": yaw_cmd_deg,
        "angle_ok":    angle_ok,
        "center_ok":   center_ok,
        "yaw_ok":      yaw_ok,
        "aligned":     aligned,
        "lines":       [title] + lines,
    }


def assess_perch(
    dist_m:   float | None,
    width_m:  float | None,
    height_m: float | None,
    length_override: float | None = None,
    width_override:  float | None = None,
) -> tuple[bool | None, str]:
    """
    Đánh giá xem thanh ngang có đáp được không.

    length_override / width_override: kích thước chính xác hơn (từ fit 3D /
    mask xoay) nếu có, thay cho kích thước bbox.

    Returns:
        (True = đáp được, False = không, None = thiếu dữ liệu)
        Chuỗi lý do/thông báo (ASCII để hiển thị được bằng cv2.putText).
    """
    if dist_m is None or width_m is None or height_m is None:
        return None, "NO DEPTH DATA"

    length_m = length_override if length_override else max(width_m, height_m)
    bar_w_m  = width_override  if width_override  else min(width_m, height_m)

    reasons = []
    ok = True

    if not (MIN_DIST_M <= dist_m <= MAX_DIST_M):
        ok = False
        reasons.append(f"Dist {dist_m:.2f}m not in [{MIN_DIST_M:.2f}, {MAX_DIST_M:.2f}]m")

    if not (MIN_LENGTH_M <= length_m <= MAX_LENGTH_M):
        ok = False
        reasons.append(f"Len {length_m*100:.1f}cm not in [{MIN_LENGTH_M*100:.0f}, {MAX_LENGTH_M*100:.0f}]cm")

    if not (MIN_WIDTH_M <= bar_w_m <= MAX_WIDTH_M):
        ok = False
        reasons.append(f"Width {bar_w_m*100:.1f}cm not in [{MIN_WIDTH_M*100:.0f}, {MAX_WIDTH_M*100:.0f}]cm")

    if ok:
        return True, "LANDABLE - CAN PERCH"
    else:
        return False, "NOT LANDABLE: " + "; ".join(reasons)


def fmt_m(val: float | None) -> str:
    """Định dạng giá trị mét thành chuỗi dễ đọc."""
    if val is None:
        return "N/A"
    if val < 1.0:
        return f"{val * 100:.1f} cm"
    return f"{val:.3f} m"


def fmt_deg(val: float | None) -> str:
    return "N/A" if val is None else f"{val:+.1f} deg"


# ===========================================================================
# Vẽ overlay lên ảnh
# ===========================================================================

def draw_dashed_line(
    img: np.ndarray,
    p1: tuple[float, float],
    p2: tuple[float, float],
    colour: tuple,
    thickness: int = 1,
    dash: int = 12,
    gap: int = 8,
) -> None:
    """Vẽ đường nét đứt."""
    a0 = np.array(p1, dtype=np.float64)
    b0 = np.array(p2, dtype=np.float64)
    d = b0 - a0
    length = float(np.linalg.norm(d))
    if length < 1:
        return
    u = d / length
    s = 0.0
    while s < length:
        e = min(s + dash, length)
        a = a0 + u * s
        b = a0 + u * e
        cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                 colour, thickness, cv2.LINE_AA)
        s += dash + gap


def draw_alignment_guides(
    img: np.ndarray,
    axis: dict | None,
    guidance: dict | None,
) -> None:
    """
    Vẽ 2 trục căn dọc/ngang qua tâm ảnh (+ dải dung sai), trục chính của
    thanh, cung góc lệch và mũi tên lệch tâm.
    """
    h, w = img.shape[:2]
    cx0, cy0 = w // 2, h // 2

    # --- 2 trục căn dọc / ngang --------------------------------------
    draw_dashed_line(img, (cx0, 0), (cx0, h), COLOR_GUIDE, 1)
    draw_dashed_line(img, (0, cy0), (w, cy0), COLOR_GUIDE, 1)
    t = CENTER_TOL_PX
    for off in (-t, t):
        draw_dashed_line(img, (cx0 + off, 0), (cx0 + off, h), COLOR_MUTED, 1, 3, 12)
        draw_dashed_line(img, (0, cy0 + off), (w, cy0 + off), COLOR_MUTED, 1, 3, 12)
    cv2.circle(img, (cx0, cy0), 7, COLOR_GUIDE, 1, cv2.LINE_AA)
    cv2.putText(img, "V", (cx0 + 5, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                COLOR_GUIDE, 1, cv2.LINE_AA)
    cv2.putText(img, "H", (w - 18, cy0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                COLOR_GUIDE, 1, cv2.LINE_AA)

    if axis is None or guidance is None:
        return

    bx, by = axis["center"]
    vx, vy = axis["dir"]
    angle  = axis["angle_deg"]
    col    = COLOR_OK if guidance["angle_ok"] else COLOR_AXIS

    # Trục thanh kéo dài toàn ảnh (cv2.line tự clip)
    ext = float(max(w, h)) * 2
    cv2.line(img, (int(bx - vx * ext), int(by - vy * ext)),
             (int(bx + vx * ext), int(by + vy * ext)), col, 1, cv2.LINE_AA)
    # Đoạn thanh
    p1 = tuple(int(c) for c in axis["p1"])
    p2 = tuple(int(c) for c in axis["p2"])
    cv2.line(img, p1, p2, col, 3, cv2.LINE_AA)
    cv2.circle(img, p1, 5, col, -1, cv2.LINE_AA)
    cv2.circle(img, p2, 5, col, -1, cv2.LINE_AA)

    # Đường ngang tham chiếu qua tâm thanh + cung góc
    half = max(axis["length_px"] / 2.0, 40.0)
    draw_dashed_line(img, (bx - half, by), (bx + half, by), COLOR_GUIDE, 1, 6, 6)
    r = int(min(60.0, half))
    start, end = sorted((0.0, -angle))   # ellipse: góc theo chiều kim đồng hồ
    cv2.ellipse(img, (int(bx), int(by)), (r, r), 0, start, end, col, 2, cv2.LINE_AA)
    txt = f"{angle:+.1f} deg" + ("" if axis["from_mask"] else " (bbox)")
    org = (int(bx + r + 6), int(by - 8))
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)

    # Mũi tên từ tâm ảnh → tâm thanh
    off_col = COLOR_OK if guidance["center_ok"] else COLOR_WARN
    if abs(guidance["dx_px"]) + abs(guidance["dy_px"]) > 4:
        cv2.arrowedLine(img, (cx0, cy0), (int(bx), int(by)), off_col, 2,
                        cv2.LINE_AA, tipLength=0.06)


def draw_guidance_panel(img: np.ndarray, guidance: dict | None) -> None:
    """Panel căn chỉnh ở góc trên-phải."""
    if guidance is None:
        lines = [("ALIGNMENT: NO TARGET", COLOR_MUTED),
                 ("Searching perch bar ...", COLOR_MUTED)]
        border = COLOR_MUTED
    else:
        lines  = guidance["lines"]
        border = lines[0][1]

    font, scale, th = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    pad, line_h = 8, 22
    h, w = img.shape[:2]
    tw = max(cv2.getTextSize(t, font, scale, th)[0][0] for t, _ in lines)
    x0 = w - tw - pad * 2 - 10
    y0 = 10
    x1 = w - 10
    y1 = y0 + pad * 2 + line_h * len(lines) - 6

    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), COLOR_HUD_BG, -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
    cv2.rectangle(img, (x0, y0), (x1, y1), border, 1, cv2.LINE_AA)

    for i, (t, c) in enumerate(lines):
        ty = y0 + pad + 14 + i * line_h
        cv2.putText(img, t, (x0 + pad, ty), font, scale, c,
                    2 if i == 0 else th, cv2.LINE_AA)


def draw_yaw_indicator(img: np.ndarray, yaw_cmd_deg: float | None, y: int) -> None:
    """Thông báo lớn giữa màn hình: cần xoay drone TRÁI/PHẢI bao nhiêu độ."""
    if yaw_cmd_deg is None:
        text, col = "YAW: N/A (not enough depth on bar)", COLOR_MUTED
    elif abs(yaw_cmd_deg) <= YAW_TOL_DEG:
        text, col = f"YAW ALIGNED - PARALLEL TO BAR ({yaw_cmd_deg:+.1f} deg)", COLOR_OK
    elif yaw_cmd_deg > 0:
        text, col = f"ROTATE DRONE RIGHT {abs(yaw_cmd_deg):.1f} deg  >>>", COLOR_WARN
    else:
        text, col = f"<<<  ROTATE DRONE LEFT {abs(yaw_cmd_deg):.1f} deg", COLOR_WARN

    font, scale, th = cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
    (tw, tht), _ = cv2.getTextSize(text, font, scale, th)
    w = img.shape[1]
    x = (w - tw) // 2
    overlay = img.copy()
    cv2.rectangle(overlay, (x - 12, y - tht - 12), (x + tw + 12, y + 10), COLOR_HUD_BG, -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    cv2.rectangle(img, (x - 12, y - tht - 12), (x + tw + 12, y + 10), col, 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, col, th, cv2.LINE_AA)


def draw_perch_detection(
    img: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    label: str,
    dist_m: float | None,
    width_m: float | None,
    height_m: float | None,
    verdict: bool | None,
    mask_polygon: np.ndarray | None = None,
    axis: dict | None = None,
) -> None:
    """Vẽ detection box xoay chéo (Oriented Bounding Box), nhãn kích thước (nếu có depth), verdict lên ảnh."""
    if verdict is True:
        colour = COLOR_OK
    elif verdict is False:
        colour = COLOR_FAIL
    else:
        colour = COLOR_DETECT

    overlay = img.copy()
    alpha   = 0.28

    # Lấy thông tin trục và góc xoay của thanh
    if axis is None:
        axis = bar_axis_2d((x1, y1, x2, y2), mask_polygon)

    cx_f, cy_f = axis["center"]
    vx, vy = axis["dir"]
    nx, ny = -vy, vx
    hl = axis["length_px"] / 2.0
    hw = axis["thick_px"] / 2.0

    # 4 đỉnh của khung hình chữ nhật xoay chéo bám sát theo thanh (Rotated Bounding Box)
    rot_box = np.array([
        (cx_f + vx * hl + nx * hw, cy_f + vy * hl + ny * hw),
        (cx_f - vx * hl + nx * hw, cy_f - vy * hl + ny * hw),
        (cx_f - vx * hl - nx * hw, cy_f - vy * hl - ny * hw),
        (cx_f + vx * hl - nx * hw, cy_f + vy * hl - ny * hw),
    ], dtype=np.int32)

    cx, cy = int(cx_f), int(cy_f)

    # 1. Fill nền mờ + vẽ 4 cạnh hình chữ nhật xoay chéo
    cv2.fillPoly(overlay, [rot_box], colour)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.polylines(img, [rot_box], True, colour, 2, cv2.LINE_AA)
    bright = tuple(min(255, int(c * 1.5)) for c in colour)
    cv2.polylines(img, [rot_box], True, bright, 1, cv2.LINE_AA)

    # 2. Nếu có mask polygon segmentation thì vẽ viền nét mảnh theo mask
    if mask_polygon is not None and len(mask_polygon) >= 3:
        pts = mask_polygon.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(img, [pts], True, (0, 255, 255), 1, cv2.LINE_AA)

    # Crosshair tại tâm thanh
    arm = 14
    cv2.line(img, (cx - arm, cy), (cx + arm, cy), (0, 255, 180), 2)
    cv2.line(img, (cx, cy - arm), (cx, cy + arm), (0, 255, 180), 2)
    cv2.circle(img, (cx, cy), 5, (0, 255, 180), -1)

    # Thông tin nhãn (chế độ realtime không có depth → chỉ hiện label)
    if dist_m is None and width_m is None and height_m is None:
        lines = [label]
    else:
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
    yaw_cmd_deg: float | None = None,
) -> int:
    """Vẽ banner verdict + tóm tắt số liệu ở phía dưới ảnh. Trả về y đỉnh banner."""
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
    cv2.putText(img, reason, (12, y0 + 30), font, 0.65, v_colour, 2, cv2.LINE_AA)

    # Dòng số liệu
    stats = (
        f"Dist={fmt_m(dist_m)}  "
        f"Length={fmt_m(length_m)}  "
        f"Width={fmt_m(bar_w_m)}  "
        f"YawCmd={fmt_deg(yaw_cmd_deg)}"
    )
    cv2.putText(img, stats, (12, y0 + 58), font, 0.58, (210, 220, 255), 1, cv2.LINE_AA)

    # Thresholds reminder
    thr = (
        f"Thresholds - Dist:[{MIN_DIST_M*100:.0f},{MAX_DIST_M*100:.0f}]cm  "
        f"Len:[{MIN_LENGTH_M*100:.0f},{MAX_LENGTH_M*100:.0f}]cm  "
        f"W:[{MIN_WIDTH_M*100:.0f},{MAX_WIDTH_M*100:.0f}]cm  "
        f"Yaw tol:{YAW_TOL_DEG:.0f}deg   K=continue search"
    )
    cv2.putText(img, thr, (12, y0 + 80), font, 0.42, COLOR_MUTED, 1, cv2.LINE_AA)
    return y0


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
    fps_txt = f"  FPS: {fps:.1f}" if state == "SEARCHING" else ""
    lines = [
        f"State: {state}{fps_txt}",
        f"Detections: {n_det}  Conf: {conf:.2f}  Mount: {CAMERA_MOUNT}",
        "T=Analyze  K=Continue  D=Depth  S=Save  Q=Quit",
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
# Burst capture + analysis (chỉ chạy khi bấm T)
# ===========================================================================

class BurstResult:
    """Lưu kết quả tổng hợp từ N burst frames."""
    __slots__ = [
        "annotated", "depth_color", "detections_info",
        "best_dist_m", "best_length_m", "best_bar_w_m",
        "yaw_cmd_deg", "angle2d_deg", "guidance",
        "verdict", "reason", "n_frames",
    ]

    def __init__(self):
        self.annotated:        Optional[np.ndarray] = None
        self.depth_color:      Optional[np.ndarray] = None
        self.detections_info:  list[dict]           = []
        self.best_dist_m:      Optional[float]      = None
        self.best_length_m:    Optional[float]      = None
        self.best_bar_w_m:     Optional[float]      = None
        self.yaw_cmd_deg:      Optional[float]      = None
        self.angle2d_deg:      Optional[float]      = None
        self.guidance:         Optional[dict]       = None
        self.verdict:          Optional[bool]       = None
        self.reason:           str                  = ""
        self.n_frames:         int                  = 0


def _median(vals: list[float]) -> float | None:
    return float(np.median(vals)) if vals else None


def run_burst(
    source,
    depth_est: StereoDepth,
    yolo: YOLO,
    focal_px: float,
    n_burst: int = N_BURST,
    conf: float = 0.35,
) -> BurstResult:
    """
    Thu thập n_burst frame, chạy depth + YOLO trên từng frame, tính kích
    thước + góc yaw của thanh mục tiêu (detection conf cao nhất), tổng hợp
    kết quả bằng trung vị.
    """
    result = BurstResult()
    result.n_frames = n_burst
    cx_pp, _ = principal_point_from_Q(depth_est.Q)

    all_dist:   list[float] = []
    all_width:  list[float] = []
    all_height: list[float] = []
    all_len3d:  list[float] = []
    all_thick:  list[float] = []
    all_yaw:    list[float] = []
    all_angle:  list[float] = []
    all_cx:     list[float] = []
    all_cy:     list[float] = []

    last_rect:        Optional[np.ndarray] = None
    last_depth_color: Optional[np.ndarray] = None
    last_dets:        list[dict]           = []
    last_axis:        Optional[dict]       = None

    print(f"[BURST] Capturing {n_burst} frames …")

    for frame_i in range(n_burst):
        ok, fl, fr = source.read()
        if not ok or fl is None or fr is None:
            print(f"  [BURST] Frame {frame_i+1}: read failed, skipping.")
            continue

        # Depth (chỉ tính ở đây – không tính trong realtime)
        left_rect, _, depth_map = depth_est.process(fl, fr)

        # YOLO
        dets = extract_detections(yolo(left_rect, conf=conf, verbose=False)[0],
                                  yolo.names)

        for d in dets:
            x1, y1, x2, y2 = d["box"]
            d["dist_m"], d["width_m"], d["height_m"] = box_physical_size(
                depth_map, x1, y1, x2, y2, focal_px, CALIB_UNIT
            )

        frame_axis = None
        yaw_v = len3d = None
        if dets:
            tgt = dets[0]   # thanh mục tiêu = conf cao nhất
            frame_axis = bar_axis_2d(tgt["box"], tgt["mask_poly"])
            all_angle.append(frame_axis["angle_deg"])
            all_cx.append(frame_axis["center"][0])
            all_cy.append(frame_axis["center"][1])

            if tgt["dist_m"] is not None:
                all_dist.append(tgt["dist_m"])
                if frame_axis["from_mask"] and focal_px > 0:
                    all_thick.append(frame_axis["thick_px"] * tgt["dist_m"] / focal_px)
            if tgt["width_m"] is not None:
                all_width.append(tgt["width_m"])
            if tgt["height_m"] is not None:
                all_height.append(tgt["height_m"])

            region = detection_region_mask(depth_map.shape, tgt["box"], tgt["mask_poly"])
            yaw_v, len3d = bar_yaw_from_depth(depth_map, region, focal_px,
                                              cx_pp, CALIB_UNIT)
            if yaw_v is not None:
                all_yaw.append(yaw_v)
            if len3d is not None:
                all_len3d.append(len3d)

        # Giữ lại frame cuối để hiển thị
        last_rect        = left_rect
        last_depth_color = depth_to_color(depth_map, MAX_DEPTH_M, CALIB_UNIT)
        last_dets        = dets
        if frame_axis is not None:
            last_axis = frame_axis

        print(f"  [BURST] Frame {frame_i+1}/{n_burst}: {len(dets)} detection(s)"
              f"  yaw={fmt_deg(yaw_v)}")

    # -----------------------------------------------------------------------
    # Tổng hợp: lấy trung vị
    # -----------------------------------------------------------------------
    med_dist   = _median(all_dist)
    med_width  = _median(all_width)
    med_height = _median(all_height)
    med_len3d  = _median(all_len3d)
    med_thick  = _median(all_thick)
    med_yaw    = _median(all_yaw)
    med_angle  = _median(all_angle)

    verdict, reason = assess_perch(med_dist, med_width, med_height,
                                   length_override=med_len3d,
                                   width_override=med_thick)

    med_length = med_len3d or (max(med_width, med_height) if (med_width and med_height) else None)
    med_bar_w  = med_thick or (min(med_width, med_height) if (med_width and med_height) else None)

    # Góc yaw cần xoay
    if CAMERA_MOUNT == "down":
        yaw_cmd = -med_angle if med_angle is not None else None
    else:
        yaw_cmd = med_yaw

    result.best_dist_m   = med_dist
    result.best_length_m = med_length
    result.best_bar_w_m  = med_bar_w
    result.yaw_cmd_deg   = yaw_cmd
    result.angle2d_deg   = med_angle
    result.verdict       = verdict
    result.reason        = reason

    print(f"\n[RESULT] dist={fmt_m(med_dist)}  length={fmt_m(med_length)}  "
          f"width={fmt_m(med_bar_w)}")
    if yaw_cmd is None:
        print("[RESULT] Yaw: N/A (không đủ depth trên thanh)")
    elif abs(yaw_cmd) <= YAW_TOL_DEG:
        print(f"[RESULT] Yaw: đã song song với thanh ({yaw_cmd:+.1f} deg)")
    else:
        print(f"[RESULT] Yaw: cần xoay drone {'PHẢI' if yaw_cmd > 0 else 'TRÁI'} "
              f"{abs(yaw_cmd):.1f} deg")
    print(f"[RESULT] {reason}\n")

    # -----------------------------------------------------------------------
    # Vẽ annotated frame cuối
    # -----------------------------------------------------------------------
    if last_rect is not None:
        annotated = last_rect.copy()
        h_img, w_img = annotated.shape[:2]

        for j, info in enumerate(last_dets):
            x1, y1, x2, y2 = info["box"]
            v, _ = assess_perch(info["dist_m"], info["width_m"], info["height_m"])
            info["verdict"]  = v
            lv  = max(info["width_m"] or 0, info["height_m"] or 0) or None
            bwv = min(info["width_m"] or 0, info["height_m"] or 0) or None
            info["length_m"] = lv
            info["bar_w_m"]  = bwv

            draw_perch_detection(
                annotated,
                x1, y1, x2, y2,
                f"{'TARGET ' if j == 0 else ''}{info['name']} {info['conf']:.0%}",
                info["dist_m"], info["width_m"], info["height_m"],
                v,
                info.get("mask_poly"),
            )

        # Căn chỉnh (dùng giá trị trung vị)
        if med_angle is not None and all_cx:
            guidance = compute_guidance(
                med_angle, (_median(all_cx), _median(all_cy)),
                w_img, h_img,
                yaw_cmd_deg=yaw_cmd, dist_m=med_dist, focal_px=focal_px,
            )
        else:
            guidance = None
        result.guidance = guidance

        draw_alignment_guides(annotated, last_axis, guidance)
        draw_guidance_panel(annotated, guidance)

        # Verdict banner tổng hợp (trung vị)
        y_banner = draw_verdict_banner(
            annotated,
            verdict, reason,
            med_dist, med_length, med_bar_w,
            yaw_cmd,
        )
        if last_axis is not None:
            draw_yaw_indicator(annotated, yaw_cmd, y_banner - 18)

        result.annotated = annotated

    if last_depth_color is not None:
        result.depth_color     = draw_depth_annotation(last_depth_color, last_dets)
        result.detections_info = last_dets

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
    show_depth     = SHOW_DEPTH_ON_START
    depth_win_open = False
    burst_result: Optional[BurstResult] = None
    state          = "SEARCHING"   # SEARCHING | ANALYZING | RESULT
    fps_display    = 0.0
    t_last         = time.monotonic()
    snapshot_dir   = Path(SNAPSHOT_DIR)
    n_det_live     = 0
    conf_thresh    = YOLO_CONF
    display: Optional[np.ndarray] = None

    WIN_MAIN  = "Perch Detector  [T=Analyze  K=Continue  D=Depth  S=Save  Q=Quit]"
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
    print(f"    Góc ngang max: ±{MAX_BAR_ANGLE_DEG:.1f} deg")
    print(f"    Yaw tol     : ±{YAW_TOL_DEG:.1f} deg   |   Center tol: ±{CENTER_TOL_PX}px")
    print(f"  Camera mount  : {CAMERA_MOUNT}")
    print(f"  Burst size    : {N_BURST} frames")
    print(f"  YOLO conf     : {YOLO_CONF}")
    print("=" * 62)
    print("  Realtime: chỉ tìm thanh đáp (không depth)")
    print("  T = Phân tích (depth + yaw + verdict)  |  K = Tiếp tục tìm")
    print("  D = Depth  |  S = Save  |  Q = Thoát")
    print("=" * 62 + "\n")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    while True:
        if state == "RESULT" and burst_result is not None and burst_result.annotated is not None:
            # ------------------------------------------------------------
            # Đóng băng kết quả phân tích – không xử lý frame mới
            # ------------------------------------------------------------
            display = burst_result.annotated.copy()
            draw_hud_top(display, state, len(burst_result.detections_info),
                         fps_display, conf_thresh)
        else:
            # ------------------------------------------------------------
            # SEARCHING: chỉ rectify ảnh trái + YOLO (KHÔNG tính depth)
            # ------------------------------------------------------------
            state = "SEARCHING"
            t_now       = time.monotonic()
            fps_display = 1.0 / max(t_now - t_last, 1e-6)
            t_last      = t_now

            ok, fl, fr = source.read()
            if not ok or fl is None:
                time.sleep(0.01)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                continue

            left_rect = rectify_left(depth_est, fl)
            dets = extract_detections(
                yolo(left_rect, conf=conf_thresh, verbose=False)[0], yolo.names
            )
            n_det_live = len(dets)

            display = left_rect.copy()
            for j, d in enumerate(dets):
                x1, y1, x2, y2 = d["box"]
                draw_perch_detection(
                    display, x1, y1, x2, y2,
                    f"{'TARGET ' if j == 0 else ''}{d['name']} {d['conf']:.0%}",
                    None, None, None, None, d["mask_poly"],
                )

            h_img, w_img = display.shape[:2]
            if dets:
                axis = bar_axis_2d(dets[0]["box"], dets[0]["mask_poly"])
                guidance = compute_guidance(axis["angle_deg"], axis["center"],
                                            w_img, h_img)
            else:
                axis, guidance = None, None

            draw_alignment_guides(display, axis, guidance)
            draw_guidance_panel(display, guidance)
            draw_hud_top(display, state, n_det_live, fps_display, conf_thresh)

        # Hiển thị
        cv2.imshow(WIN_MAIN, display)

        # Cửa sổ depth chỉ có khi đã phân tích (bấm T)
        want_depth = (show_depth and state == "RESULT" and burst_result is not None
                      and burst_result.depth_color is not None)
        if want_depth:
            cv2.imshow(WIN_DEPTH, burst_result.depth_color)
            depth_win_open = True
        elif depth_win_open:
            cv2.destroyWindow(WIN_DEPTH)
            depth_win_open = False

        # ----------------------------------------------------------------
        # Xử lý phím bấm
        # ----------------------------------------------------------------
        key = cv2.waitKey(1 if state == "SEARCHING" else 30) & 0xFF

        if key in (ord("q"), 27):       # Q / ESC → thoát
            break

        elif key == ord("t"):           # T → phân tích (burst + depth + yaw)
            state = "ANALYZING"
            busy = display.copy()
            draw_hud_top(busy, state, n_det_live, fps_display, conf_thresh)
            draw_yaw_indicator(busy, None, busy.shape[0] // 2)
            cv2.putText(busy, "ANALYZING (depth + yaw) ...",
                        (busy.shape[1] // 2 - 170, busy.shape[0] // 2 - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_DETECT, 2, cv2.LINE_AA)
            cv2.imshow(WIN_MAIN, busy)
            cv2.waitKey(1)

            burst_result = run_burst(
                source, depth_est, yolo, focal_px,
                n_burst=N_BURST,
                conf=conf_thresh,
            )
            state = "RESULT"

        elif key in (ord("k"), ord("r")):   # K / R → tiếp tục tìm
            if state == "RESULT":
                print("[INFO] Tiếp tục quá trình tìm thanh đáp …")
            burst_result = None
            state = "SEARCHING"
            t_last = time.monotonic()

        elif key == ord("d"):           # D → toggle depth map
            show_depth = not show_depth
            print(f"[INFO] Depth window: {'ON' if show_depth else 'OFF'}")

        elif key == ord("s"):           # S → save snapshot
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            tag = time.strftime("%Y%m%d_%H%M%S")
            if state == "RESULT" and burst_result is not None and burst_result.annotated is not None:
                cv2.imwrite(
                    str(snapshot_dir / f"{tag}_annotated.png"),
                    burst_result.annotated,
                )
                if burst_result.depth_color is not None:
                    cv2.imwrite(
                        str(snapshot_dir / f"{tag}_depth.png"),
                        burst_result.depth_color,
                    )
            else:
                cv2.imwrite(str(snapshot_dir / f"{tag}_live.png"), display)
            print(f"[INFO] Snapshot saved → {snapshot_dir}/{tag}_*.png")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    source.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
