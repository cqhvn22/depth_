# Lenna Stereo Camera – Raspberry Pi 5 Setup

Hướng dẫn này giúp bạn chạy toàn bộ pipeline stereo camera trên **Raspberry Pi 5** với **picamera2**.

---

## Phần cứng

| Yêu cầu | Chi tiết |
|---|---|
| Board | Raspberry Pi 5 |
| Camera | 2× camera CSI (IMX219 / IMX477 / IMX708, v.v.) |
| Kết nối | Camera trái → **CAM0**, Camera phải → **CAM1** |
| OS | Raspberry Pi OS Bookworm (64-bit, recommended) |

> **Lưu ý:** Raspberry Pi 5 có **2 cổng CSI** riêng biệt (CAM0 và CAM1), không giống Pi 4 chỉ có 1.  
> Câu lệnh mặc định dùng `--left-sensor 0` (CAM0) và `--right-sensor 1` (CAM1).

---

## Cài đặt phụ thuộc

```bash
# Cập nhật hệ thống
sudo apt update && sudo apt upgrade -y

# Cài picamera2 (thường đã có sẵn trên Raspberry Pi OS Bookworm)
sudo apt install -y python3-picamera2

# Cài OpenCV với contrib (cho WLS filter trong stereo_depth.py)
pip install opencv-contrib-python

# Hoặc nếu muốn cài toàn bộ:
pip install opencv-contrib-python numpy
```

### Kích hoạt camera

Đảm bảo camera được bật trong raspi-config:

```bash
sudo raspi-config
# → Interface Options → Camera → Enable
sudo reboot
```

Kiểm tra camera được nhận diện:

```bash
# Liệt kê camera
libcamera-hello --list-cameras
```

---

## Cách chạy

### 1. Test camera cơ bản

```bash
cd Python
python3 camera_test.py
```

Nhấn **ESC** để thoát.

### 2. Test độ sâu (depth)

Cần file `stereo_calibration.npz` trước (xem bước 3–4):

```bash
cd Python
python3 depth_test.py
```

### 3. Chụp ảnh hiệu chỉnh (Calibration capture)

```bash
cd Calibration
python3 stereo_calibration_capture.py \
    --left-sensor 0 \
    --right-sensor 1 \
    --capture-width 1920 \
    --capture-height 1080 \
    --output-width 960 \
    --output-height 540 \
    --framerate 30 \
    --board-size 9x6 \
    --target-pairs 30 \
    --output calibration_images
```

**Điều khiển:**
- `SPACE` hoặc `S` → Lưu cặp ảnh
- `A` → Bật/tắt chụp tự động
- `R` → Reset bộ đếm
- `Q` hoặc `ESC` → Thoát

### 4. Tính toán hiệu chỉnh (Calibration)

```bash
cd Calibration
python3 calibration.py
```

Kết quả được lưu vào `stereo_calibration.npz`.

---

## Tham số flip-method

Nếu camera bị lật ngược, dùng `--flip-method`:

| Giá trị | Kết quả |
|---|---|
| 0 | Không lật (mặc định) |
| 2 | Xoay 180° |
| 4 | Lật ngang (horizontal flip) |
| 6 | Lật dọc (vertical flip) |
| 1 | Xoay 90° ngược chiều kim đồng hồ |
| 3 | Xoay 90° theo chiều kim đồng hồ |

---

## So sánh với Jetson

| | NVIDIA Jetson | Raspberry Pi 5 |
|---|---|---|
| Camera API | GStreamer + `nvarguscamerasrc` | `picamera2` |
| Camera backend | `cv2.CAP_GSTREAMER` | `Picamera2` native |
| Camera ports | Sensor ID 0, 1 | CAM0, CAM1 |
| Flip | `nvvidconv flip-method` | `cv2.rotate` / `cv2.flip` |

---

## Lỗi thường gặp

**`ImportError: No module named 'picamera2'`**
```bash
sudo apt install python3-picamera2
```

**`RuntimeError: Could not open camera`**
- Kiểm tra cáp ribbon camera được cắm đúng vào cổng CAM0/CAM1
- Chạy `libcamera-hello --list-cameras` để xem camera có được nhận không
- Đảm bảo không có process khác đang dùng camera

**`ImportError: No module named 'cv2.ximgproc'`** (cho depth_test.py)
```bash
pip install opencv-contrib-python
```

**Frame timing difference cao (> 50 ms)**  
- Với picamera2, hai camera chạy trên thread độc lập, không có hardware sync
- Nếu cần sync chặt hơn, giảm tốc độ khung hình (`--framerate 15`) hoặc tăng ngưỡng `--max-time-difference`
