# Perch detector trên Raspberry Pi 5 — NCNN FP16

`Python/perch_detector.py` đã áp dụng chung thuật toán trong `perch_video_seg.py`:
hai mắt có mask, hướng dẫn căn góc 2D, T thu burst 7 cặp mới, median riêng một
target và giữ kết quả. SGBM/WLS và Plasma vẫn dùng nguyên `StereoDepth.process()`
và `depth_to_color()` của `Calibration/video_depth_estimation.py`.

Mô hình là **YOLOv8n-seg đã train trong `best-seg.pt`**, không thay bằng trọng số
COCO. File detector cũ được giữ tại `Python/perch_detector_legacy.py` trong dự án
gốc. Bản mới dùng NCNN + NumPy/OpenCV, **không cần PyTorch hoặc Ultralytics trên Pi**.

## Chuyển sang Pi và cài đặt

Dùng **Raspberry Pi OS 64-bit**; chạy `uname -m` phải ra `aarch64` hoặc `arm64`.
Gói `perch_pi5_deploy.zip` chứa hai model đã chuyển đổi và các file chạy cần thiết.
Chép sang Pi, giải nén rồi vào thư mục `perch_pi5/`. Nếu chép toàn dự án thì vào
thư mục gốc dự án. Không chép `.venv` Windows sang Pi.

```bash
bash Python/setup_perch_pi5.sh
.venv-pi5/bin/python Python/perch_detector.py
```

Script cài đặt dùng apt cho Picamera2/OpenCV/NumPy/PyYAML, tạo venv có
`--system-site-packages`, rồi cài NCNN bằng `--no-deps` để không kéo một bản
OpenCV khác đè lên bản contrib của hệ thống. Lệnh cài thủ công tương đương:

```bash
sudo apt update
sudo apt install -y python3-venv python3-picamera2 python3-opencv python3-numpy python3-yaml libgomp1 libvulkan1
python3 -m venv --system-site-packages .venv-pi5
.venv-pi5/bin/python -m pip install --no-deps -r Python/requirements-perch-pi5.txt
```

Kiểm tra camera trước khi chạy:

```bash
rpicam-hello --list-cameras
.venv-pi5/bin/python -c "import cv2, ncnn; print(cv2.__version__, ncnn.__version__); assert hasattr(cv2.ximgproc, 'createRightMatcher')"
```

Mặc định: CAM0 trái, CAM1 phải, 960×540 ở 15 FPS. Chỉnh trực tiếp CONFIG ở đầu
`perch_detector.py`, hoặc truyền tham số:

```bash
.venv-pi5/bin/python Python/perch_detector.py --left-camera 0 --right-camera 1 --camera-fps 15
```

Calibration phải thuộc đúng hai camera, thứ tự trái/phải, trường nhìn và camera
mode đang dùng. Nếu đã hiệu chỉnh ở capture 1920×1080 rồi resize về 960×540,
có thể giữ cấu hình capture đó bằng `--width 1920 --height 1080`. Không dùng
flip/rotate mới mà chưa hiệu chỉnh lại.

## Điều khiển

| Phím | Chức năng |
|---|---|
| T | Thu 7 cặp camera mới sau trigger, đo và đánh giá, đóng băng RESULT |
| R | Xóa kết quả, về live |
| Space | Pause/resume; từ RESULT quay về live |
| D | Hiện/ẩn ô depth |
| S | Lưu ảnh hai mắt, màn hình 4 ô, mask trái, depth mét và JSON kết quả |
| Q / Esc | Thoát và giải phóng camera |

LIVE chỉ segmentation/căn góc. Kích thước/khoảng cách dùng để kết luận chỉ tính
trong burst sau T, ít nhất 3 cặp hợp lệ. Hướng dẫn là độ nghiêng và vị trí thanh
trong ảnh 2D, không phải ước lượng đầy đủ tư thế 3D hoặc lệnh điều khiển drone.
Ngưỡng vẫn là khoảng cách 0.20–5.00 m, dài 0.10–2.00 m, rộng 0.02–0.20 m.

Camera adapter dùng `last_sequence`, bỏ các sequence trùng và bỏ cặp đang đệm
trước T. `camera.py` cấu hình Picamera2 **BGR888**, có byte order RGB, nên adapter
chuyển sang BGR của OpenCV. Nếu bạn đã sửa `camera.py` sang RGB888 (byte order
BGR), chạy `--camera-color bgr` để tránh đổi màu hai lần. Quy ước này theo
[tài liệu Picamera2](https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf).

`delta_ms` của lớp camera hiện tại là chênh lệch timestamp lúc lấy ảnh bằng phần
mềm, chưa phải timestamp phơi sáng hoặc đồng bộ phần cứng. Tùy chọn
`--max-pair-gap-ms` chỉ lọc chênh lệch lấy ảnh khi bạn đã đo được ngưỡng phù hợp;
mặc định không lọc để tránh bỏ hết cặp do hai lần capture tuần tự.

## Model và tối ưu CPU

| Profile | Thư mục model | Input H×W | Trọng số `.bin` |
|---|---|---|---|
| fast (mặc định) | `models/perch_pi5_640_fp16_ncnn_model/` | 384×640 | 6,589,120 bytes |
| detail | `models/perch_pi5_960_fp16_ncnn_model/` | 544×960 | 6,657,160 bytes |

Input chữ nhật giảm padding so với input vuông mà vẫn giữ tỷ lệ hình ảnh. Bản
640 xử lý số pixel bằng khoảng 47% bản 960. FP16 giảm trọng số bản 640 khoảng
49.7% so với NCNN FP32 (13,093,824 bytes). Checkpoint `.pt` gốc vốn đã lưu phần
lớn trọng số FP16 (6,855,604 bytes), nên không thể nói nhỏ hơn `.pt` gốc 2 lần.
NCNN còn fuse các phép toán và dùng kernel ARM/packing; runtime này bỏ cả chi
phí nạp PyTorch/Ultralytics. NCNN được chọn theo
[hướng dẫn Raspberry Pi của Ultralytics](https://docs.ultralytics.com/guides/raspberry-pi/).

Ultralytics NCNN exporter hiện hỗ trợ FP32/FP16, chưa có tùy chọn INT8 cho đường
export này. Không ép `int8=True` để tránh tạo mô hình không được hỗ trợ. INT8 ở
backend khác cần đánh giá riêng chất lượng mask; không được coi là tự động tốt
hơn chỉ vì file nhỏ hơn. Tham khảo [NCNN export](https://docs.ultralytics.com/integrations/ncnn/).

Chọn profile chất lượng cao hơn:

```bash
.venv-pi5/bin/python Python/perch_detector.py --profile detail
```

Mặc định NCNN 3 thread, OpenCV 1 thread; không chạy hai mạng song song tranh cả
4 lõi với stereo/capture. Có thể thử `--threads 2` hoặc `--threads 4` sau khi đo
trên máy. `--fp32-arithmetic` giữ storage FP16 nhưng tắt arithmetic FP16 nếu cần
đối chiếu số học trên ARM.

Giảm chi phí mask mắt phải khi cần (hai ảnh vẫn cập nhật, không vẽ mask cũ lên
frame mới ở các cặp bỏ suy luận phải):

```bash
.venv-pi5/bin/python Python/perch_detector.py --right-mask-every 3
```

Không giảm độ phân giải stereo xuống khác thuật toán tham chiếu để tăng FPS.
`--downscale` mặc định 0.5, cùng tham số SGBM/WLS và cách nội suy của file gốc.

### Đo tốc độ trên Pi thật

Không có truy cập Pi thật trong phiên phát triển này, nên chưa xác nhận camera,
FPS, nhiệt độ hay bộ nhớ trên Pi. Không thể khẳng định một profile/thread nhanh
nhất cho mọi Pi. Dùng ảnh snapshot trên Pi để so sánh thực tế:

```bash
.venv-pi5/bin/python Python/benchmark_perch_pi5.py --image Python/perch_snapshots/ANH_SEG.png
```

Công cụ thử profile 640/960 và thread 1/2/3/4, ghi median/p95 vào
`models/benchmark_pi5.json`. Đây là latency một mắt gồm xử lý mask, chưa phải FPS
toàn bộ stereo/camera/giao diện. Gói ZIP không kèm hai video lớn; nếu đã chép toàn
dự án có thể dùng `--video Python/l1.mp4 --frame 187` thay cho ảnh.

Kiểm tra pipeline camera không mở cửa sổ, tự trigger sau 10 cặp đã xử lý:

```bash
.venv-pi5/bin/python Python/perch_detector.py --headless --trigger-frame 10 --max-frames 18 --save-last
```

## Kiểm tra đã thực hiện trên Windows

- 32 test đạt, gồm camera giả lập: sequence trùng, bỏ cặp trước T, timeout,
  giải phóng camera và thứ tự màu RGB/BGR.
- Đối chiếu 28 cặp dự đoán (7 vị trí × 2 mắt × 2 profile) trên ảnh 960×540:
  không lệch số lượng detection; mask IoU trung bình 0.999873, thấp nhất 0.998513
  so với PyTorch **cùng kích thước input**. So với Ultralytics NCNN: mask IoU 1.0.
- Đây là độ khớp với model gốc trên video mẫu, **không phải mAP hoặc độ chính xác
  so với nhãn chuẩn**. Thay input từ 960 xuống 640 vẫn có thể đổi mask so với profile
  960; kiểm tra hai profile trên cảnh sử dụng trước khi chọn tốc độ/chất lượng.
- Pipeline NCNN đã chạy hai video, preview → T → 7 cặp → RESULT; log xác nhận
  `torch imported=False`. Depth của thuật toán dùng chung vẫn qua test trùng từng
  pixel với file tham chiếu ở scale 0.5 và 1.0.

Chi tiết: `models/validation_pi5.json`. Mỗi model có `manifest.json` lưu hash của
checkpoint nguồn, kích thước input và hash từng file xuất.

Muốn xuất lại ở môi trường có PyTorch/Ultralytics (không cần làm trên Pi):

```bash
python Python/export_perch_pi5.py
python Python/export_perch_pi5.py --width 960 --height 544
```

Các thư mục output đã tồn tại sẽ bị từ chối để tránh ghi đè model đang dùng.
