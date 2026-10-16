# Perch detector từ hai video + YOLOv8n-seg

File chạy: `Python/perch_video_seg.py`. Dựa trên pipeline rectification →
StereoSGBM/WLS → YOLO → đánh giá kích thước/khoảng cách của `perch_detector.py`.
Chạy trên Windows/Linux, không cần Picamera2.

Phần depth dùng trực tiếp `StereoDepth` và `depth_to_color` từ
`Calibration/video_depth_estimation.py`. `SegmentDepth` chỉ bổ sung phép biến đổi
mask; kế thừa nguyên `process()`, không viết lại thuật toán depth. Mặc định
`--downscale 0.5` giống file tham chiếu.

## Chạy ngay trên máy này

Từ thư mục gốc dự án trong PowerShell:

```powershell
.\.venv\Scripts\python.exe Python\perch_video_seg.py
```

Các đường dẫn mặc định được lấy theo vị trí script, không phụ thuộc thư mục làm việc:

| Thành phần | Đường dẫn trong dự án |
|---|---|
| Mắt trái | `Python/l1.mp4` |
| Mắt phải | `Python/r1.mp4` |
| Mô hình đã train | `best-seg.pt` ở thư mục gốc |
| Hiệu chỉnh stereo | `Calibration/stereo_calibration.npz` |
| Snapshot | `Python/perch_video_snapshots/` |

Script kiểm tra `model.task == 'segment'`; không thay bằng model detection hoặc
tự tải trọng số khác. Class lấy từ `best-seg.pt`; nếu mô hình có nhiều class,
dùng `--classes 0` (hoặc ID đúng của thanh đậu).

## Cài trên máy khác

Python 3.10 trở lên, ví dụ:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r Python\requirements-perch-video.txt
```

OpenCV cần bản **contrib** để có `ximgproc`/WLS. Ultralytics có thể kéo theo
`opencv-python`; nếu `cv2.ximgproc` bị thiếu do trùng bản OpenCV, sửa trong venv:

```powershell
.\.venv\Scripts\python.exe -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless
.\.venv\Scripts\python.exe -m pip install opencv-contrib-python
```

Môi trường `.venv` tạo trong phiên này dùng PyTorch CPU. Máy có CUDA cần cài
PyTorch tương ứng trước khi dùng `--device 0`.

## Điều khiển và xuất kết quả

| Phím | Chức năng |
|---|---|
| T | Thu **7 cặp frame tiếp theo**, đo và đánh giá thanh đang chọn, rồi đóng băng kết quả |
| R | Xóa kết quả/burst hiện tại, quay về live ở vị trí video đang dừng |
| Space | Tạm dừng/tiếp tục live hoặc burst; từ RESULT trở lại live |
| B | Chạy lại cả hai video từ đầu, xóa tracking và kết quả |
| D | Bật/tắt ô depth; hai mắt luôn hiển thị |
| S | Lưu toàn màn hình 4 ô, ảnh mắt trái/phải, depth màu/mét `.npy`, mask trái và kết quả JSON |
| Q / Esc | Thoát |

Giao diện 2×2: **trái/phải** ở hàng trên (mỗi mắt có segmentation riêng),
**depth/hướng dẫn và kết quả** ở hàng dưới. Hai mắt đã rectify để quan sát đồng
thời; số đo dùng mask trái và depth trái. Nhãn `R1`, `R2` ở mắt phải chỉ đánh số
phát hiện của mắt phải, không phải ID tracking ghép với mắt trái.

Quy trình giống chế độ trigger của `perch_detector.py`:

1. LIVE: xem segmentation và gợi ý căn góc; chưa đo kích thước/kết luận đáp.
2. T: chuyển sang CAPTURING, dùng 7 cặp frame **sau khi bấm T**. Chỉ lấy số đo
   của ID thanh được chọn lúc bấm, không trộn nhiều thanh hoặc dữ liệu trước T.
3. RESULT: lấy trung vị khoảng cách/dài/rộng, áp dụng ngưỡng của file gốc và giữ
   kết quả cùng hai ảnh cuối burst. Cần ít nhất 3 frame có depth hợp lệ. Nếu burst
   bị ngắt/hết video hoặc mất ID ở frame cuối thì trả INSUFFICIENT DATA.
4. R để về LIVE; T từ RESULT để đo một burst mới từ vị trí video hiện tại.

Thanh có ID `TARGET` được chọn theo mask lớn/confidence cao, ưu tiên giữ ID đó
khi còn thấy. Nếu chưa có thanh ở thời điểm T, burst chọn thanh đầu tiên xuất hiện.

### Gợi ý căn góc trước khi đo

Đường xanh là trục dài của mask, nét đứt trắng là phương ngang, dấu cộng là tâm
ảnh. Bảng hướng dẫn hiển thị độ nghiêng trong ảnh, hướng xoay **ảnh nhìn** CW/CCW
để đưa thanh về ngang, hướng đưa tâm quan sát tới thanh (LEFT/RIGHT/UP/DOWN), và
nhắc lùi/căn lại nếu thanh bị cắt ở mép. Mặc định sai số góc cho phép ±5°, sai số
tâm ±12% chiều rộng/cao; chỉnh bằng `--align-angle` và `--align-center`.

Đây là gợi ý **2D trong ảnh**, không phải ước lượng đầy đủ roll/pitch/yaw 3D hay
lệnh điều khiển drone. Gợi ý không chặn phím T hoặc thay đổi ngưỡng đánh giá gốc.
Kết quả CAN LAND chỉ có nghĩa đạt điều kiện kích thước/khoảng cách.

Màu xanh: trong ngưỡng kích thước/khoảng cách; đỏ: ngoài ngưỡng; vàng: chưa đủ
depth hoặc lịch sử. Đây là bước sàng lọc hình học, chưa đánh giá độ bền, vật cản,
góc tiếp cận hay khả năng điều khiển drone.

Chạy không mở cửa sổ, xuất video 4 ô (chỉ live, chưa bấm T):

```powershell
.\.venv\Scripts\python.exe Python\perch_video_seg.py --headless --output Python\perch_result.mp4
```

Kiểm tra tự động thao tác T sau cặp frame 0, thu 7 cặp tiếp và lưu kết quả:

```powershell
.\.venv\Scripts\python.exe Python\perch_video_seg.py --headless --trigger-frame 0 --max-frames 8 --save-last
```

Video xuất giữ FPS nguồn và mỗi cặp đầu vào tạo một frame đầu ra. Tốc độ xử lý
CPU có thể chậm hơn thời gian thực; script không bỏ frame để chạy đuổi. Mặc định
dừng khi một trong hai video hết. `--loop` tua lại đồng thời cả hai video;
headless có loop phải kèm `--max-frames`.
Headless có `--trigger-frame` sẽ dừng khi burst hoàn tất; frame trigger được tính
từ 0 tương đối với `--left-start`/`--right-start`. `--burst-frames` chỉnh độ dài
burst, `--min-samples` chỉnh số cặp đo hợp lệ tối thiểu.

## Các cải thiện thuật toán

- Dùng trực tiếp bitmap từng instance từ `masks.data`, bật `retina_masks=True`
  và `imgsz=960` (đúng kích thước train lưu trong checkpoint). Mặc định dự đoán
  trên **ảnh video gốc** để tránh hình dạng bị biến đổi/cắt khi rectify, sau đó
  resize/remap mask bằng chính map của từng mắt. Mask sau biến đổi cùng hệ tọa độ
  với ảnh đã rectify và depth. Có thể dùng `--seg-space rectified` để so sánh với
  cách cũ. Không lấy một polygon rồi tô lại làm mất lỗ hoặc nối các phần rời nhau.
- Loại các đảo mask nhỏ, giữ lỗ và các thành phần lớn. Không làm nở mask hoặc
  lấp toàn bộ bounding box để tránh kéo nền vào vật thể.
- Chỉ lấy depth trong lòng mask; loại biên bằng distance transform thích ứng
  với thanh mảnh, lọc nhiễu bằng median/MAD. Không dùng depth nền trong box.
- Depth giữ nguyên file tham chiếu: SGBM block 7; 96 mức disparity khi downscale
  nhỏ hơn 1, 128 mức khi downscale bằng 1; speckle window 100; WLS lambda 8000,
  sigmaColor 1.5; nội suy disparity bằng INTER_LINEAR và reproject bằng Q.
  Pixel không hợp lệ có depth bằng 0. Không thêm padding, tự tăng range hoặc lọc
  confidence/vùng remap ngoài những gì `StereoDepth.process()` gốc thực hiện.
- Hiển thị depth cũng dùng `depth_to_color()` gốc với colormap Plasma. Chỉ chuyển
  một bản sao depth sang mét để đo vật thể và lưu snapshot; không sửa depth gốc.
  Yêu cầu ít nhất 20 mẫu và 15% lõi mask áp dụng cho số đo/vật thể, không xóa
  pixel trên bản đồ depth.
- Đo chiều dài/rộng bằng `minAreaRect` theo hướng thanh, tránh dùng box trục X/Y
  khiến thanh nghiêng bị đo rộng quá mức. Kích thước là **kích thước chiếu trong
  mặt phẳng ảnh ở depth đại diện**, chưa bù thanh nghiêng ra/vào camera.
- Ghép vật thể cùng class bằng IoU của mask, rồi median trong burst của một ID.
  Không gộp số đo nhiều thanh. Không đủ mẫu depth hoặc mất target thì UNKNOWN.
  Ghép IoU có thể đổi ID khi chuyển động nhanh, che khuất hoặc các thanh giao nhau.
- Giữ ngưỡng từ file gốc: khoảng cách 0.20–5.00 m; dài 0.10–2.00 m;
  rộng 0.02–0.20 m. Hàm `assess()` là nơi chỉnh các ngưỡng này.

Tham số dự đoán mask tham khảo tài liệu chính thức:
[Ultralytics Predict](https://docs.ultralytics.com/modes/predict/#inference-arguments).

## Chỉnh chất lượng/tốc độ và đồng bộ

```powershell
# Segmentation nhẹ hơn; depth vẫn dùng mặc định 0.5 của file tham chiếu
.\.venv\Scripts\python.exe Python\perch_video_seg.py --imgsz 640

# Stereo full resolution, đúng chế độ downscale=1.0 của file tham chiếu
.\.venv\Scripts\python.exe Python\perch_video_seg.py --downscale 1.0

# Nếu có GPU CUDA và PyTorch tương thích
.\.venv\Scripts\python.exe Python\perch_video_seg.py --device 0

# Bù lệch lúc bắt đầu: dùng frame 0 trái với frame 2 phải
.\.venv\Scripts\python.exe Python\perch_video_seg.py --right-start 2
```

Chỉ bù offset khi đã xác định độ lệch thực tế. Ghép cùng frame index không sửa
được video bị rớt frame, FPS biến thiên hoặc hai camera lệch thời gian lúc ghi.
Hai video phải cùng kích thước và FPS cố định; script từ chối FPS khác nhau.

File calibration có đơn vị mm; mặc định `--unit mm`. Nếu thay calibration bằng
đơn vị khác phải đổi `--unit cm` hoặc `--unit m` tương ứng. Stereo được resize
về kích thước calibration trước remap, không dùng Q sai tỷ lệ. Điều này chỉ đúng
khi video giữ nguyên trường nhìn của ảnh calibration. Hành vi resize/cảnh báo
khi khác tỷ lệ khung hình được giữ nguyên theo file tham chiếu. Calibration sai
không thể sửa bằng segmentation.

`--conf` điều chỉnh ngưỡng phát hiện, `--imgsz` điều chỉnh chi tiết đầu vào;
`--min-depth-pixels`, `--min-depth-fraction` điều chỉnh yêu cầu của số đo trong mask.
Không nên hạ ngưỡng depth chỉ để ép ra kết luận xanh. Chất lượng segmentation
thực tế cần đánh giá bằng nhãn chuẩn trên các frame đại diện; các thay đổi suy luận
và hậu xử lý không bảo đảm mọi mask đều đúng.

Các tùy chọn cũ `--num-disparities`, `--wls-conf`, `--allow-resize` đã bỏ để giữ
thuật toán depth giống file tham chiếu. Các ảnh/chẩn đoán cũ trong
`perch_video_check/` ghi lại phiên bản thử nghiệm trước đây; không đại diện cho
depth hiện tại.

## Kiểm tra hồi quy

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s Python -p test_perch_video_seg.py -v
```

Các ca kiểm tra: loại depth nền, thanh nghiêng/mảnh, depth nhiễu hoặc thiếu,
giữ lỗ mask, tách lịch sử hai vật thể, không dùng verdict cũ khi mất depth,
dừng/tua lại hai video đồng thời và từ chối FPS khác nhau; preview không gọi phép
đo vật lý, gợi ý góc/tâm, T thu đúng 7 mẫu mới, đóng băng/reset, burst bị ngắt và
không đổi target giữa burst, cùng bố cục 4 ô.
Kiểm tra đối chiếu trực tiếp ảnh rectify, disparity và depth trên các frame
0, 187, 956 ở cả scale 0.5/1.0 với `StereoDepth` gốc; yêu cầu từng pixel bằng nhau.
