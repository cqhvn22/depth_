# Run YOLOv8 + Stereo Depth Estimation on recorded videos
# Usage: .\run_yolo_depth.ps1
# From: D:\Lenna-Stereo-Camera

$PYTHON = "C:/Users/ADMIN/AppData/Local/Programs/Python/Python312/python.exe"
$SCRIPT = "Calibration/video_yolo_depth.py"
$LEFT   = "Calibration/recordings/left_20260930_164731.mp4"
$RIGHT  = "Calibration/recordings/right_20260930_164731.mp4"
$CALIB  = "stereo_calibration.npz"

& $PYTHON $SCRIPT --left $LEFT --right $RIGHT --calib $CALIB
