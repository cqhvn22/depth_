#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
if [[ "$(uname -m)" != "aarch64" && "$(uname -m)" != "arm64" ]]; then
  echo "Use 64-bit Raspberry Pi OS on the Pi 5 (uname -m: aarch64)." >&2
  exit 1
fi
sudo apt-get update
sudo apt-get install -y python3-venv python3-picamera2 python3-opencv python3-numpy python3-yaml libgomp1 libvulkan1
python3 -m venv --system-site-packages .venv-pi5
.venv-pi5/bin/python -m pip install --no-deps -r Python/requirements-perch-pi5.txt
.venv-pi5/bin/python -c 'import cv2, ncnn, numpy, yaml; from picamera2 import Picamera2; assert hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "createRightMatcher"), "OpenCV contrib/WLS is missing"; print("Ready: OpenCV", cv2.__version__, "NCNN", ncnn.__version__)'
echo "Run: .venv-pi5/bin/python Python/perch_detector.py"
