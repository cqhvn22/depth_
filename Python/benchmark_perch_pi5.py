"""Benchmark the actual target CPU; no torch or ultralytics required.

python Python/benchmark_perch_pi5.py --image path/to/frame.jpg
or use --video Python/l1.mp4 (default), --frame 187.
Reports network+mask latency, not full stereo-camera FPS.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import cv2
import numpy as np
from perch_ncnn import NCNNSegmenter

ROOT = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", type=Path)
    p.add_argument("--video", type=Path, default=ROOT / "Python/l1.mp4")
    p.add_argument("--frame", type=int, default=187)
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--threads", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--profiles", type=int, nargs="+", choices=[640, 960], default=[640, 960])
    p.add_argument("--fp32-arithmetic", action="store_true")
    p.add_argument("--output", type=Path, default=ROOT / "models/benchmark_pi5.json")
    args = p.parse_args()
    if args.runs < 1 or min(args.threads) < 1:
        p.error("runs and threads must be positive")
    if args.image:
        image = cv2.imread(str(args.image))
    else:
        cap = cv2.VideoCapture(str(args.video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        _, image = cap.read()
        cap.release()
    if image is None:
        p.error("Supply an image/video available on this machine")
    image = cv2.resize(image, (960, 540))
    cv2.setNumThreads(1)
    results = []
    for width in args.profiles:
        for threads in args.threads:
            model = NCNNSegmenter(ROOT / "models" / f"perch_pi5_{width}_fp16_ncnn_model",
                                  threads, fp16_arithmetic=not args.fp32_arithmetic)
            for _ in range(3):
                model.predict(image)
            elapsed = []
            for _ in range(args.runs):
                tick = time.perf_counter()
                model.predict(image)
                elapsed.append((time.perf_counter()-tick)*1000)
            item = dict(profile=width, threads=threads, median_ms=float(np.median(elapsed)),
                        p95_ms=float(np.percentile(elapsed, 95)))
            results.append(item)
            print(json.dumps(item), flush=True)
    fastest = min(results, key=lambda x:x["median_ms"])
    report = dict(host=platform.platform(), machine=platform.machine(), measurements=results,
                  fastest=fastest, note="One-eye model latency only, excluding stereo, capture and UI. "
                  "Smaller inputs may change segmentation; check masks before choosing a profile.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[FASTEST] {fastest}. Written to {args.output}")


if __name__ == "__main__":
    main()
