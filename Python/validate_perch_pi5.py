"""Off-device export validation against PyTorch/Ultralytics, not an accuracy dataset.

Run in the export environment: python Python/validate_perch_pi5.py
Writes sampled mask agreement and sizes to models/validation_pi5.json.
"""
import json
import os
from pathlib import Path
import platform

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))

import cv2
import numpy as np
from perch_ncnn import NCNNSegmenter


def agreement(a, b):
    boxes_a = a.boxes.data.numpy()
    boxes_b = b.boxes.data.cpu().numpy()
    report = dict(count_native=len(boxes_a), count_reference=len(boxes_b), ious=[])
    if len(boxes_a) != len(boxes_b):
        return report
    for i in range(len(boxes_a)):
        ma = a.masks.data.numpy()[i] > 0
        mb = b.masks.data.cpu().numpy()[i] > 0
        report["ious"].append(float(np.count_nonzero(ma & mb) / max(1, np.count_nonzero(ma | mb))))
    return report


def main():
    from ultralytics import YOLO
    import torch
    torch.set_num_threads(3)
    cv2.setNumThreads(1)
    pt = YOLO(str(ROOT / "best-seg.pt"))
    records = []
    for width, height in ((640, 384), (960, 544)):
        directory = ROOT / "models" / f"perch_pi5_{width}_fp16_ncnn_model"
        native = NCNNSegmenter(directory, threads=3)
        ultra = YOLO(str(directory), task="segment")
        for eye in ("l", "r"):
            cap = cv2.VideoCapture(str(ROOT / "Python" / f"{eye}1.mp4"))
            try:
                for index in (0, 187, 350, 650, 956, 1200, 1400):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                    ok, image = cap.read()
                    if not ok:
                        raise RuntimeError(f"Could not decode {eye}:{index}")
                    # Same 960x540 input as default live cameras on the Pi.
                    image = cv2.resize(image, (960, 540), interpolation=cv2.INTER_LINEAR)
                    result = native.predict(image)[0]
                    reference = pt.predict(image, imgsz=[height, width], rect=False, conf=.35,
                                           iou=.5, retina_masks=True, verbose=False)[0]
                    wrapped = ultra.predict(image, conf=.35, iou=.5, retina_masks=True, verbose=False)[0]
                    records.append(dict(profile=width, eye=eye, frame=index,
                                        vs_pytorch=agreement(result, reference),
                                        vs_ultralytics_ncnn=agreement(result, wrapped)))
            finally:
                cap.release()
    sizes = {p.name:sum(x.stat().st_size for x in p.iterdir() if x.is_file())
             for p in (ROOT / "models").glob("*_ncnn_model")}
    summary = {}
    for key in ("vs_pytorch", "vs_ultralytics_ncnn"):
        ious = [iou for item in records for iou in item[key]["ious"]]
        summary[key] = dict(count_mismatches=sum(r[key]["count_native"] != r[key]["count_reference"] for r in records),
                            matched_masks=len(ious), minimum_iou=min(ious) if ious else None,
                            mean_iou=float(np.mean(ious)) if ious else None)
    report = dict(host=platform.platform(), pi_hardware_tested=False, samples=len(records),
                  note="Mask agreement on provided video samples, not mAP/IoU against ground-truth labels.",
                  summary=summary, model_bytes=sizes, frames=records)
    output = ROOT / "models/validation_pi5.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(dict(summary=summary, output=str(output)), indent=2), flush=True)
    if summary["vs_ultralytics_ncnn"]["count_mismatches"] or summary["vs_ultralytics_ncnn"]["minimum_iou"] < .99:
        raise RuntimeError("Lightweight postprocessing does not match Ultralytics closely enough")


if __name__ == "__main__":
    main()
