"""YOLOv8 instance segmentation on NCNN + NumPy/OpenCV, without PyTorch.

Pre/postprocessing follows Ultralytics retina_masks: letterbox, class-aware
NMS, native-resolution prototype masks, and native-coordinate box cropping.
The small result adapter is consumed by perch_video_seg.infer().
"""
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
import time

import cv2
import numpy as np
import yaml


class ArrayTensor:
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def cpu(self):
        return self
    def numpy(self):
        return self.data


class ArrayBoxes:
    def __init__(self, data):
        self.data = ArrayTensor(data)
    def __len__(self):
        return len(self.data)
    def cpu(self):
        return self


def letterbox(image, shape):
    h, w = image.shape[:2]
    gain = min(shape[0] / h, shape[1] / w)
    resized = (round(w * gain), round(h * gain))
    dw, dh = (shape[1] - resized[0]) / 2, (shape[0] - resized[1]) / 2
    left, top = round(dw - 0.1), round(dh - 0.1)
    if resized != (w, h):
        image = cv2.resize(image, resized, interpolation=cv2.INTER_LINEAR)
    image = cv2.copyMakeBorder(image, top, round(dh + 0.1), left, round(dw + 0.1),
                               cv2.BORDER_CONSTANT, value=(114, 114, 114))
    tensor = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32) / 255.0
    return tensor, (resized[0] / w, resized[1] / h), (left, top)


def class_nms(boxes, scores, classes, iou_threshold=0.5, max_det=300):
    order = np.argsort(-scores, kind="stable")
    keep = []
    areas = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    while len(order) and len(keep) < max_det:
        selected = int(order[0])
        keep.append(selected)
        rest = order[1:]
        if not len(rest):
            break
        lo = np.maximum(boxes[selected, :2], boxes[rest, :2])
        hi = np.minimum(boxes[selected, 2:], boxes[rest, 2:])
        intersection = np.maximum(0, hi - lo).prod(axis=1)
        overlap = intersection / np.maximum(areas[selected] + areas[rest] - intersection, 1e-7)
        order = rest[(classes[rest] != classes[selected]) | (overlap <= iou_threshold)]
    return np.asarray(keep, np.int32)


def native_masks(protos, coefficients, boxes, shape):
    """Match native/retina mask scaling, including odd letterbox padding."""
    channels, mh, mw = protos.shape
    h, w = shape
    logits = (coefficients @ protos.reshape(channels, -1)).reshape(-1, mh, mw)
    gain = min(4 * mh / h, 4 * mw / w)
    padding = (4 * mh - round(h * gain)) | (4 * mw - round(w * gain))
    factor = 4 // math.gcd(4, padding | padding // 2)
    output = np.empty((len(boxes), h, w), np.uint8)
    xs, ys = np.arange(w)[None, :], np.arange(h)[:, None]
    for i, (logit, box) in enumerate(zip(logits, boxes)):
        if factor > 1:
            logit = cv2.resize(logit, (mw * factor, mh * factor), interpolation=cv2.INTER_LINEAR)
        sh, sw = logit.shape
        scale = min(sh / h, sw / w)
        content_h, content_w = round(h * scale), round(w * scale)
        top = round((sh - content_h) / 2 - 0.1)
        left = round((sw - content_w) / 2 - 0.1)
        content = logit[top:top + content_h, left:left + content_w]
        native = cv2.resize(content, (w, h), interpolation=cv2.INTER_LINEAR) > 0
        native &= (xs >= box[0]) & (xs < box[2]) & (ys >= box[1]) & (ys < box[3])
        output[i] = native
    return output


class NCNNSegmenter:
    """Fixed-shape batch-1 NCNN YOLOv8-seg, no downloads or torch imports."""
    task = "segment"

    def __init__(self, model_dir, threads=3, fp16_arithmetic=True):
        directory = Path(model_dir)
        for name in ("model.ncnn.param", "model.ncnn.bin", "metadata.yaml"):
            if not (directory / name).is_file():
                raise RuntimeError(f"Missing NCNN model component: {directory / name}")
        metadata = yaml.safe_load((directory / "metadata.yaml").read_text(encoding="utf-8"))
        if metadata.get("task") != "segment" or metadata.get("batch") != 1:
            raise ValueError("Use a batch=1 instance segmentation NCNN export")
        names = metadata["names"]
        self.names = {int(k): v for k, v in names.items()} if isinstance(names, dict) else dict(enumerate(names))
        if set(self.names) != set(range(len(self.names))):
            raise ValueError("Model class IDs must be contiguous from zero")
        self.imgsz = tuple(int(x) for x in metadata["imgsz"])
        if len(self.imgsz) != 2 or min(self.imgsz) < 32:
            raise ValueError("Invalid NCNN input shape")
        try:
            import ncnn
        except ImportError as exc:
            raise RuntimeError("Install NCNN: python -m pip install -r Python/requirements-perch-pi5.txt") from exc
        self.ncnn = ncnn
        self.net = ncnn.Net()
        self.net.opt.num_threads = threads
        self.net.opt.use_vulkan_compute = False
        self.net.opt.use_fp16_packed = True
        self.net.opt.use_fp16_storage = True
        self.net.opt.use_fp16_arithmetic = fp16_arithmetic
        self.net.opt.use_packing_layout = True
        if self.net.load_param(str(directory / "model.ncnn.param")) != 0:
            raise RuntimeError("NCNN could not load network parameters")
        if self.net.load_model(str(directory / "model.ncnn.bin")) != 0:
            raise RuntimeError("NCNN could not load model weights")
        self.input_name = self.net.input_names()[0]
        self.output_names = sorted(self.net.output_names())
        self.speed = {}

    def predict(self, source, conf=0.35, iou=0.5, retina_masks=True, classes=None,
                device="cpu", imgsz=None, verbose=False):
        if device != "cpu" or not retina_masks:
            raise ValueError("NCNN Pi runner requires device=cpu and retina_masks=True")
        if source.ndim != 3 or source.shape[2] != 3 or source.dtype != np.uint8:
            raise ValueError("Expected a uint8 OpenCV BGR image")
        started = time.perf_counter()
        tensor, gain, pad = letterbox(source, self.imgsz)
        preprocessed = time.perf_counter()
        with self.net.create_extractor() as extractor:
            if extractor.input(self.input_name, self.ncnn.Mat(tensor)) != 0:
                raise RuntimeError("NCNN input failed")
            outputs = []
            for name in self.output_names:
                status, output = extractor.extract(name)
                if status != 0:
                    raise RuntimeError(f"NCNN output failed: {name}")
                outputs.append(np.array(output, dtype=np.float32))
        inferred = time.perf_counter()
        if len(outputs) != 2:
            raise ValueError("Expected YOLOv8-seg detection and prototype outputs")
        prediction = next((x for x in outputs if x.ndim == 2), None)
        protos = next((x for x in outputs if x.ndim == 3), None)
        if prediction is None or protos is None:
            raise ValueError(f"Unexpected NCNN output shapes: {[x.shape for x in outputs]}")
        nc, nm = len(self.names), protos.shape[0]
        if prediction.shape[0] != 4 + nc + nm:
            raise ValueError("This adapter supports the exported YOLOv8 segmentation head only")
        rows = prediction.T
        class_ids = rows[:, 4:4 + nc].argmax(axis=1)
        scores = rows[np.arange(len(rows)), 4 + class_ids]
        valid = np.isfinite(rows).all(axis=1) & (scores > conf)
        if classes is not None:
            valid &= np.isin(class_ids, classes)
        rows, scores, class_ids = rows[valid], scores[valid], class_ids[valid]
        boxes = np.concatenate((rows[:, :2] - rows[:, 2:4] / 2,
                                rows[:, :2] + rows[:, 2:4] / 2), axis=1)
        keep = class_nms(boxes, scores, class_ids, iou)
        boxes, scores, class_ids, coefficients = boxes[keep], scores[keep], class_ids[keep], rows[keep, 4 + nc:]
        boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - pad[0]) / gain[0], 0, source.shape[1])
        boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - pad[1]) / gain[1], 0, source.shape[0])
        masks = native_masks(protos, coefficients, boxes, source.shape[:2]) if len(boxes) else None
        if masks is not None:
            nonempty = masks.any(axis=(1, 2))
            boxes, scores, class_ids, masks = boxes[nonempty], scores[nonempty], class_ids[nonempty], masks[nonempty]
        data = np.column_stack((boxes, scores, class_ids)).astype(np.float32)
        self.speed = dict(preprocess=(preprocessed-started)*1000,
                          inference=(inferred-preprocessed)*1000,
                          postprocess=(time.perf_counter()-inferred)*1000)
        return [SimpleNamespace(boxes=ArrayBoxes(data),
                                masks=SimpleNamespace(data=ArrayTensor(masks)) if masks is not None and len(masks) else None,
                                speed=self.speed)]
