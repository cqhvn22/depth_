"""Perch detection from l1.mp4/r1.mp4 using the trained YOLOv8n-seg model.

Run without arguments: python Python/perch_video_seg.py
See perch_video_seg_README.md for controls, tuning and measurement assumptions.
No Picamera2 dependency; all default paths are relative to this file.
Both eyes + depth + 2D alignment guidance. T captures a 7-pair measurement burst;
R returns to live, SPACE pauses, B restarts, S saves, D toggles depth, Q quits.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import sys
import textwrap
import time

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "Calibration"))
from video_depth_estimation import StereoDepth, depth_to_color

UNIT_TO_M = {"mm": 0.001, "cm": 0.01, "m": 1.0}
OK, FAIL, UNKNOWN = (40, 220, 60), (40, 60, 240), (0, 200, 255)


class VideoPair:
    """Read matching frame indices; stop at the shorter stream, never loop one eye."""

    def __init__(self, left: Path, right: Path, starts: tuple[int, int]):
        self.left = cv2.VideoCapture(str(left))
        self.right = cv2.VideoCapture(str(right))
        self.starts = starts
        try:
            for cap, path in ((self.left, left), (self.right, right)):
                if not cap.isOpened():
                    raise RuntimeError(f"Cannot open video: {path}")
            rates = [cap.get(cv2.CAP_PROP_FPS) for cap in (self.left, self.right)]
            if not all(np.isfinite(f) and f > 0 for f in rates):
                raise RuntimeError("Video FPS is missing; re-encode both videos at a fixed FPS.")
            if abs(rates[0] - rates[1]) > max(rates) * 0.001:
                raise RuntimeError(f"Different FPS {rates}. Re-encode to the same constant FPS first.")
            self.fps = rates[0]
            counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                      for cap in (self.left, self.right)]
            if any(count <= start for count, start in zip(counts, starts)):
                raise RuntimeError("Start frame is outside the video or frame count is unavailable.")
            self.total = min(count - start for count, start in zip(counts, starts))
            sizes = [(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                     for cap in (self.left, self.right)]
            if sizes[0] != sizes[1]:
                raise RuntimeError(f"Left/right video sizes differ: {sizes}")
            self.size = sizes[0]
            self.rewind()
            print(f"[VIDEO] {self.size}, {self.fps:.3f} FPS, {self.total} paired frames")
        except Exception:
            self.release()
            raise

    def rewind(self):
        for cap, start in zip((self.left, self.right), self.starts):
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, start):
                raise RuntimeError(f"Cannot seek video to frame {start}")
        self.index = 0

    def read(self):
        if self.index >= self.total:
            return None
        ok_l, left = self.left.read()
        ok_r, right = self.right.read()
        if not ok_l or not ok_r:
            return None
        index = self.index
        self.index += 1
        return index, left, right

    def release(self):
        self.left.release()
        self.right.release()


class SegmentDepth(StereoDepth):
    """Use the reference StereoDepth unchanged; add only mask projection.

    process() is inherited directly, returning rectified left, disparity and
    depth in calibration units, with the reference's zero-valued invalid pixels.
    """

    def __init__(self, args, video_size=None):
        super().__init__(str(args.calib), downscale=args.downscale)
        self.size = (self.left_map_x.shape[1], self.left_map_x.shape[0])
        self.maps = [self.left_map_x, self.left_map_y, self.right_map_x, self.right_map_y]

    def rectify_mask(self, mask, eye="left"):
        """Project a mask into its eye's rectified image; depth stays unchanged."""
        resized = cv2.resize(mask.astype(np.uint8), self.size, interpolation=cv2.INTER_NEAREST)
        offset = 0 if eye == "left" else 2
        rectified = cv2.remap(resized, self.maps[offset], self.maps[offset + 1], cv2.INTER_NEAREST)
        return rectified

    def rectify_right(self, right):
        """Right-eye preview only, using the reference's resize/remap operations."""
        if (right.shape[1], right.shape[0]) != self.size:
            right = cv2.resize(right, self.size, interpolation=cv2.INTER_LINEAR)
        return cv2.remap(right, self.right_map_x, self.right_map_y, cv2.INTER_LINEAR)


def clean_mask(mask: np.ndarray, min_area: int) -> np.ndarray | None:
    """Remove tiny disconnected islands; keep holes and large occluded pieces.

    No convex hull, dilation or polygon fill: these would invent foreground.
    """
    binary = (mask > 0.5).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    threshold = max(min_area, int(areas.max() * 0.01))
    keep = np.flatnonzero(areas >= threshold) + 1
    if not len(keep):
        return None
    return np.isin(labels, keep).astype(np.uint8)


@dataclass
class Measurement:
    distance: float | None = None
    length: float | None = None
    width: float | None = None
    valid_fraction: float = 0.0
    reason: str = "No reliable depth in mask"

    def values(self):
        if self.distance is None:
            return None
        return self.distance, self.length, self.width


def measure_mask(mask, depth_m, focal, min_pixels=20, min_fraction=0.15):
    """Estimate projected dimensions from an oriented mask, with robust masked depth.

    Border erosion only selects depth samples; it never changes the displayed
    mask or its measured outline. Dimensions assume one representative Z plane.
    """
    result = Measurement()
    distance_to_edge = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    margin = min(2.0, float(distance_to_edge.max()) * 0.25)
    core = (mask > 0) & (distance_to_edge > margin)
    valid = core & np.isfinite(depth_m) & (depth_m > 0)
    result.valid_fraction = float(valid.sum() / max(1, core.sum()))
    samples = depth_m[valid]
    if samples.size < min_pixels or result.valid_fraction < min_fraction:
        return result, None
    median = float(np.median(samples))
    mad = float(np.median(np.abs(samples - median)))
    # MAD alone can be zero for a split between two flat depth planes.
    q25, q75 = np.percentile(samples, (25, 75))
    if (q75 - q25) / median > 0.40 or 1.4826 * mad / median > 0.20:
        result.reason = "Depth spread too large"
        return result, None
    inliers = samples[np.abs(samples - median) <= max(0.005, 3 * 1.4826 * mad)]
    if inliers.size < min_pixels or inliers.size / max(1, core.sum()) < min_fraction:
        result.reason = "Too few depth inliers"
        return result, None
    z = float(np.median(inliers))
    points = cv2.findNonZero(mask)
    rect = cv2.minAreaRect(points)
    length_px, width_px = sorted(rect[1], reverse=True)
    if width_px < 1 or focal <= 0:
        result.reason = "Mask too thin to measure"
        return result, None
    result.distance = z
    result.length = length_px * z / focal
    result.width = width_px * z / focal
    result.reason = "Measured"
    return result, cv2.boxPoints(rect)


def assess(values):
    """Same distance/length/width thresholds as perch_detector.py, in metres."""
    if values is None:
        return None, "UNKNOWN: insufficient depth/history"
    distance, length, width = values
    reasons = []
    if not 0.20 <= distance <= 5.00:
        reasons.append("distance outside 0.20-5.00m")
    if not 0.10 <= length <= 2.00:
        reasons.append("length outside 0.10-2.00m")
    if not 0.02 <= width <= 0.20:
        reasons.append("width outside 0.02-0.20m")
    return (False, "; ".join(reasons)) if reasons else (True, "WITHIN SIZE/DIST LIMITS")


@dataclass
class Track:
    identity: int
    class_id: int
    mask: np.ndarray
    last_frame: int
    history: deque = field(default_factory=lambda: deque(maxlen=7))


class MaskTracks:
    """Conservative same-class mask IoU association; never pool distinct objects."""

    def __init__(self, history=7, min_samples=3):
        self.tracks: list[Track] = []
        self.next_id = 1
        self.history, self.min_samples = history, min_samples

    def reset_measurements(self):
        for track in self.tracks:
            track.history.clear()

    def update(self, detections, frame_index):
        self.tracks = [t for t in self.tracks if frame_index - t.last_frame <= 3]
        candidates = []
        for i, det in enumerate(detections):
            for j, track in enumerate(self.tracks):
                if det["class_id"] != track.class_id:
                    continue
                intersection = np.count_nonzero((det["mask"] > 0) & (track.mask > 0))
                union = np.count_nonzero((det["mask"] > 0) | (track.mask > 0))
                iou = intersection / max(1, union)
                if iou >= 0.25:
                    candidates.append((iou, i, j))
        matched, used_tracks = {}, set()
        for _, i, j in sorted(candidates, reverse=True):
            if i not in matched and j not in used_tracks:
                matched[i] = self.tracks[j]
                used_tracks.add(j)
        for i, det in enumerate(detections):
            track = matched.get(i)
            if track is None:
                track = Track(self.next_id, det["class_id"], det["mask"], frame_index,
                              deque(maxlen=self.history))
                self.next_id += 1
                self.tracks.append(track)
            track.mask, track.last_frame = det["mask"], frame_index
            values = det["measurement"].values()
            track.history.append((frame_index, values))
            recent = [v for frame, v in track.history
                      if frame_index - frame < self.history and v is not None]
            # A previous measurement must never hide a failed current measurement.
            stable = (tuple(float(x) for x in np.median(recent, axis=0))
                      if values is not None and len(recent) >= self.min_samples else None)
            det.update(track_id=track.identity, stable=stable, samples=len(recent))
            det["verdict"], det["reason"] = assess(stable)
            if values is None:
                det["reason"] = det["measurement"].reason


def infer(model, image, depth, focal, args, mask_projector=None):
    prediction = model.predict(
        source=image, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
        retina_masks=True, device=args.device, classes=args.classes,
        verbose=False)[0]
    boxes, masks = prediction.boxes, prediction.masks
    if boxes is None or len(boxes) == 0:
        return []
    if masks is None or len(masks.data) != len(boxes):
        raise RuntimeError("Detections have no matching segmentation masks; use best-seg.pt.")
    # retina_masks guarantees native coordinates; refuse a silently misaligned mask.
    raw_masks = masks.data.cpu().numpy()
    if tuple(raw_masks.shape[1:]) != image.shape[:2]:
        raise RuntimeError("Mask/image size mismatch; update Ultralytics (retina_masks required).")
    detections = []
    for box, raw in zip(boxes.cpu().data.numpy(), raw_masks):
        if mask_projector is not None:
            raw = mask_projector(raw)
        if depth is not None and raw.shape != depth.shape:
            raise RuntimeError("Projected mask and depth coordinates do not match.")
        mask = clean_mask(raw, args.min_mask_area)
        if mask is None:
            continue
        class_id = int(box[5])
        if depth is None:
            measurement = Measurement(reason="Press T to measure")
            oriented = cv2.boxPoints(cv2.minAreaRect(cv2.findNonZero(mask)))
        else:
            measurement, _ = measure_mask(
                mask, depth, focal, args.min_depth_pixels, args.min_depth_fraction)
            oriented = cv2.boxPoints(cv2.minAreaRect(cv2.findNonZero(mask)))
        x, y, w, h = cv2.boundingRect(cv2.findNonZero(mask))
        detections.append(dict(
            mask=mask, class_id=class_id, name=model.names[class_id], confidence=float(box[4]),
            box=(x, y, x + w, y + h), measurement=measurement, oriented=oriented))
    return detections


def text(image, value, position, color=(240, 240, 240), scale=0.5):
    cv2.putText(image, value, position, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, value, position, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, 1, cv2.LINE_AA)


def select_target(detections, preferred_id=None):
    for det in detections:
        if det["track_id"] == preferred_id:
            return det
    return max(detections, key=lambda d: np.count_nonzero(d["mask"]) * d["confidence"],
               default=None)


def alignment_hint(mask, angle_tolerance=5.0, center_tolerance=0.12):
    """Image-plane roll/centering only; no depth, IMU or inferred 3D attitude."""
    if mask is None or not np.any(mask):
        return dict(angle_deg=None, offset=None, ready=False, axis=None,
                    messages=["Find a perch visible in both eyes."])
    rect = cv2.minAreaRect(cv2.findNonZero(mask))
    corners = cv2.boxPoints(rect)
    edges = np.roll(corners, -1, axis=0) - corners
    vector = edges[int(np.argmax(np.linalg.norm(edges, axis=1)))]
    if vector[0] < 0:
        vector = -vector
    major, minor = sorted(rect[1], reverse=True)
    angle = float(np.degrees(np.arctan2(vector[1], vector[0]))) if major >= 1.5 * max(minor, 1) else None
    moments = cv2.moments(mask)
    center = np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]])
    h, w = mask.shape
    dx, dy = float((center[0] - w / 2) / w), float((center[1] - h / 2) / h)
    clipped = bool(np.any(mask[0]) or np.any(mask[-1]) or np.any(mask[:, 0]) or np.any(mask[:, -1]))
    messages = []
    if angle is None:
        messages.append("Bar axis unclear: change viewpoint to see its length.")
    elif abs(angle) > angle_tolerance:
        direction = "CCW" if angle > 0 else "CW"
        messages.append(f"Level bar: image rotation {direction} by {abs(angle):.1f} deg.")
    else:
        messages.append(f"Roll aligned (within +/-{angle_tolerance:g} deg).")
    if abs(dx) > center_tolerance or abs(dy) > center_tolerance:
        directions = []
        if abs(dx) > center_tolerance:
            directions.append("RIGHT" if dx > 0 else "LEFT")
        if abs(dy) > center_tolerance:
            directions.append("DOWN" if dy > 0 else "UP")
        messages.append("Aim toward bar: " + " + ".join(directions) + ".")
    else:
        messages.append("Bar centered in view.")
    if clipped:
        messages.append("Move back / reframe: part of the bar is outside the image.")
    axis = np.stack((np.array(rect[0]) - vector / 2, np.array(rect[0]) + vector / 2))
    ready = angle is not None and abs(angle) <= angle_tolerance and abs(dx) <= center_tolerance and abs(dy) <= center_tolerance and not clipped
    return dict(angle_deg=angle, offset=(dx, dy), ready=bool(ready), axis=axis.tolist(), messages=messages)


class PerchBurst:
    """T-triggered N-frame median for one target, with no pre-trigger samples."""

    def __init__(self, target_id=None, requested=7, min_samples=3):
        self.target_id, self.requested, self.min_samples = target_id, requested, min_samples
        self.n_frames, self.samples, self.last_seen = 0, [], False

    @property
    def done(self):
        return self.n_frames >= self.requested

    def add(self, detections):
        if self.done:
            return
        if self.target_id is None:
            first = select_target(detections)
            self.target_id = first["track_id"] if first else None
        target = next((d for d in detections if d["track_id"] == self.target_id), None)
        self.last_seen = target is not None
        if target is not None and target["measurement"].values() is not None:
            self.samples.append(target["measurement"].values())
        self.n_frames += 1

    def result(self):
        values = tuple(float(v) for v in np.median(self.samples, axis=0)) if self.samples else None
        verdict, reason = assess(values)
        if not self.done:
            verdict, reason = None, f"Incomplete burst: {self.n_frames}/{self.requested} pairs"
        elif len(self.samples) < self.min_samples:
            verdict, reason = None, f"Insufficient depth: {len(self.samples)}/{self.min_samples} valid pairs"
        elif not self.last_seen:
            verdict, reason = None, "Target lost at end of burst; align and trigger again"
        return dict(target_id=self.target_id, n_frames=self.n_frames, requested=self.requested,
                    valid_frames=len(self.samples), values=values, verdict=verdict, reason=reason)


class PerchWorkflow:
    """LIVE -> T -> CAPTURING -> RESULT (frozen); R returns to LIVE."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.state, self.paused, self.target_id = "LIVE", False, None
        self.burst, self.result = None, None

    def trigger(self, n_frames=7, min_samples=3):
        if self.state == "CAPTURING":
            return False
        self.burst = PerchBurst(self.target_id, n_frames, min_samples)
        self.result = None
        self.state, self.paused = "CAPTURING", False
        return True

    def observe(self, detections):
        for det in detections:
            det.update(stable=None, verdict=None,
                       reason="Press T to measure" if self.state == "LIVE" else "Capturing burst")
        if self.state == "CAPTURING":
            self.burst.add(detections)
            self.target_id = self.burst.target_id
            if self.burst.done:
                self.finish(detections)

    def finish(self, detections):
        self.result = self.burst.result()
        self.state, self.paused = "RESULT", True
        for det in detections:
            if det["track_id"] == self.result["target_id"]:
                det.update(stable=self.result["values"], verdict=self.result["verdict"],
                           reason=self.result["reason"])


def draw_eye(image, detections, label, target_id=None, guide=None):
    annotated = image.copy()
    for det in detections:
        color = OK if det["verdict"] is True else FAIL if det["verdict"] is False else UNKNOWN
        mask = det["mask"].astype(bool)
        annotated[mask] = (0.7 * annotated[mask] + 0.3 * np.array(color)).astype(np.uint8)
        contours, _ = cv2.findContours(det["mask"], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated, contours, -1, color, 1, cv2.LINE_AA)
        if det["oriented"] is not None:
            cv2.polylines(annotated, [np.rint(det["oriented"]).astype(np.int32)], True, color, 1)
        x1, y1, _, _ = det["box"]
        selected = det["track_id"] == target_id
        caption = f"#{det['track_id']} {det['name']} {det['confidence']:.0%}" + ("  TARGET" if selected else "")
        text(annotated, caption, (max(5, min(x1, image.shape[1] - 320)), max(60, y1 - 10)), color)
    h, w = image.shape[:2]
    cv2.drawMarker(annotated, (w // 2, h // 2), (255, 220, 100), cv2.MARKER_CROSS, 26, 1)
    if guide and guide["axis"] is not None:
        axis = np.rint(guide["axis"]).astype(np.int32)
        cx, cy = np.rint(axis.mean(axis=0)).astype(int)
        cv2.line(annotated, tuple(axis[0]), tuple(axis[1]), (255, 255, 0), 2, cv2.LINE_AA)
        for x in range(max(0, cx - 100), min(w, cx + 100), 16):
            cv2.line(annotated, (x, cy), (min(w - 1, x + 8), cy), (255, 255, 255), 1)
    text(annotated, label, (12, 28), scale=0.65)
    return annotated


def compose_view(left, right, depth_color, detections, right_detections, guide,
                 state, index, total, fps, target_id, burst=None, result=None,
                 show_depth=True, paused=False):
    """A stable 2x2 layout: left/right eyes, reference depth, guidance/result."""
    left_view = draw_eye(left, detections, "LEFT - rectified", target_id, guide)
    right_view = draw_eye(right, right_detections, "RIGHT - rectified")
    depth_view = depth_color.copy() if show_depth else np.zeros_like(left)
    if show_depth:
        for det in detections:
            contours, _ = cv2.findContours(det["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(depth_view, contours, -1, UNKNOWN, 1)
    text(depth_view, "DEPTH - Plasma (reference)" if show_depth else "DEPTH HIDDEN - press D", (12, 28))
    panel = np.full_like(left, (24, 23, 20))
    h, w = panel.shape[:2]
    scale = min(w / 960, h / 540)
    # Fixed native panels are sized from calibration; resize only the final window.
    def line(value, y, color=(235, 235, 235), size=0.58):
        text(panel, value, (int(18 * scale), int(y * scale)), color, size * scale)
    line(f"{state}" + (" | PAUSED" if paused else "") + f" | pair {index+1}/{total} | {fps:.1f} FPS", 30)
    controls = "T=measure burst   R=live/reset   SPACE=pause"
    line(controls + ("   B=restart" if total != "LIVE" else ""), 60, size=0.49)
    line("D=depth   S=snapshot   Q/ESC=quit", 84, size=0.49)
    line(f"ALIGNMENT - target #{target_id}" if target_id is not None else "ALIGNMENT - no target", 122, UNKNOWN)
    if guide["angle_deg"] is not None:
        line(f"Image roll error: {guide['angle_deg']:+.1f} deg", 149)
    for i, message in enumerate(guide["messages"]):
        line(message, 178 + i * 26, OK if guide["ready"] else UNKNOWN, 0.51)
    if state == "CAPTURING":
        line(f"MEASURING: {burst.n_frames}/{burst.requested} stereo pairs", 308, UNKNOWN, 0.72)
        line("Median is computed separately for the selected target.", 340, size=0.51)
    elif result is not None:
        verdict = result["verdict"]
        color = OK if verdict is True else FAIL if verdict is False else UNKNOWN
        title = "CAN LAND (size/distance)" if verdict is True else "CANNOT LAND" if verdict is False else "INSUFFICIENT DATA"
        line(title, 306, color, 0.78)
        for i, message in enumerate(textwrap.wrap(result["reason"], width=76)):
            line(message, 336 + i * 24, color, 0.49)
        if result["values"] is not None:
            z, length, width = result["values"]
            line(f"Distance {z:.3f} m | Length {length*100:.1f} cm | Width {width*100:.1f} cm", 419, size=0.54)
        line(f"Burst: {result['n_frames']}/{result['requested']} pairs | valid: {result['valid_frames']}", 451, size=0.51)
    else:
        line("NOT MEASURED - align, then press T", 306, UNKNOWN, 0.67)
        line("Only the next burst is used for the landing assessment.", 341, size=0.51)
    line("2D view alignment; projected dimensions at representative depth.", 503, size=0.45)
    return cv2.vconcat((cv2.hconcat((left_view, right_view)), cv2.hconcat((depth_view, panel)))), left_view, right_view, depth_view


def records(detections):
    return [dict(track_id=d["track_id"], class_id=d["class_id"], name=d["name"],
                 confidence=d["confidence"], box=d["box"],
                 distance_m=d["measurement"].distance, length_m=d["measurement"].length,
                 width_m=d["measurement"].width,
                 valid_depth_fraction=d["measurement"].valid_fraction,
                 stable=d["stable"], samples=d["samples"], verdict=d["verdict"], reason=d["reason"])
            for d in detections]


def save_snapshot(directory, annotated, depth_color, depth, detections, index,
                  right=None, view=None, summary=None):
    directory.mkdir(parents=True, exist_ok=True)
    tag = time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1000000000:09d}_f{index:06d}"
    for suffix, img in (("seg", annotated), ("right", right), ("view", view), ("depth", depth_color)):
        if img is None:
            continue
        if not cv2.imwrite(str(directory / f"{tag}_{suffix}.png"), img):
            raise RuntimeError(f"Cannot save snapshot in {directory}")
    np.save(directory / f"{tag}_depth_m.npy", depth)
    for det in detections:
        if not cv2.imwrite(str(directory / f"{tag}_mask_{det['track_id']}.png"), det["mask"] * 255):
            raise RuntimeError(f"Cannot save instance mask in {directory}")
    (directory / f"{tag}.json").write_text(
        json.dumps(dict(pair_index=index, workflow=summary, detections=records(detections)), indent=2, allow_nan=False),
        encoding="utf-8")
    print(f"[SAVE] {directory / tag}")


def refresh_view(frame, workflow, args, source, show_depth):
    view, left_view, right_view, depth_view = compose_view(
        frame["left"], frame["right"], frame["depth_color"], frame["detections"],
        frame["right_detections"], frame["guide"], workflow.state, frame["index"],
        source.total, frame["fps"], workflow.target_id, workflow.burst, workflow.result,
        show_depth, workflow.paused)
    frame.update(view=view, left_view=left_view, right_view=right_view, depth_view=depth_view)
    frame["summary"] = dict(frame.get("summary", {}), state=workflow.state, target_id=workflow.target_id,
                            alignment=frame["guide"], burst=workflow.result,
                            left_frame=args.left_start + frame["index"],
                            right_frame=args.right_start + frame["index"])


def save_current(directory, frame):
    save_snapshot(directory, frame["left_view"], frame["depth_view"], frame["depth"],
                  frame["detections"], frame["index"], right=frame["right_view"],
                  view=frame["view"], summary=frame["summary"])


def process_stereo_pair(model, estimator, tracks, workflow, args, index, left, right,
                        right_mask=True):
    """Shared camera/video algorithm. The backend only supplies YOLO masks."""
    tick = time.perf_counter()
    left_rect, _, depth_raw = estimator.process(left, right)
    reference_color = depth_to_color(depth_raw, args.max_depth, args.unit)
    depth = depth_raw * UNIT_TO_M[args.unit]
    right_rect = estimator.rectify_right(right)
    image = left if args.seg_space == "original" else left_rect
    projector = estimator.rectify_mask if args.seg_space == "original" else None
    capture_depth = depth if workflow.state == "CAPTURING" else None
    detections = infer(model, image, capture_depth, float(estimator.Q[2, 3]), args, projector)
    tracks.update(detections, index)
    right_detections = []
    if right_mask:
        image_r = right if args.seg_space == "original" else right_rect
        project_r = (lambda mask: estimator.rectify_mask(mask, "right")) if args.seg_space == "original" else None
        right_detections = infer(model, image_r, None, float(estimator.Q[2, 3]), args, project_r)
        for i, det in enumerate(right_detections):
            det.update(track_id=f"R{i+1}", stable=None, samples=0, verdict=None,
                       reason="Right preview; measurement uses left depth")
    if workflow.state == "LIVE":
        target = select_target(detections, workflow.target_id)
        workflow.target_id = target["track_id"] if target else None
    workflow.observe(detections)
    target = next((d for d in detections if d["track_id"] == workflow.target_id), None)
    guide = alignment_hint(target["mask"] if target else None, args.align_angle, args.align_center)
    return dict(left=left_rect, right=right_rect, depth_color=reference_color, depth=depth,
                detections=detections, right_detections=right_detections, guide=guide,
                index=index, fps=1 / max(time.perf_counter()-tick, 1e-6))


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--left", type=Path, default=HERE / "l1.mp4")
    p.add_argument("--right", type=Path, default=HERE / "r1.mp4")
    p.add_argument("--model", type=Path, default=ROOT / "best-seg.pt")
    p.add_argument("--calib", type=Path, default=ROOT / "Calibration" / "stereo_calibration.npz")
    p.add_argument("--unit", choices=UNIT_TO_M, default="mm")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--iou", type=float, default=0.50, help="YOLO NMS IoU")
    p.add_argument("--imgsz", type=int, default=960, help="Inference size (multiple of 32)")
    p.add_argument("--seg-space", choices=("original", "rectified"), default="original",
                   help="Segment original video then rectify masks, or segment rectified image")
    p.add_argument("--device", default="cpu", help="cpu or 0 for CUDA GPU")
    p.add_argument("--classes", type=int, nargs="+", help="Optional perch class IDs")
    p.add_argument("--downscale", type=float, default=0.5,
                   help="Same stereo scale/default as Calibration/video_depth_estimation.py")
    p.add_argument("--min-mask-area", type=int, default=16, help="Minimum connected component pixels")
    p.add_argument("--min-depth-pixels", type=int, default=20)
    p.add_argument("--min-depth-fraction", type=float, default=0.15)
    p.add_argument("--history", type=int, default=7)
    p.add_argument("--min-samples", type=int, default=3, help="Valid frames per object before verdict")
    p.add_argument("--burst-frames", type=int, default=7, help="Next paired frames measured after T")
    p.add_argument("--align-angle", type=float, default=5.0, help="Image roll tolerance in degrees")
    p.add_argument("--align-center", type=float, default=0.12, help="Center tolerance as fraction of image size")
    p.add_argument("--trigger-frame", type=int,
                   help="Automatically press T after this zero-based pair (headless testing)")
    p.add_argument("--left-start", type=int, default=0, help="First left frame (zero based)")
    p.add_argument("--right-start", type=int, default=0, help="First right frame (zero based)")
    p.add_argument("--loop", action="store_true", help="Restart BOTH streams together at EOF")
    p.add_argument("--headless", action="store_true", help="No GUI; process until EOF/max-frames")
    p.add_argument("--max-frames", type=int, help="Limit processed pairs (useful for smoke tests)")
    p.add_argument("--max-depth", type=float, default=5.0, help="Depth colormap maximum in metres")
    p.add_argument("--output", type=Path, help="Save 2x2 left/right/depth/guidance video (.mp4)")
    p.add_argument("--snapshot-dir", type=Path, default=HERE / "perch_video_snapshots")
    p.add_argument("--save-last", action="store_true", help="Save last processed pair and masks at exit")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    for key in ("left", "right", "model", "calib"):
        if not getattr(args, key).is_file():
            parser.error(f"File not found: {getattr(args, key)}")
    if not (0 < args.conf <= 1 and 0 < args.iou <= 1 and 0 < args.downscale <= 1):
        parser.error("conf/iou/downscale must be in (0, 1]")
    if args.imgsz <= 0 or args.imgsz % 32:
        parser.error("imgsz must be a positive multiple of 32")
    if not (0 < args.min_depth_fraction <= 1 and args.max_depth > 0):
        parser.error("Invalid depth fraction or colormap maximum")
    if min(args.left_start, args.right_start) < 0 or min(args.min_mask_area, args.min_depth_pixels) < 1:
        parser.error("Start frames must be nonnegative; mask/depth pixel minima must be positive")
    if not (1 <= args.min_samples <= args.history) or (args.max_frames is not None and args.max_frames <= 0):
        parser.error("Require 1 <= min-samples <= history and positive max-frames")
    if args.burst_frames < args.min_samples:
        parser.error("burst-frames must be >= min-samples")
    if not (0 < args.align_angle < 90 and 0 < args.align_center < 0.5):
        parser.error("Require 0 < align-angle < 90 and 0 < align-center < 0.5")
    if args.trigger_frame is not None and args.trigger_frame < 0:
        parser.error("trigger-frame must be nonnegative")
    if args.headless and args.loop and args.max_frames is None:
        parser.error("headless + loop needs max-frames so the run can finish")
    if args.output and args.output.suffix.lower() != ".mp4":
        parser.error("output must have the .mp4 extension")
    if args.output and args.output.resolve() in {args.left.resolve(), args.right.resolve(),
                                                args.model.resolve(), args.calib.resolve()}:
        parser.error("output must not overwrite an input file")
    try:
        # Keep Ultralytics' first-run settings inside this project.
        os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install dependencies: pip install -r Python/requirements-perch-video.txt") from exc
    source = writer = None
    processed, last, cycle = 0, None, 0
    started = time.perf_counter()
    try:
        model = YOLO(str(args.model))
        if model.task != "segment":
            raise ValueError(f"Model task is {model.task!r}; best-seg.pt must be a segmentation model.")
        print(f"[MODEL] {args.model} | task={model.task} | classes={model.names}")
        source = VideoPair(args.left, args.right, (args.left_start, args.right_start))
        estimator = SegmentDepth(args, source.size)
        tracks = MaskTracks(args.history, args.min_samples)
        workflow = PerchWorkflow()
        auto_triggered = False
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"),
                                     source.fps, (estimator.size[0] * 2, estimator.size[1] * 2))
            if not writer.isOpened():
                raise RuntimeError(f"Cannot create video: {args.output}")
        window, show_depth = "Stereo Perch - T to measure", True
        if not args.headless:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window, 1440, 810)
        print(f"[CONTROLS] T={args.burst_frames}-pair burst, R=live/reset, SPACE=pause, B=restart, D=depth, S=save, Q=quit")
        while True:
            tick = time.perf_counter()
            if not workflow.paused:
                pair = source.read()
                if pair is None:
                    if workflow.state == "CAPTURING":
                        workflow.finish(last["detections"] if last else [])
                        if last:
                            refresh_view(last, workflow, args, source, show_depth)
                        if args.headless:
                            break
                        continue
                    if args.loop and source.index == source.total:
                        source.rewind()
                        tracks = MaskTracks(args.history, args.min_samples)
                        workflow.reset()
                        cycle += 1
                        continue
                    print("[VIDEO] End of paired stream.")
                    break
                index, left, right = pair
                last = process_stereo_pair(model, estimator, tracks, workflow, args, index, left, right)
                detections = last["detections"]
                processed += 1
                limit_reached = args.max_frames is not None and processed >= args.max_frames
                if workflow.state == "CAPTURING" and not workflow.burst.done and (limit_reached or source.index == source.total):
                    workflow.finish(detections)
                refresh_view(last, workflow, args, source, show_depth)
                if writer is not None:
                    writer.write(last["view"])
                if processed == 1 or processed % 30 == 0:
                    print(json.dumps(dict(pair=index, cycle=cycle, state=workflow.state, detections=records(detections)),
                                     allow_nan=False))
                if workflow.state == "RESULT":
                    print("[RESULT] " + json.dumps(workflow.result, allow_nan=False), flush=True)
                if limit_reached or (args.headless and workflow.state == "RESULT"):
                    break
            if args.headless:
                key = -1
            else:
                refresh_view(last, workflow, args, source, show_depth)
                cv2.imshow(window, last["view"])
                delay = max(1, round((1 / source.fps - (time.perf_counter() - tick)) * 1000))
                key = cv2.waitKey(30 if workflow.paused else delay) & 0xFF
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
            if key in (ord("q"), 27):
                break
            if (args.trigger_frame is not None and not auto_triggered and
                    last["index"] >= args.trigger_frame and workflow.state == "LIVE"):
                key, auto_triggered = ord("t"), True
            if key == ord("t"):
                if workflow.trigger(args.burst_frames, args.min_samples):
                    tracks.reset_measurements()
                    print(f"[BURST] Target #{workflow.target_id}; collecting next {args.burst_frames} pairs", flush=True)
            elif key == ord(" "):
                if workflow.state == "RESULT":
                    workflow.reset()
                    tracks.reset_measurements()
                else:
                    workflow.paused = not workflow.paused
            elif key == ord("d"):
                show_depth = not show_depth
            elif key == ord("r"):
                workflow.reset()
                tracks.reset_measurements()
            elif key == ord("b"):
                source.rewind()
                tracks = MaskTracks(args.history, args.min_samples)
                workflow.reset()
                cycle += 1
            elif key == ord("s"):
                save_current(args.snapshot_dir, last)
        if last is None:
            raise RuntimeError("No stereo pairs could be decoded.")
        if args.save_last:
            refresh_view(last, workflow, args, source, show_depth)
            save_current(args.snapshot_dir, last)
        print(f"[DONE] {processed} pairs in {time.perf_counter()-started:.1f}s")
    finally:
        if source is not None:
            source.release()
        if writer is not None:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, KeyError, OSError, cv2.error) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
