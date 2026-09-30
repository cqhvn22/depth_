"""
train_cylinder_seg.py
=====================

Full pipeline for training a YOLOv8n-seg model to detect cylindrical objects,
with a built-in test/inference mode that renders true segmentation borders
(contours) instead of plain bounding boxes.

Dataset source:
    Roboflow – arturo-perez-fzrw0 / cylinder-eiuoi
    API key : HwZYq22onvjG9WU7kJSZ

Usage
-----
  # 1. Download dataset + train (all-in-one)
  python train_cylinder_seg.py train

  # 2. Train only (dataset already downloaded)
  python train_cylinder_seg.py train --skip-download

  # 3. Resume an interrupted training
  python train_cylinder_seg.py train --resume

  # 4. Test on webcam (live)
  python train_cylinder_seg.py test --source webcam

  # 5. Test on an image or video file
  python train_cylinder_seg.py test --source path/to/image.jpg
  python train_cylinder_seg.py test --source path/to/video.mp4

  # 6. Test with a specific model weights file
  python train_cylinder_seg.py test --source webcam --weights runs/segment/cylinder_seg/weights/best.pt

Controls (test mode)
--------------------
  Q / ESC  : Quit
  SPACE    : Pause / Resume (video / webcam)
  S        : Save current frame as PNG snapshot
  T        : Cycle confidence threshold  (0.25 → 0.35 → 0.50)
  M        : Toggle mask fill ON / OFF
  +/-      : Increase / Decrease mask opacity
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROBOFLOW_API_KEY = "HwZYq22onvjG9WU7kJSZ"
ROBOFLOW_WORKSPACE = "arturo-perez-fzrw0"
ROBOFLOW_PROJECT = "cylinder-eiuoi"
ROBOFLOW_VERSION = 1          # Change if the dataset has multiple versions

BASE_MODEL = "yolov8n-seg.pt"  # pretrained backbone
TRAIN_EPOCHS = 100
IMG_SIZE = 640
BATCH_SIZE = 16
PROJECT_NAME = "cylinder_seg_project"
RUN_NAME = "cylinder_seg"

CONF_CYCLE = [0.25, 0.35, 0.50]

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(0)
_PALETTE: list[tuple[int, int, int]] = [
    tuple(int(c) for c in _RNG.integers(80, 230, 3)) for _ in range(100)
]


def class_colour(cls_id: int) -> tuple[int, int, int]:
    return _PALETTE[int(cls_id) % len(_PALETTE)]


# ---------------------------------------------------------------------------
# Download dataset from Roboflow
# ---------------------------------------------------------------------------

def download_dataset(dest_dir: str = "cylinder_dataset") -> str:
    """Download the cylinder dataset from Roboflow and return the data.yaml path."""
    print("\n[Dataset] Downloading from Roboflow …")
    try:
        from roboflow import Roboflow  # type: ignore
    except ImportError:
        sys.exit(
            "[ERROR] 'roboflow' package not found.\n"
            "  Install with:  pip install roboflow"
        )

    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    version = project.version(ROBOFLOW_VERSION)

    dataset = version.download("yolov8", location=dest_dir, overwrite=True)
    yaml_path = os.path.join(dest_dir, "data.yaml")

    if not os.path.exists(yaml_path):
        # Some Roboflow downloads place yaml one level deeper
        for root, _, files in os.walk(dest_dir):
            for f in files:
                if f.endswith(".yaml"):
                    yaml_path = os.path.join(root, f)
                    break

    print(f"[Dataset] data.yaml → {yaml_path}\n")
    return yaml_path


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def run_train(args: argparse.Namespace) -> None:
    """Download dataset (optional) and fine-tune YOLOv8n-seg."""
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        sys.exit("[ERROR] 'ultralytics' not found.  pip install ultralytics")

    # --- Dataset ---
    if args.skip_download:
        yaml_path = args.data_yaml or "cylinder_dataset/data.yaml"
        if not os.path.exists(yaml_path):
            sys.exit(
                f"[ERROR] data.yaml not found at '{yaml_path}'.\n"
                "  Run without --skip-download, or pass --data-yaml <path>."
            )
    else:
        yaml_path = download_dataset(args.dataset_dir)

    # --- Model ---
    weights = BASE_MODEL
    if args.resume:
        last_ckpt = Path(f"runs/segment/{RUN_NAME}/weights/last.pt")
        if last_ckpt.exists():
            weights = str(last_ckpt)
            print(f"[Train] Resuming from {weights}")
        else:
            print(f"[Train] No checkpoint found at {last_ckpt}, starting fresh.")

    print(f"[Train] Loading base model : {weights}")
    model = YOLO(weights)

    # --- Training ---
    print(
        f"[Train] Starting training …\n"
        f"  Dataset  : {yaml_path}\n"
        f"  Epochs   : {args.epochs}\n"
        f"  Image sz : {args.imgsz}\n"
        f"  Batch    : {args.batch}\n"
        f"  Device   : {args.device}\n"
    )

    results = model.train(
        data=yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=PROJECT_NAME,
        name=RUN_NAME,
        resume=args.resume,
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        # Training settings
        patience=30,
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
    )

    print("\n[Train] Training complete!")
    best = Path(PROJECT_NAME) / RUN_NAME / "weights" / "best.pt"
    print(f"[Train] Best weights → {best.resolve()}")

    # Quick validation
    print("\n[Train] Running validation on best weights …")
    val_model = YOLO(str(best))
    val_model.val(data=yaml_path, imgsz=args.imgsz, device=args.device)


# ---------------------------------------------------------------------------
# Inference / visualisation helpers
# ---------------------------------------------------------------------------

def draw_mask_border(
    img: np.ndarray,
    mask_polygon: np.ndarray,
    colour: tuple[int, int, int],
    fill: bool = True,
    alpha: float = 0.30,
) -> None:
    """
    Render a segmentation mask with:
      • Semi-transparent fill (when fill=True)
      • Glowing / anti-aliased border outline – NO plain bounding box

    Parameters
    ----------
    img          : BGR image to draw on (modified in-place).
    mask_polygon : (N, 2) int32 array of polygon vertices in pixel space.
    colour       : BGR colour tuple.
    fill         : Whether to draw the semi-transparent fill.
    alpha        : Opacity of the fill (0 = transparent, 1 = opaque).
    """
    if mask_polygon is None or len(mask_polygon) < 3:
        return

    pts = mask_polygon.reshape((-1, 1, 2)).astype(np.int32)

    # --- Semi-transparent fill ---
    if fill:
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], colour)
        cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)

    # --- Outer glow (thicker, dimmer) ---
    glow = tuple(max(0, int(c * 0.6)) for c in colour)
    cv2.polylines(img, [pts], isClosed=True, color=glow,   thickness=6,
                  lineType=cv2.LINE_AA)

    # --- Main border ---
    cv2.polylines(img, [pts], isClosed=True, color=colour, thickness=3,
                  lineType=cv2.LINE_AA)

    # --- Bright inner highlight ---
    bright = tuple(min(255, int(c * 1.6)) for c in colour)
    cv2.polylines(img, [pts], isClosed=True, color=bright, thickness=1,
                  lineType=cv2.LINE_AA)


def draw_label(
    img: np.ndarray,
    text_lines: list[str],
    anchor: tuple[int, int],
    colour: tuple[int, int, int],
) -> None:
    """Draw a multi-line label box at anchor (top-left corner)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thick = 1
    pad = 5
    line_h = 20

    text_w = max(
        cv2.getTextSize(ln, font, scale, thick)[0][0]
        for ln in text_lines
    )
    box_h = line_h * len(text_lines) + pad * 2

    ax, ay = anchor
    # Keep inside image
    h_img, w_img = img.shape[:2]
    ay = max(0, min(ay, h_img - box_h))
    ax = max(0, min(ax, w_img - text_w - pad * 2))

    # Background
    bg = tuple(max(0, int(c * 0.4)) for c in colour)
    cv2.rectangle(img, (ax, ay), (ax + text_w + pad * 2, ay + box_h), bg, -1)
    cv2.rectangle(img, (ax, ay), (ax + text_w + pad * 2, ay + box_h), colour, 1)

    for i, ln in enumerate(text_lines):
        ty = ay + pad + line_h * (i + 1) - 2
        cv2.putText(img, ln, (ax + pad, ty),
                    font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def draw_crosshair(
    img: np.ndarray,
    cx: int,
    cy: int,
    colour: tuple[int, int, int] = (0, 255, 80),
) -> None:
    arm = 14
    cv2.line(img, (cx - arm, cy), (cx + arm, cy), colour, 2, cv2.LINE_AA)
    cv2.line(img, (cx, cy - arm), (cx, cy + arm), colour, 2, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 4, colour, -1)


def centroid_from_polygon(poly: np.ndarray, fallback: tuple[int, int]) -> tuple[int, int]:
    M = cv2.moments(poly.reshape((-1, 1, 2)).astype(np.int32))
    if M["m00"] != 0:
        return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    return fallback


def overlay_hud(
    img: np.ndarray,
    fps: float,
    n_det: int,
    conf: float,
    paused: bool,
    fill_masks: bool,
    mask_alpha: float,
    source_label: str,
) -> None:
    """Draw HUD info in the top-left corner."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = [
        f"FPS: {fps:.1f}   Source: {source_label}",
        f"Detections: {n_det}   Conf: {conf:.2f}  [T]=cycle",
        f"Mask fill: {'ON' if fill_masks else 'OFF'}  [M]   "
        f"Opacity: {mask_alpha:.2f}  [+/-]",
        "[PAUSED] SPACE=resume" if paused else "",
    ]
    for i, ln in enumerate(lines):
        if not ln:
            continue
        # Shadow
        cv2.putText(img, ln, (12, 28 + i * 24),
                    font, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        # Text
        cv2.putText(img, ln, (12, 28 + i * 24),
                    font, 0.55, (180, 255, 180), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Test / inference
# ---------------------------------------------------------------------------

def run_test(args: argparse.Namespace) -> None:
    """Run inference on webcam or image/video with mask-border visualisation."""
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        sys.exit("[ERROR] 'ultralytics' not found.  pip install ultralytics")

    # --- Resolve weights ---
    weights = args.weights
    if weights is None:
        candidates = [
            Path(PROJECT_NAME) / RUN_NAME / "weights" / "best.pt",
            Path("runs/segment") / RUN_NAME / "weights" / "best.pt",
            Path(BASE_MODEL),
        ]
        for c in candidates:
            if c.exists():
                weights = str(c)
                break
        if weights is None:
            sys.exit(
                "[ERROR] No model weights found.\n"
                "  Train first, or pass --weights <path/to/best.pt>"
            )
    print(f"[Test] Loading model : {weights}")
    model = YOLO(weights)

    # --- Source ---
    src = args.source.lower()
    is_webcam = src in ("webcam", "0", "cam")
    is_image = False

    if is_webcam:
        cap = cv2.VideoCapture(0)
        source_label = "Webcam"
        print("[Test] Opening webcam … press Q/ESC to quit.")
    else:
        src_path = Path(args.source)
        if not src_path.exists():
            sys.exit(f"[ERROR] Source not found: {src_path}")

        ext = src_path.suffix.lower()
        if ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"):
            is_image = True
            source_label = src_path.name
            img = cv2.imread(str(src_path))
            if img is None:
                sys.exit(f"[ERROR] Cannot read image: {src_path}")
        else:
            cap = cv2.VideoCapture(str(src_path))
            source_label = src_path.name
            print(f"[Test] Opening video : {src_path}")

    conf_idx = 0
    conf = CONF_CYCLE[conf_idx]
    fill_masks = True
    mask_alpha = 0.30
    paused = False
    snapshot_dir = Path("cylinder_snapshots")
    fps_display = 0.0
    t_last = time.monotonic()

    print("[Test] Controls: Q/ESC=quit  SPACE=pause  S=snapshot  T=conf  M=mask  +/-=opacity\n")

    # ---- Image mode (single frame) ----
    if is_image:
        annotated = _infer_frame(model, img, conf, fill_masks, mask_alpha)
        fps_display = 0.0
        n_det = 0  # updated inside _infer_frame but not returned
        overlay_hud(annotated, 0.0, 0, conf, False, fill_masks, mask_alpha, source_label)
        win = "Cylinder Detection – Image  [S=save  Q=quit]"
        cv2.imshow(win, annotated)
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("s"):
                snapshot_dir.mkdir(parents=True, exist_ok=True)
                out_path = snapshot_dir / f"snapshot_{int(time.time())}.png"
                cv2.imwrite(str(out_path), annotated)
                print(f"  Snapshot → {out_path}")
            elif key == ord("t"):
                conf_idx = (conf_idx + 1) % len(CONF_CYCLE)
                conf = CONF_CYCLE[conf_idx]
                annotated = _infer_frame(model, img, conf, fill_masks, mask_alpha)
                overlay_hud(annotated, 0.0, 0, conf, False, fill_masks, mask_alpha, source_label)
                cv2.imshow(win, annotated)
                print(f"  Conf → {conf:.2f}")
            elif key == ord("m"):
                fill_masks = not fill_masks
                annotated = _infer_frame(model, img, conf, fill_masks, mask_alpha)
                overlay_hud(annotated, 0.0, 0, conf, False, fill_masks, mask_alpha, source_label)
                cv2.imshow(win, annotated)
            elif key == ord("+") or key == ord("="):
                mask_alpha = min(0.85, mask_alpha + 0.05)
                annotated = _infer_frame(model, img, conf, fill_masks, mask_alpha)
                overlay_hud(annotated, 0.0, 0, conf, False, fill_masks, mask_alpha, source_label)
                cv2.imshow(win, annotated)
            elif key == ord("-"):
                mask_alpha = max(0.05, mask_alpha - 0.05)
                annotated = _infer_frame(model, img, conf, fill_masks, mask_alpha)
                overlay_hud(annotated, 0.0, 0, conf, False, fill_masks, mask_alpha, source_label)
                cv2.imshow(win, annotated)
        cv2.destroyAllWindows()
        return

    # ---- Video / Webcam mode ----
    win = "Cylinder Detection  [Q=quit SPACE=pause S=snap T=conf M=mask +/-=opacity]"
    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                if is_webcam:
                    print("[Test] No frame received – retrying …")
                    time.sleep(0.05)
                    continue
                else:
                    print("[Test] End of video.")
                    break

            t0 = time.monotonic()
            annotated, n_det = _infer_frame(model, frame, conf, fill_masks, mask_alpha,
                                            return_n_det=True)
            t1 = time.monotonic()

            fps_display = 1.0 / max(t1 - t_last, 1e-6)
            t_last = t1

            overlay_hud(annotated, fps_display, n_det, conf, paused,
                        fill_masks, mask_alpha, source_label)

            cv2.imshow(win, annotated)
            delay_ms = 1
        else:
            delay_ms = 50

        key = cv2.waitKey(delay_ms) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            paused = not paused
            print("Paused." if paused else "Resumed.")
        elif key == ord("s"):
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            out_path = snapshot_dir / f"snapshot_{int(time.time())}.png"
            cv2.imwrite(str(out_path), annotated)
            print(f"  Snapshot → {out_path}")
        elif key == ord("t"):
            conf_idx = (conf_idx + 1) % len(CONF_CYCLE)
            conf = CONF_CYCLE[conf_idx]
            print(f"  Conf → {conf:.2f}")
        elif key == ord("m"):
            fill_masks = not fill_masks
            print(f"  Mask fill: {'ON' if fill_masks else 'OFF'}")
        elif key == ord("+") or key == ord("="):
            mask_alpha = min(0.85, mask_alpha + 0.05)
            print(f"  Mask opacity: {mask_alpha:.2f}")
        elif key == ord("-"):
            mask_alpha = max(0.05, mask_alpha - 0.05)
            print(f"  Mask opacity: {mask_alpha:.2f}")

    cap.release()
    cv2.destroyAllWindows()
    print("[Test] Done.")


# ---------------------------------------------------------------------------
# Core inference frame processor
# ---------------------------------------------------------------------------

def _infer_frame(
    model,
    frame: np.ndarray,
    conf: float,
    fill_masks: bool,
    mask_alpha: float,
    return_n_det: bool = False,
):
    """
    Run YOLOv8-seg inference on one frame and draw mask contours.

    Returns
    -------
    annotated : BGR image with visualisation.
    n_det     : number of detections (only when return_n_det=True).
    """
    results = model(frame, conf=conf, verbose=False)
    annotated = frame.copy()

    boxes = results[0].boxes
    masks = results[0].masks   # None if model has no seg head
    n_det = len(boxes) if boxes is not None else 0

    if boxes is not None and n_det > 0:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            conf_v = float(box.conf[0])
            name = model.names[cls_id]
            colour = class_colour(cls_id)

            # ---- Segmentation mask border (preferred) ----
            poly: np.ndarray | None = None
            if masks is not None and i < len(masks):
                xy = masks[i].xy
                if xy is not None and len(xy) > 0 and len(xy[0]) >= 3:
                    poly = xy[0].astype(np.int32)

            if poly is not None:
                draw_mask_border(annotated, poly, colour, fill=fill_masks, alpha=mask_alpha)
                cx, cy = centroid_from_polygon(poly, ((x1 + x2) // 2, (y1 + y2) // 2))
            else:
                # Fallback: draw just the outline rectangle (no filled box)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2, cv2.LINE_AA)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            draw_crosshair(annotated, cx, cy)

            # ---- Label ----
            label_lines = [
                f"{name}  {conf_v:.0%}",
                f"({cx}, {cy}) px",
            ]
            # Place label above the bounding box (or mask bounding rect)
            draw_label(annotated, label_lines, (x1, max(0, y1 - 50)), colour)

    if return_n_det:
        return annotated, n_det
    return annotated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="YOLOv8n-seg cylinder detection: train & test pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ---- train sub-command ----
    tr = sub.add_parser("train", help="Download dataset and train the model.")
    tr.add_argument("--skip-download", action="store_true",
                    help="Skip Roboflow download (use existing dataset).")
    tr.add_argument("--data-yaml", default=None,
                    help="Path to data.yaml (used with --skip-download).")
    tr.add_argument("--dataset-dir", default="cylinder_dataset",
                    help="Directory to download the dataset into (default: cylinder_dataset).")
    tr.add_argument("--epochs", type=int, default=TRAIN_EPOCHS,
                    help=f"Number of training epochs (default: {TRAIN_EPOCHS}).")
    tr.add_argument("--imgsz", type=int, default=IMG_SIZE,
                    help=f"Input image size (default: {IMG_SIZE}).")
    tr.add_argument("--batch", type=int, default=BATCH_SIZE,
                    help=f"Batch size (default: {BATCH_SIZE}).")
    tr.add_argument("--device", default="",
                    help="Device: '' (auto), 'cpu', '0' (GPU 0), '0,1' (multi-GPU).")
    tr.add_argument("--resume", action="store_true",
                    help="Resume from last.pt checkpoint.")

    # ---- test sub-command ----
    ts = sub.add_parser("test", help="Run inference on webcam or image/video.")
    ts.add_argument("--source", default="webcam",
                    help="Input source: 'webcam' | path/to/image | path/to/video "
                         "(default: webcam).")
    ts.add_argument("--weights", default=None,
                    help="Path to model weights .pt file. "
                         "Auto-detects best.pt from training output if not given.")
    ts.add_argument("--conf", type=float, default=0.35,
                    help="Initial confidence threshold (default: 0.35).")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "train":
        run_train(args)
    elif args.cmd == "test":
        run_test(args)


if __name__ == "__main__":
    main()
