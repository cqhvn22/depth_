"""Export trained YOLOv8n-seg to rectangular FP16 NCNN for Raspberry Pi 5 CPU.

python Python/export_perch_pi5.py
python Python/export_perch_pi5.py --width 960 --height 544
Requires ultralytics==8.4.178, ncnn, pnnx==20260526.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parent.parent


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "best-seg.pt")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.weights.is_file():
        parser.error(f"Missing checkpoint: {args.weights}")
    if min(args.width, args.height) < 32 or args.width % 32 or args.height % 32:
        parser.error("width/height must be positive multiples of 32")
    output = args.output or ROOT / "models" / f"perch_pi5_{args.width}_{args.precision}_ncnn_model"
    if output.exists():
        parser.error(f"Output already exists: {output}. Choose another --output.")
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
    from ultralytics import YOLO, __version__
    output.parent.mkdir(parents=True, exist_ok=True)
    # Export beside a copy, leaving the trained checkpoint and other profiles intact.
    staging = output.parent / f".perch_export_{time.time_ns()}"
    staging.mkdir()
    checkpoint = staging / "perch.pt"
    shutil.copy2(args.weights, checkpoint)
    model = YOLO(str(checkpoint))
    if model.task != "segment":
        parser.error("Checkpoint must have an instance segmentation head")
    quantize = 16 if args.precision == "fp16" else 32
    exported = Path(model.export(format="ncnn", imgsz=[args.height, args.width],
                                 quantize=quantize, batch=1, device="cpu", verbose=False))
    for name in ("model.ncnn.param", "model.ncnn.bin", "metadata.yaml"):
        if not (exported / name).is_file():
            raise RuntimeError(f"Export is incomplete: {exported / name}")
    staging_root = staging.resolve()
    if not exported.resolve().is_relative_to(staging_root):
        raise RuntimeError("Refusing to move export outside this invocation's staging directory")
    if output.resolve().parent != staging_root.parent:
        raise RuntimeError("Output must remain in the chosen model parent directory")
    shutil.move(str(exported), str(output))
    manifest = dict(source_weights=args.weights.name, source_sha256=sha256(args.weights),
                    architecture="yolov8n-seg", task="segment", format="ncnn",
                    weight_precision=args.precision, input_shape=[1, 3, args.height, args.width],
                    ultralytics=__version__, classes=model.names,
                    files={name:dict(bytes=(output/name).stat().st_size, sha256=sha256(output/name))
                           for name in ("model.ncnn.param", "model.ncnn.bin", "metadata.yaml")},
                    note=f"{args.precision.upper()} weight storage. Measure throughput on the actual Pi 5.")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # All remaining staging entries were created by this invocation only.
    if (staging.resolve().parent != output.resolve().parent or
            not staging.name.startswith(".perch_export_")):
        raise RuntimeError("Refusing to remove an unexpected staging path")
    shutil.rmtree(staging.resolve())
    print(json.dumps(dict(output=str(output), **manifest), indent=2), flush=True)


if __name__ == "__main__":
    main()
