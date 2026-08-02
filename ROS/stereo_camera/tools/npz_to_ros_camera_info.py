#!/usr/bin/env python3
"""Convert this project's OpenCV stereo .npz file to ROS CameraInfo YAML files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml


def matrix_block(array: np.ndarray) -> dict:
    array = np.asarray(array)
    return {
        "rows": int(array.shape[0]),
        "cols": int(array.shape[1]),
        "data": [float(value) for value in array.reshape(-1)],
    }


def camera_info(
    name: str,
    width: int,
    height: int,
    k: np.ndarray,
    d: np.ndarray,
    r: np.ndarray,
    p: np.ndarray,
) -> dict:
    d = np.asarray(d).reshape(-1)
    return {
        "image_width": width,
        "image_height": height,
        "camera_name": name,
        "camera_matrix": matrix_block(k),
        "distortion_model": "plumb_bob",
        "distortion_coefficients": {
            "rows": 1,
            "cols": int(d.size),
            "data": [float(value) for value in d],
        },
        "rectification_matrix": matrix_block(r),
        "projection_matrix": matrix_block(p),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--translation-unit",
        choices=("mm", "m"),
        default="mm",
        help="Unit used by T/P/Q in the NPZ. This calibration uses mm.",
    )
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=False)
    width, height = (int(value) for value in data["image_size"])

    p_right_ros = data["P_right"].astype(float).copy()
    if args.translation_unit == "mm":
        p_right_ros[0, 3] /= 1000.0
        p_right_ros[1, 3] /= 1000.0

    args.output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "left_camera_info.yaml": camera_info(
            "imx219_left_960x540",
            width,
            height,
            data["K_left"],
            data["D_left"],
            data["R_left"],
            data["P_left"],
        ),
        "right_camera_info.yaml": camera_info(
            "imx219_right_960x540",
            width,
            height,
            data["K_right"],
            data["D_right"],
            data["R_right"],
            p_right_ros,
        ),
    }

    for filename, content in outputs.items():
        with open(args.output_dir / filename, "w", encoding="utf-8") as stream:
            yaml.safe_dump(content, stream, sort_keys=False)

    baseline_mm = float(np.linalg.norm(data["T"]))
    print(f"Image size: {width}x{height}")
    print(f"Baseline: {baseline_mm:.3f} mm")
    print(f"Wrote files to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
