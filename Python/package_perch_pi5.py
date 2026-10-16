"""Build a portable Pi deployment zip from explicit runtime files and FP16 models."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parent.parent


def main():
    files = ["LICENSE", "Calibration/video_depth_estimation.py", "Calibration/stereo_calibration.npz",
             "Python/perch_detector.py", "Python/perch_ncnn.py", "Python/perch_video_seg.py",
             "Python/camera.py", "Python/requirements-perch-pi5.txt", "Python/setup_perch_pi5.sh",
             "Python/perch_detector_pi5_README.md", "Python/perch_video_seg_README.md",
             "Python/benchmark_perch_pi5.py", "models/validation_pi5.json"]
    for profile in (640, 960):
        files.extend(f"models/perch_pi5_{profile}_fp16_ncnn_model/{name}"
                     for name in ("model.ncnn.param", "model.ncnn.bin", "metadata.yaml", "manifest.json"))
    for path in files:
        if not (ROOT / path).is_file():
            raise FileNotFoundError(path)
    manifest = {path:dict(bytes=(ROOT/path).stat().st_size,
                         sha256=hashlib.sha256((ROOT/path).read_bytes()).hexdigest()) for path in files}
    archive = ROOT / "perch_pi5_deploy.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in files:
            z.write(ROOT / path, "perch_pi5/" + path)
        z.write(ROOT / "Python/perch_detector_pi5_README.md", "perch_pi5/README_PI5.md")
        z.writestr("perch_pi5/deployment_manifest.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise RuntimeError("Zip integrity check failed")
        for path, info in manifest.items():
            if hashlib.sha256(z.read("perch_pi5/" + path)).hexdigest() != info["sha256"]:
                raise RuntimeError(f"Zip payload hash mismatch: {path}")
    print(f"Created {archive} ({archive.stat().st_size:,} bytes), {len(files)} runtime/model files; hashes verified.")


if __name__ == "__main__":
    main()
