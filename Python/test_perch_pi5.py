"""Pi camera source and lightweight NCNN postprocessing tests; no Pi/Torch needed."""
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest

import numpy as np

from perch_detector import PicameraPairSource, build_parser
from perch_ncnn import class_nms, letterbox, NCNNSegmenter


class FakeCamera:
    def __init__(self, sequences, open_ok=True):
        self.frames = [SimpleNamespace(sequence=i, left=np.full((4, 4, 3), (255, 0, 0), np.uint8),
                                       right=np.full((4, 4, 3), (0, 0, 255), np.uint8), delta_ms=1.0)
                       for i in sequences]
        self.open_ok, self.released = open_ok, False
    def open(self):
        return self.open_ok
    def start(self):
        return True
    def read(self, **kwargs):
        return (True, self.frames.pop(0)) if self.frames else (False, None)
    def release(self):
        self.released = True


class PiCameraTests(unittest.TestCase):
    def test_duplicate_sequences_do_not_count_and_rgb_becomes_bgr(self):
        camera = FakeCamera([1, 1, 2])
        source = PicameraPairSource(build_parser().parse_args([]), lambda **kw: camera)
        try:
            a, left, right = source.read()
            b, *_ = source.read()
            self.assertEqual((a, b, source.sequence), (0, 1, 2))
            np.testing.assert_array_equal(left[0, 0], (0, 0, 255))
            np.testing.assert_array_equal(right[0, 0], (255, 0, 0))
        finally:
            source.release()
        self.assertTrue(camera.released)

    def test_t_discards_pair_buffered_before_trigger(self):
        camera = FakeCamera([10, 11, 11, 12])
        source = PicameraPairSource(build_parser().parse_args([]), lambda **kw: camera)
        try:
            source.read()
            source.discard_buffered()  # sequence 11 was captured before T
            index, *_ = source.read()
            self.assertEqual((index, source.sequence), (1, 12))
        finally:
            source.release()

    def test_start_failure_releases_cameras(self):
        camera = FakeCamera([], open_ok=False)
        with self.assertRaisesRegex(RuntimeError, "Cannot start"):
            PicameraPairSource(build_parser().parse_args([]), lambda **kw: camera)
        self.assertTrue(camera.released)

    def test_camera_stall_is_reported(self):
        camera = FakeCamera([])
        args = build_parser().parse_args(["--camera-timeout", "0.01"])
        source = PicameraPairSource(args, lambda **kw: camera)
        try:
            with self.assertRaises(TimeoutError):
                source.read()
        finally:
            source.release()


class LightweightBackendTests(unittest.TestCase):
    def test_imports_do_not_load_torch_or_ultralytics(self):
        script = ("import sys; sys.path.insert(0, 'Python'); import perch_detector, perch_ncnn; "
                  "assert 'torch' not in sys.modules; assert 'ultralytics' not in sys.modules")
        subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parent.parent,
                       check=True, capture_output=True)

    def test_letterbox_preserves_aspect_and_rgb_channel_order(self):
        image = np.full((100, 200, 3), (255, 0, 0), np.uint8)
        tensor, gain, pad = letterbox(image, (384, 640))
        self.assertEqual(tensor.shape, (3, 384, 640))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertEqual(pad, (0, 32))
        self.assertEqual(gain, (3.2, 3.2))
        np.testing.assert_array_equal(tensor[:, 100, 100], (0, 0, 1))

    def test_nms_keeps_distinct_classes_but_suppresses_duplicate_boxes(self):
        boxes = np.array([[0, 0, 100, 100]] * 3, np.float32)
        scores = np.array([.9, .7, .8], np.float32)
        classes = np.array([0, 0, 1])
        np.testing.assert_array_equal(class_nms(boxes, scores, classes), (0, 2))
        self.assertEqual(len(class_nms(boxes[:0], scores[:0], classes[:0])), 0)

    def test_exported_model_returns_native_masks_on_cpu(self):
        directory = Path(__file__).resolve().parent.parent / "models/perch_pi5_640_fp16_ncnn_model"
        if not directory.is_dir():
            self.skipTest("NCNN model not present")
        try:
            model = NCNNSegmenter(directory, threads=2)
        except ImportError:
            self.skipTest("NCNN not installed")
        output = model.predict(np.zeros((540, 960, 3), np.uint8))[0]
        self.assertEqual(output.boxes.data.numpy().shape[1], 6)
        if output.masks is not None:
            self.assertEqual(output.masks.data.numpy().shape[1:], (540, 960))


if __name__ == "__main__":
    unittest.main()
