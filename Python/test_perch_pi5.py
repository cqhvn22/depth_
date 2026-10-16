"""Pi camera source and lightweight NCNN postprocessing tests; no Pi/Torch needed."""
from pathlib import Path
import contextlib
import io
import json
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from perch_detector import PicameraPairSource, build_parser
from perch_ncnn import ArrayBoxes, ArrayTensor, class_nms, letterbox, NCNNSegmenter
from perch_video_seg import (MaskTracks, PerchWorkflow, SegmentDepth,
                             process_stereo_pair, refresh_view, save_current)


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


class TriggerDepthTests(unittest.TestCase):
    def test_stereo_work_only_runs_in_trigger_burst_and_reset_clears_depth(self):
        """Exercise the pipeline with real SGBM/WLS and deterministic masks."""
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            yy, xx = np.indices((160, 320), dtype=np.float32)
            q = np.array([[1, 0, 0, -160], [0, 1, 0, -80],
                          [0, 0, 0, 200], [0, 0, 1 / 60, 0]], np.float64)
            calib = directory / 'calib.npz'
            np.savez(calib, left_map_x=xx, left_map_y=yy,
                     right_map_x=xx, right_map_y=yy, Q=q)
            args = build_parser().parse_args(['--calib', str(calib)])
            args.imgsz, args.device = 640, 'cpu'
            with contextlib.redirect_stdout(io.StringIO()):
                estimator = SegmentDepth(args)
            left = np.random.default_rng(42).integers(0, 256, (160, 320, 3), np.uint8)
            right = np.zeros_like(left)
            right[:, :-20] = left[:, 20:]
            mask = np.zeros((1, 160, 320), np.uint8)
            mask[:, 70:90, 160:280] = 1
            prediction = SimpleNamespace(
                boxes=ArrayBoxes(np.array([[160, 70, 280, 90, .9, 0]], np.float32)),
                masks=SimpleNamespace(data=ArrayTensor(mask)))
            model = SimpleNamespace(names={0: 'Perch'}, predict=Mock(return_value=[prediction]))
            tracks, workflow = MaskTracks(), PerchWorkflow()
            source = SimpleNamespace(total='LIVE')
            compute = Mock(wraps=estimator.matcher.compute)
            compute_right = Mock(wraps=estimator.right_matcher.compute)
            filtering = Mock(wraps=estimator.wls_filter.filter)
            estimator.matcher = SimpleNamespace(compute=compute)
            estimator.right_matcher = SimpleNamespace(compute=compute_right)
            estimator.wls_filter = SimpleNamespace(filter=filtering)
            with patch('perch_video_seg.cv2.reprojectImageTo3D', wraps=cv2.reprojectImageTo3D) as reproject:
                for index in range(3):
                    live = process_stereo_pair(model, estimator, tracks, workflow, args, index, left, right)
                    self.assertIsNone(live['depth'])
                    self.assertIsNone(live['depth_color'])
                    self.assertTrue(live['detections'])
                    self.assertTrue(live['right_detections'])
                    self.assertIsNone(live['detections'][0]['measurement'].values())
                for operation in (compute, compute_right, filtering, reproject):
                    operation.assert_not_called()
                self.assertTrue(workflow.trigger())
                tracks.reset_measurements()
                for index in range(3, 10):
                    result = process_stereo_pair(model, estimator, tracks, workflow, args, index, left, right)
                    self.assertIsNotNone(result['depth'])
                self.assertEqual(workflow.state, 'RESULT')
                self.assertTrue(workflow.paused)
                self.assertEqual(workflow.result['valid_frames'], 7)
                self.assertTrue(workflow.result['verdict'])
                for visible in (True, False, True):
                    refresh_view(result, workflow, args, source, visible)
                self.assertTrue(result['summary']['depth_computed'])
                with contextlib.redirect_stdout(io.StringIO()):
                    save_current(directory / 'result', result)
                stored = next((directory / 'result').glob('*_depth_m.npy'))
                np.testing.assert_array_equal(np.load(stored, allow_pickle=False), result['depth'])
                workflow.reset()
                tracks.reset_measurements()
                live = process_stereo_pair(model, estimator, tracks, workflow, args, 10, left, right)
                refresh_view(live, workflow, args, source, True)
                self.assertIsNone(live['depth'])
                self.assertFalse(live['summary']['depth_computed'])
                with contextlib.redirect_stdout(io.StringIO()):
                    save_current(directory / 'live', live)
                self.assertEqual(list((directory / 'live').glob('*_depth_m.npy')), [])
                summary = json.loads(next((directory / 'live').glob('*.json')).read_text())
                self.assertFalse(summary['workflow']['depth_computed'])
                for operation in (compute, compute_right, filtering, reproject):
                    self.assertEqual(operation.call_count, 7)


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
