"""Geometry/depth/association regression tests; no model, camera or network needed.

Run: .venv/Scripts/python.exe -m unittest discover -s Python -p test_perch_video_seg.py -v
"""

from pathlib import Path
import contextlib
import io
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import cv2
import numpy as np

from perch_video_seg import (MaskTracks, Measurement, SegmentDepth, VideoPair,
                             ROOT, PerchBurst, PerchWorkflow, alignment_hint,
                             build_parser, clean_mask, compose_view, infer, measure_mask)
from video_depth_estimation import StereoDepth, depth_to_color


def bar(shape=(240, 320), center=(160, 120), size=(160, 20), angle=0):
    mask = np.zeros(shape, np.uint8)
    points = cv2.boxPoints((center, size, angle)).round().astype(np.int32)
    cv2.fillPoly(mask, [points], 1)
    return mask


def detection(mask, z, class_id=0):
    return dict(mask=mask, class_id=class_id,
                measurement=Measurement(z, 0.5 if z else None, 0.05 if z else None))


class MaskMeasurementTests(unittest.TestCase):
    def test_background_is_excluded(self):
        mask = bar(angle=35)
        # Most of the axis-aligned box contains background at a different Z.
        depth = np.full(mask.shape, 8.0, dtype=np.float32)
        depth[mask > 0] = 1.0
        result, _ = measure_mask(mask, depth, 400)
        self.assertAlmostEqual(result.distance, 1.0)
        self.assertAlmostEqual(result.length, 0.40, delta=0.008)
        self.assertAlmostEqual(result.width, 0.05, delta=0.008)

    def test_rotated_bar_keeps_thickness(self):
        depth = np.ones((240, 320), np.float32)
        flat, _ = measure_mask(bar(), depth, 400)
        tilted, _ = measure_mask(bar(angle=45), depth, 400)
        self.assertAlmostEqual(flat.width, tilted.width, delta=0.008)
        self.assertAlmostEqual(flat.length, tilted.length, delta=0.008)

    def test_outliers_do_not_change_distance(self):
        mask = bar()
        depth = np.ones(mask.shape, np.float32)
        depth[112:114, 90:230] = 9.0
        result, _ = measure_mask(mask, depth, 400)
        self.assertAlmostEqual(result.distance, 1.0)

    def test_missing_or_sparse_depth_remains_unknown(self):
        mask = bar()
        depth = np.full(mask.shape, np.nan, dtype=np.float32)
        result, _ = measure_mask(mask, depth, 400)
        self.assertIsNone(result.values())
        depth[118:122, 150:160] = 1.0
        result, _ = measure_mask(mask, depth, 400)
        self.assertIsNone(result.values())

    def test_large_depth_spread_is_rejected(self):
        mask = bar()
        depth = np.ones(mask.shape, np.float32)
        depth[:, 160:] = 2.0
        result, _ = measure_mask(mask, depth, 400)
        self.assertIsNone(result.values())

    def test_thin_mask_depth_core_survives(self):
        mask = bar(size=(160, 2))
        result, _ = measure_mask(mask, np.ones(mask.shape, np.float32), 400)
        self.assertAlmostEqual(result.distance, 1.0)

    def test_cleanup_keeps_holes_removes_tiny_islands(self):
        mask = bar()
        mask[116:124, 140:180] = 0
        mask[1, 1] = 1
        cleaned = clean_mask(mask, 16)
        self.assertEqual(int(cleaned[1, 1]), 0)
        self.assertEqual(int(cleaned[120, 160]), 0)
        self.assertEqual(int(cleaned[112, 160]), 1)


class TrackingTests(unittest.TestCase):
    def test_separate_objects_and_detection_order(self):
        tracker = MaskTracks(min_samples=3)
        mask_a = bar(center=(90, 50), size=(100, 10))
        mask_b = bar(center=(210, 180), size=(100, 10))
        for frame in range(3):
            detections = [detection(mask_a, 1.0), detection(mask_b, 4.0)]
            if frame % 2:
                detections.reverse()
            tracker.update(detections, frame)
        by_id = {d["track_id"]: d for d in detections}
        self.assertEqual(by_id[1]["stable"][0], 1.0)
        self.assertEqual(by_id[2]["stable"][0], 4.0)

    def test_invalid_current_depth_does_not_reuse_old_verdict(self):
        tracker = MaskTracks(min_samples=3)
        mask = bar()
        for frame in range(3):
            detections = [detection(mask, 1.0)]
            tracker.update(detections, frame)
        self.assertTrue(detections[0]["verdict"])
        detections = [detection(mask, None)]
        tracker.update(detections, 3)
        self.assertIsNone(detections[0]["verdict"])
        self.assertIsNone(detections[0]["stable"])

    def test_minimum_history_and_class_isolation(self):
        tracker = MaskTracks(min_samples=3)
        mask = bar()
        for frame in range(2):
            detections = [detection(mask, 1.0)]
            tracker.update(detections, frame)
            self.assertIsNone(detections[0]["verdict"])
        detections = [detection(mask, 1.0, class_id=1)]
        tracker.update(detections, 2)
        self.assertEqual(detections[0]["track_id"], 2)
        self.assertIsNone(detections[0]["stable"])


class AlignmentAndBurstTests(unittest.TestCase):
    def make_detection(self, identity=1, z=1.0):
        return dict(detection(bar(), z), track_id=identity, confidence=0.9)

    def test_level_centered_bar_ready_and_tilt_has_correct_correction(self):
        self.assertTrue(alignment_hint(bar())["ready"])
        for angle, direction in ((20, "CCW"), (-20, "CW")):
            guide = alignment_hint(bar(angle=angle))
            self.assertFalse(guide["ready"])
            self.assertAlmostEqual(guide["angle_deg"], angle, delta=1.0)
            self.assertIn(direction, guide["messages"][0])

    def test_centering_clipping_and_missing_target(self):
        guide = alignment_hint(bar(center=(250, 120), size=(80, 10)))
        self.assertIn("RIGHT", " ".join(guide["messages"]))
        clipped = alignment_hint(bar(center=(20, 120)))
        self.assertIn("outside", " ".join(clipped["messages"]))
        self.assertFalse(clipped["ready"])
        self.assertIsNone(alignment_hint(None)["angle_deg"])

    def test_preview_does_not_measure_physical_sizes(self):
        class Tensor:
            def __init__(self, values):
                self.values = values
            def __len__(self):
                return len(self.values)
            def cpu(self):
                return self
            def numpy(self):
                return self.values
        class Boxes:
            data = Tensor(np.array([[80, 110, 240, 130, .9, 0]], np.float32))
            def __len__(self):
                return 1
            def cpu(self):
                return self
        pred = SimpleNamespace(boxes=Boxes(), masks=SimpleNamespace(data=Tensor(bar()[None])))
        model = SimpleNamespace(names={0: "Perch"}, predict=lambda **kwargs: [pred])
        with patch("perch_video_seg.measure_mask", side_effect=AssertionError("Preview measured depth")):
            output = infer(model, np.zeros((240, 320, 3), np.uint8), None, 400,
                           build_parser().parse_args([]))
        self.assertIsNone(output[0]["measurement"].values())
        self.assertIsNotNone(output[0]["oriented"])

    def test_live_then_trigger_collects_only_new_target_samples_and_freezes(self):
        workflow = PerchWorkflow()
        for _ in range(5):
            det = self.make_detection(z=4.0)
            workflow.observe([det])
            self.assertIsNone(det["verdict"])
            self.assertIsNone(workflow.result)
        workflow.target_id = 1
        self.assertTrue(workflow.trigger())
        for frame in range(7):
            detections = [self.make_detection(1, 10.0 if frame == 3 else 1.0),
                          self.make_detection(2, 4.0)]
            workflow.observe(detections)
            if frame < 6:
                self.assertEqual(workflow.state, "CAPTURING")
                self.assertIsNone(detections[0]["verdict"])
        self.assertEqual(workflow.state, "RESULT")
        self.assertTrue(workflow.paused)
        self.assertEqual(workflow.result["values"][0], 1.0)
        self.assertTrue(workflow.result["verdict"])
        self.assertIsNone(detections[1]["verdict"])
        self.assertTrue(workflow.trigger())
        self.assertEqual(workflow.burst.samples, [])
        self.assertFalse(workflow.trigger())  # repeated T cannot restart a running capture
        workflow.reset()
        self.assertEqual(workflow.state, "LIVE")
        self.assertFalse(workflow.paused)
        self.assertIsNone(workflow.result)

    def test_lost_target_does_not_switch_to_another_bar(self):
        burst = PerchBurst(1, requested=3, min_samples=2)
        burst.add([self.make_detection(1)])
        burst.add([self.make_detection(1)])
        burst.add([self.make_detection(2, 4.0)])
        self.assertIsNone(burst.result()["verdict"])
        self.assertEqual(burst.result()["target_id"], 1)
        self.assertEqual(burst.result()["valid_frames"], 2)

    def test_out_of_range_target_cannot_land(self):
        workflow = PerchWorkflow()
        workflow.target_id = 1
        workflow.trigger(3, 2)
        for _ in range(3):
            workflow.observe([self.make_detection(1, 8.0)])
        self.assertFalse(workflow.result["verdict"])
        self.assertIn("distance", workflow.result["reason"])

    def test_incomplete_and_missing_depth_bursts_remain_unknown(self):
        burst = PerchBurst(1, requested=7, min_samples=3)
        for _ in range(3):
            burst.add([self.make_detection(1)])
        self.assertIsNone(burst.result()["verdict"])
        for _ in range(4):
            burst.add([self.make_detection(1, None)])
        # Three real samples suffice, as in a median burst, without reusing preview data.
        self.assertTrue(burst.result()["verdict"])
        missing = PerchBurst(1, requested=3, min_samples=2)
        for _ in range(3):
            missing.add([self.make_detection(1, None)])
        self.assertIsNone(missing.result()["verdict"])

    def test_two_eye_layout_always_has_four_equal_panels(self):
        left = np.full((240, 320, 3), (25, 0, 0), np.uint8)
        right = np.full_like(left, (0, 25, 0))
        depth = np.full_like(left, (0, 0, 25))
        view, *_ = compose_view(left, right, depth, [], [], alignment_hint(None),
                                "LIVE", 0, 100, 2.0, None)
        self.assertEqual(view.shape, (480, 640, 3))
        np.testing.assert_array_equal(view[220, 300], (25, 0, 0))
        np.testing.assert_array_equal(view[220, 620], (0, 25, 0))
        np.testing.assert_array_equal(view[460, 300], (0, 0, 25))


class StereoGeometryTests(unittest.TestCase):
    def test_depth_method_and_defaults_are_the_reference(self):
        self.assertIs(SegmentDepth.process, StereoDepth.process)
        self.assertEqual(build_parser().parse_args([]).downscale, 0.5)

    def test_real_video_outputs_exactly_match_reference_at_both_scales(self):
        paths = (ROOT / "Python/l1.mp4", ROOT / "Python/r1.mp4")
        if not all(path.is_file() for path in paths):
            self.skipTest("Provided stereo videos unavailable")
        caps = [cv2.VideoCapture(str(path)) for path in paths]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                for scale in (0.5, 1.0):
                    args = build_parser().parse_args(["--downscale", str(scale)])
                    actual = SegmentDepth(args)
                    reference = StereoDepth(str(args.calib), downscale=scale)
                    for frame_index in (0, 187, 956):
                        with self.subTest(scale=scale, frame=frame_index):
                            frames = []
                            for cap in caps:
                                self.assertTrue(cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index))
                                ok, frame = cap.read()
                                self.assertTrue(ok)
                                frames.append(frame)
                            observed = actual.process(*frames)
                            expected = reference.process(*frames)
                            for result, target in zip(observed, expected):
                                self.assertEqual(result.dtype, target.dtype)
                                np.testing.assert_array_equal(result, target)
                            np.testing.assert_array_equal(
                                depth_to_color(observed[2], 5.0, "mm"),
                                depth_to_color(expected[2], 5.0, "mm"))
        finally:
            for cap in caps:
                cap.release()

    def test_original_mask_uses_calibration_resize_and_remap(self):
        # Work at 2x video resolution and remap x -> x+10 at calibration scale.
        estimator = SegmentDepth.__new__(SegmentDepth)
        estimator.size = (100, 60)
        yy, xx = np.indices((60, 100), dtype=np.float32)
        estimator.maps = [xx + 10, yy]
        source_mask = np.zeros((120, 200), dtype=np.uint8)
        source_mask[40:80, 80:120] = 1
        projected = estimator.rectify_mask(source_mask)
        expected = np.zeros((60, 100), dtype=np.uint8)
        expected[20:40, 30:50] = 1
        np.testing.assert_array_equal(projected, expected)

    def test_known_disparity_depth_and_downscale_units(self):
        """A 20px shift, f=200px and baseline=60mm must produce Z=0.6m."""
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "calib.npz"
            yy, xx = np.indices((160, 320), dtype=np.float32)
            q = np.array([[1, 0, 0, -160], [0, 1, 0, -80],
                          [0, 0, 0, 200], [0, 0, 1 / 60, 0]], dtype=np.float64)
            np.savez(path, left_map_x=xx, left_map_y=yy, right_map_x=xx, right_map_y=yy, Q=q)
            rng = np.random.default_rng(42)
            left = rng.integers(0, 256, (160, 320, 3), dtype=np.uint8)
            right = np.zeros_like(left)
            right[:, :-20] = left[:, 20:]
            for scale in (1.0, 0.5):
                with self.subTest(scale=scale):
                    args = build_parser().parse_args([])
                    args.calib, args.downscale = path, scale
                    est = SegmentDepth(args, (320, 160))
                    _, _, depth = est.process(left, right)
                    core = depth[30:130, 160:280]
                    valid = core[np.isfinite(core) & (core > 0)]
                    self.assertGreater(valid.size, 5000)
                    self.assertAlmostEqual(float(np.median(valid)), 600.0, delta=15.0)
                    self.assertAlmostEqual(float(np.median(valid * 0.001)), 0.6, delta=0.015)


class VideoPairTests(unittest.TestCase):
    def make_video(self, path, count, fps=30):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
        if not writer.isOpened():
            self.skipTest("mp4v encoder unavailable")
        try:
            for index in range(count):
                writer.write(np.full((48, 64, 3), index * 35, np.uint8))
        finally:
            writer.release()

    def test_shorter_stream_ends_both_and_offsets_restart_together(self):
        with tempfile.TemporaryDirectory() as temp:
            left, right = Path(temp) / "left.mp4", Path(temp) / "right.mp4"
            self.make_video(left, 5)
            self.make_video(right, 3)
            source = VideoPair(left, right, (1, 0))
            try:
                self.assertEqual(source.total, 3)
                first = source.read()
                self.assertEqual(first[0], 0)
                self.assertGreater(float(first[1].mean()), float(first[2].mean()) + 20)
                self.assertIsNotNone(source.read())
                self.assertIsNotNone(source.read())
                self.assertIsNone(source.read())
                source.rewind()
                again = source.read()
                np.testing.assert_array_equal(first[1], again[1])
                np.testing.assert_array_equal(first[2], again[2])
            finally:
                source.release()

    def test_mismatched_fps_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            left, right = Path(temp) / "left.mp4", Path(temp) / "right.mp4"
            self.make_video(left, 3, 30)
            self.make_video(right, 3, 25)
            with self.assertRaisesRegex(RuntimeError, "Different FPS"):
                VideoPair(left, right, (0, 0))


if __name__ == "__main__":
    unittest.main()
