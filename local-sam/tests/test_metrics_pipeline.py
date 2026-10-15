"""Stdlib-only tests for benchmark summaries and adaptive attempt accounting."""

from __future__ import annotations

import csv
import logging
import tempfile
import unittest
from pathlib import Path

from samlocal.metrics import stats
from samlocal.payload import Frame
from samlocal.pipeline import WindowSettings, run_sequence, valid_score
from samlocal.reporting import write_invalid_scores
from samlocal.stitch import Detection
from samlocal.backend import WindowOOMError


class FakeBackend:
    def __init__(self):
        self.fail_once = True
        self.current_peak = 0.5

    def reset_peak(self):
        self.current_peak = 0.5

    def peak(self):
        return {"peak_reserved_gib": self.current_peak, "peak_allocated_gib": self.current_peak * 0.8,
                "peak_fraction_of_budget": 0.2}

    def run_window(self, frame_dir, num_frames, prompts, *, frame_offset, on_prompt_done=None):
        if self.fail_once and num_frames > 2:
            self.fail_once = False
            self.current_peak = 2.0
            raise WindowOOMError("fake OOM")
        result = {}
        for label in prompts:
            result[label] = ([] if label == "empty" else
                             [Detection(frame_offset + i, 1, label, (0, 0, 2, 2), 0.5)
                              for i in range(num_frames) if i % 2 == 0])
            if on_prompt_done:
                on_prompt_done(label, 0.01)
        return result


class SentinelBackend(FakeBackend):
    """Every other detection carries SAM3's -10000 'no detector score' marker."""

    def __init__(self):
        super().__init__()
        self.fail_once = False

    def run_window(self, frame_dir, num_frames, prompts, *, frame_offset, on_prompt_done=None):
        return {label: [Detection(frame_offset + i, 1, label, (0, 0, 2, 2), -10000.0 if i % 2 else 0.8)
                        for i in range(num_frames)] for label in prompts}


class MetricsPipelineTests(unittest.TestCase):
    def run_sentinel(self, temp, frames=6, size=4):
        root = Path(temp)
        image = root / "0.jpg"
        image.write_bytes(b"frame")
        logs = []
        handler = logging.Handler()
        handler.emit = lambda record: logs.append(record.getMessage())
        logger = logging.getLogger("sentinel")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        result = run_sequence(SentinelBackend(), [Frame(i, image, 100 + i) for i in range(frames)], ["car"],
                              WindowSettings(size=size, min_size=2, max_size=size, overlap=2),
                              stage_root=root / "stage", logger=logger)
        logger.removeHandler(handler)
        return result, logs

    def test_sentinel_scores_are_dropped_and_recorded_once(self):
        with tempfile.TemporaryDirectory() as temp:
            result, _ = self.run_sentinel(temp)
        kept = [d for ds in result.frames.values() for d in ds]
        self.assertTrue(kept and all(0.0 <= d.score <= 1.0 for d in kept))
        recorded = [r["frame"] for r in result.invalid_scores]
        self.assertEqual(len(recorded), len(set(recorded)), "overlap frames must not be recorded twice")
        self.assertTrue(all(r["score"] == -10000.0 and r["label"] == "car" for r in result.invalid_scores))
        self.assertEqual(result.summary()["invalid_score_detections"], len(recorded))
        self.assertEqual(result.summary()["output_by_label"]["car"]["confidence"]["min"], 0.8)
        self.assertEqual(valid_score(float("nan")), False)

    def test_progress_percent_is_logged_and_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            result, logs = self.run_sentinel(temp)
        self.assertEqual(result.progress()["percent"], 100.0)
        self.assertEqual(result.progress()["frames_done"], 6)
        percents = [m for m in logs if m.startswith("progress ")]
        self.assertGreaterEqual(len(percents), 2)
        self.assertTrue(percents[0].startswith("progress 66.7% (4/6 frames)"))
        self.assertTrue(percents[-1].startswith("progress 100.0% (6/6 frames)"))

    def test_invalid_scores_csv_has_source_frame_index(self):
        with tempfile.TemporaryDirectory() as temp:
            result, _ = self.run_sentinel(temp)
            path = Path(temp) / "invalid_scores.csv"
            frames = [Frame(i, Path(temp) / "0.jpg", 100 + i) for i in range(6)]
            write_invalid_scores(path, 1, result.invalid_scores, frames)
            rows = list(csv.DictReader(path.open()))
        self.assertEqual(rows[0]["source_frame_index"], str(100 + int(rows[0]["frame"])))
        self.assertEqual(rows[0]["score"], "-10000.0")

    def test_non_oom_error_is_checkpointed_and_rethrown(self):
        class BrokenBackend(FakeBackend):
            def run_window(self, *args, **kwargs):
                raise RuntimeError("tracker failure")

        snapshots = []
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "0.jpg"
            image.write_bytes(b"frame")
            with self.assertRaisesRegex(RuntimeError, "tracker failure"):
                run_sequence(BrokenBackend(), [Frame(0, image, 0)], ["car"],
                             WindowSettings(size=2, min_size=2, max_size=2),
                             stage_root=root / "stage", logger=logging.getLogger("test"),
                             on_window=lambda result: snapshots.append(result.summary()))
            self.assertEqual(len(snapshots), 1)
            summary = snapshots[0]
            self.assertEqual(summary["failed_window_attempts"], 1)
            self.assertEqual(summary["oom_backoffs"], 0)
            self.assertEqual(summary["unique_frames_emitted"], 0)
            self.assertEqual(summary["window_attempt_records"][0]["error_type"], "RuntimeError")
            self.assertGreaterEqual(summary["total_seconds"], summary["backend_seconds"])
            self.assertFalse(any((root / "stage").glob("window-*")))

    def test_stats_uses_interpolated_quantiles_and_population_stddev(self):
        result = stats([0.0, 10.0])
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["p50"], 5.0)
        self.assertEqual(result["p95"], 9.5)
        self.assertEqual(result["stddev"], 5.0)
        self.assertIsNone(stats([])["p95"])

    def test_sequence_summary_includes_oom_attempt_and_zero_detection_frames(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = []
            for i in range(5):
                path = root / f"{i}.jpg"
                path.write_bytes(b"frame")
                paths.append(Frame(i, path, i))
            result = run_sequence(FakeBackend(), paths, ["object", "empty"],
                                  WindowSettings(size=4, min_size=2, max_size=4, overlap=1),
                                  stage_root=root / "stage", logger=logging.getLogger("test"))

        summary = result.summary()
        self.assertEqual(summary["window_attempts"], 4)
        self.assertEqual(summary["failed_window_attempts"], 1)
        self.assertEqual(summary["peak_reserved_gib"], 2.0)
        self.assertEqual(summary["unique_frames_emitted"], 5)
        self.assertEqual(summary["processed_window_frames"], 7)
        self.assertEqual(summary["detections_by_label"]["object"]["detections_per_frame"]["count"], 5)
        self.assertEqual(summary["prompt_seconds"]["object"]["count"], 3)
        self.assertEqual(summary["prompt_total_seconds"]["object"], 0.03)
        empty = summary["output_by_label"]["empty"]
        self.assertEqual(empty["detections"], 0)
        self.assertEqual(empty["detections_per_frame"], 0)
        self.assertEqual(empty["frames_with_detections"], 0)
        self.assertEqual(empty["zero_detection_frames"], 5)
        self.assertEqual(empty["tracks"], 0)
        self.assertEqual(empty["confidence"]["count"], 0)
        self.assertEqual(empty["track_length_frames"]["count"], 0)
        self.assertGreaterEqual(summary["window_attempt_records"][0]["staging_seconds"], 0)
        self.assertTrue(all("unique_frames" in w and "backend_seconds" in w
                            and "stitching_seconds" in w for w in summary["window_attempt_records"]))
        self.assertGreaterEqual(summary["amortized_ms_per_unique_frame"], 0)
        self.assertIn("not per-frame latency", summary["amortized_metric_note"])


if __name__ == "__main__":
    unittest.main()
