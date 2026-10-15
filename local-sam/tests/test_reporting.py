from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from samlocal import benchmark as benchmark_module
from samlocal.payload import Frame, Payload
from samlocal.reporting import ResourceSampler, input_identity
from samlocal.stitch import Detection
from samlocal.backend import WindowOOMError


class Backend:
    def load(self):
        return {"model_load_seconds": 0.01}

    def reset_peak(self):
        pass

    def peak(self):
        return {"peak_allocated_gib": 1.0, "peak_reserved_gib": 2.0, "peak_fraction_of_budget": 0.8}

    def run_window(self, directory, count, prompts, *, frame_offset, on_prompt_done):
        result = {}
        for label in prompts:
            result[label] = [Detection(frame_offset + i, 1, label, (0, 0, 2, 2), 0.8)
                             for i in range(count) if label == "car"]
            on_prompt_done(label, 0.001)
        return result

    def close(self):
        pass


class ReportingTests(unittest.TestCase):
    def test_identity_is_path_independent_and_sensitive_to_content_and_source_index(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            a, b = root / "a.jpg", root / "renamed.jpg"
            a.write_bytes(b"image")
            b.write_bytes(b"image")
            original = input_identity([Frame(0, a, 10)])["sha256"]
            self.assertEqual(original, input_identity([Frame(0, b, 10)])["sha256"])
            self.assertNotEqual(original, input_identity([Frame(0, b, 11)])["sha256"])
            b.write_bytes(b"different")
            self.assertNotEqual(original, input_identity([Frame(0, b, 10)])["sha256"])

    def test_sampler_disabled_keeps_unavailable_metrics_null(self):
        with patch("samlocal.reporting.subprocess.run") as command:
            result = ResourceSampler(0).start().stop()
        command.assert_not_called()
        self.assertFalse(result["enabled"])
        self.assertEqual(result["samples"], [])
        self.assertIsNone(result["metrics"]["gpu_utilization_pct"]["mean"])

    def test_sampler_missing_nvidia_smi_is_nonfatal(self):
        with patch("samlocal.reporting.subprocess.run", side_effect=FileNotFoundError("missing")):
            sampler = ResourceSampler(0.001).start()
            # Wait for the sampler's first poll, without running a GPU model.
            for _ in range(100):
                if sampler.samples:
                    break
                sampler._stop.wait(0.005)
            result = sampler.stop()
        self.assertIn("gpu", result["errors"])
        self.assertIsNone(result["metrics"]["gpu_power_w"]["max"])

    def test_benchmark_exports_integrated_repeat_window_class_frame_csv(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frames = []
            for i in range(5):
                path = root / f"{i}.jpg"
                path.write_bytes(b"image")
                frames.append(Frame(i, path, i * 2, 640, 360))
            payload = Payload(root, frames, ["car", "truck"])
            output = root / "logs"
            output.mkdir()
            arguments = ["benchmark_sam31.py", "--payload", str(root), "--cache-dir", str(root / "cache"),
                         "--warmup", "0", "--repeats", "2", "--window-size", "3", "--min-window", "3",
                         "--max-window", "3", "--overlap", "1", "--telemetry-interval", "0",
                         "--save-predictions", "--run-label", "test-run"]
            env = {"gpu": {"name": "fake GPU"}, "packages": {"torch": "fake"}}
            with patch("sys.argv", arguments), patch.object(benchmark_module, "environment", return_value=env), \
                 patch.object(benchmark_module, "open_payload", return_value=payload), \
                 patch.object(benchmark_module, "create_run_dir", return_value=output), \
                 patch.object(benchmark_module, "build_backend", return_value=Backend()):
                self.assertEqual(benchmark_module.main("sam3.1"), 0)
            report = json.loads((output / "benchmark.json").read_text())
            self.assertEqual(report["schema"], "local-sam-benchmark/v3")
            self.assertTrue(report["completed"])
            self.assertEqual(report["last_completed_repeat"], 2)
            for repeat in report["repeats"]:
                self.assertEqual(repeat["unique_frames_emitted"], 5)
                self.assertEqual(repeat["processed_window_frames"], 6)
                self.assertEqual(repeat["output_by_label"]["truck"]["zero_detection_frames"], 5)
                self.assertEqual(repeat["output_by_label"]["car"]["detections"], 5)
                self.assertIn("car", repeat["prompt_seconds"])
            def rows(name):
                with (output / name).open(newline="") as handle:
                    return list(csv.DictReader(handle))
            self.assertEqual(len(rows("summary.csv")), 2)
            self.assertEqual(len(rows("windows.csv")), 4)
            self.assertEqual(len(rows("attempts.csv")), 4)
            self.assertEqual(len(rows("classes.csv")), 4)
            frame_rows = rows("frames.csv")
            self.assertEqual(len(frame_rows), 10)
            self.assertTrue(all(row["repeat"] == "2" for row in frame_rows))
            self.assertEqual(frame_rows[-1]["source_frame_index"], "8")
            self.assertEqual(rows("resources.csv"), [])
            self.assertTrue((output / "predictions.json").is_file())
            self.assertIsNone(report["quality_metrics"]["mask_map"])

    def test_sam3_benchmark_reports_its_own_model_and_run_dir_name(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "0.jpg"
            image.write_bytes(b"image")
            payload = Payload(root, [Frame(0, image, 0, 640, 360)], ["car"])
            output = root / "logs"
            output.mkdir()
            env = {"gpu": {"name": "fake GPU"}, "packages": {"torch": "fake"}}
            argv = ["--payload", str(root), "--cache-dir", str(root / "cache"), "--warmup", "0", "--repeats", "1",
                    "--window-size", "2", "--min-window", "2", "--max-window", "2", "--telemetry-interval", "0",
                    "--offload-state"]
            with patch.object(benchmark_module, "environment", return_value=env), \
                 patch.object(benchmark_module, "open_payload", return_value=payload), \
                 patch.object(benchmark_module, "create_run_dir", return_value=output) as create, \
                 patch.object(benchmark_module, "build_backend", return_value=Backend()) as build:
                self.assertEqual(benchmark_module.main("sam3", argv), 0)
            create.assert_called_once()
            self.assertEqual(create.call_args.args[1], "benchmark_sam3")
            self.assertEqual(build.call_args.args[2].key, "sam3")
            report = json.loads((output / "benchmark.json").read_text())
            self.assertEqual(report["model_version"], "sam3")
            self.assertEqual(report["run_label"], "sam3")
            self.assertNotIn("multiplex_count", report["model"])
            self.assertEqual(report["model"]["checkpoint"], "facebook/sam3/sam3.pt")
            self.assertTrue(report["args"]["offload_state"])
            self.assertNotIn("grounding_batch", report["args"])
            with (output / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["model_version"], "sam3")
            self.assertEqual(row["offload_state"], "True")

    def test_terminal_oom_is_saved_as_incomplete_attempt_with_resources(self):
        class FailingBackend(Backend):
            def run_window(self, *args, **kwargs):
                raise WindowOOMError("terminal fake OOM")

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "0.jpg"
            image.write_bytes(b"image")
            payload = Payload(root, [Frame(0, image, 0, 640, 360)], ["car"])
            output = root / "logs"
            output.mkdir()
            arguments = ["benchmark_sam31.py", "--payload", str(root), "--cache-dir", str(root / "cache"),
                         "--warmup", "0", "--repeats", "1", "--window-size", "2", "--min-window", "2",
                         "--max-window", "2", "--telemetry-interval", "0"]
            env = {"gpu": {"name": "fake GPU"}, "packages": {"torch": "fake"}}
            with patch("sys.argv", arguments), patch.object(benchmark_module, "environment", return_value=env), \
                 patch.object(benchmark_module, "open_payload", return_value=payload), \
                 patch.object(benchmark_module, "create_run_dir", return_value=output), \
                 patch.object(benchmark_module, "build_backend", return_value=FailingBackend()):
                self.assertEqual(benchmark_module.main("sam3.1"), 1)
            report = json.loads((output / "benchmark.json").read_text())
            self.assertFalse(report["completed"])
            repeat = report["repeats"][0]
            self.assertFalse(repeat["completed"])
            self.assertEqual(repeat["failed_window_attempts"], 1)
            self.assertEqual(repeat["scheduler_events"][-1]["type"], "oom_at_min")
            self.assertEqual(repeat["unique_frames_emitted"], 0)
            self.assertIsNone(repeat["output_by_label"]["car"]["detections_per_frame"])
            self.assertEqual(repeat["output_by_label"]["car"]["zero_detection_frames"], 0)
            self.assertIn("resources", repeat)
            with (output / "attempts.csv").open(newline="") as handle:
                attempts = list(csv.DictReader(handle))
            self.assertEqual(len(attempts), 1)
            self.assertEqual(attempts[0]["success"], "False")
            self.assertEqual(attempts[0]["peak_allocated_gib"], "1.0")


if __name__ == "__main__":
    unittest.main()
