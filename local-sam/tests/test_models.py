"""SAM 3 vs SAM 3.1 wiring: specs, CLI surface, predictor build arguments and entry scripts."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from samlocal.backend import Sam3Backend, Sam31Backend
from samlocal.cli import add_common_args, build_backend
from samlocal.models import SAM3, SAM31, get_spec

ROOT = Path(__file__).resolve().parent.parent
LOG = logging.getLogger("test")


def parse(spec, *extra):
    parser = argparse.ArgumentParser()
    add_common_args(parser, spec)
    return parser, parser.parse_args(["--payload", "x", *extra])


class FakeCuda:
    OutOfMemoryError = type("OOM", (RuntimeError,), {})

    def is_available(self):
        return True

    def current_device(self):
        return 0

    def get_device_properties(self, device):
        return types.SimpleNamespace(name="Fake", total_memory=16 * 1024**3, major=8, minor=9)

    def set_per_process_memory_fraction(self, fraction, device):
        pass

    def reset_peak_memory_stats(self):
        pass

    def synchronize(self):
        pass

    def memory_allocated(self):
        return 3 * 1024**3

    def max_memory_reserved(self):
        return 4 * 1024**3


class SpecTests(unittest.TestCase):
    def test_specs_describe_each_model(self):
        self.assertEqual((SAM3.key, SAM3.hf_repo, SAM3.hf_checkpoint), ("sam3", "facebook/sam3", "sam3.pt"))
        self.assertEqual((SAM31.key, SAM31.hf_repo, SAM31.hf_checkpoint),
                         ("sam3.1", "facebook/sam3.1", "sam3.1_multiplex.pt"))
        self.assertIs(get_spec("sam3.1"), SAM31)
        with self.assertRaises(ValueError):
            get_spec("sam2")


class CliTests(unittest.TestCase):
    def test_sam3_exposes_only_options_it_supports(self):
        parser, args = parse(SAM3, "--offload-state")
        self.assertTrue(args.offload_state)
        for flag in ("--max-objects", "--multiplex-count", "--grounding-batch", "--compile"):
            with self.assertRaises(SystemExit), patch("sys.stderr"):
                parser.parse_args(["--payload", "x", flag, "1"] if flag != "--compile" else ["--payload", "x", flag])

    def test_sam31_exposes_multiplex_options(self):
        parser, args = parse(SAM31, "--grounding-batch", "1", "--max-objects", "8", "--compile")
        self.assertEqual((args.grounding_batch, args.max_objects, args.compile), (1, 8, True))
        with self.assertRaises(SystemExit), patch("sys.stderr"):
            parser.parse_args(["--payload", "x", "--offload-state"])

    def test_build_backend_picks_the_class_and_forwards_options(self):
        _, args = parse(SAM3, "--offload-state", "--vram-fraction", "0.8")
        backend = build_backend(args, LOG, SAM3)
        self.assertIsInstance(backend, Sam3Backend)
        self.assertTrue(backend.offload_state)
        self.assertEqual(backend.vram_fraction, 0.8)

        _, args = parse(SAM31, "--grounding-batch", "1")
        backend = build_backend(args, LOG, SAM31)
        self.assertIsInstance(backend, Sam31Backend)
        self.assertEqual(backend.grounding_batch, 1)


class BuildArgumentTests(unittest.TestCase):
    def test_sam3_build_kwargs_exclude_multiplex_only_options(self):
        kwargs = Sam3Backend(checkpoint=Path("/m/sam3.pt")).build_kwargs(types.SimpleNamespace(major=9))
        self.assertEqual(kwargs, {"checkpoint_path": "/m/sam3.pt", "version": "sam3", "compile": False,
                                  "async_loading_frames": True})

    def test_sam31_build_kwargs_keep_multiplex_options(self):
        backend = Sam31Backend(checkpoint=None, max_objects=8)
        kwargs = backend.build_kwargs(types.SimpleNamespace(major=8))
        self.assertEqual(kwargs["version"], "sam3.1")
        self.assertEqual((kwargs["max_num_objects"], kwargs["multiplex_count"], kwargs["use_fa3"]), (8, 16, False))

    def run_load(self, backend, predictor):
        builder = types.ModuleType("sam3.model_builder")
        seen = {}

        def build_sam3_predictor(**kwargs):
            seen.update(kwargs)
            return predictor

        builder.build_sam3_predictor = build_sam3_predictor
        torch = types.SimpleNamespace(cuda=FakeCuda())
        with patch.dict(sys.modules, {"torch": torch, "sam3": types.ModuleType("sam3"),
                                      "sam3.model_builder": builder}):
            return backend.load(), seen

    def test_sam3_load_never_touches_the_multiplex_tracker(self):
        class Predictor:
            @property
            def model(self):
                raise AssertionError("SAM 3 load must not inspect predictor.model")

        info, seen = self.run_load(Sam3Backend(checkpoint=None, logger=LOG, offload_state=True), Predictor())

        self.assertEqual(seen["version"], "sam3")
        self.assertNotIn("max_num_objects", seen)
        self.assertEqual(info["model_version"], "sam3")
        self.assertTrue(info["offload_state_to_cpu"])
        self.assertNotIn("shared_conditioning_guard", info)
        self.assertNotIn("batched_grounding", info)

    def test_sam31_load_installs_guard_and_caps_grounding_batch(self):
        class Tracker:
            def clear_all_points_in_frame(self, inference_state, frame_idx, obj_id):
                pass

            def _reset_tracking_results(self, inference_state):
                pass

        model = types.SimpleNamespace(tracker=Tracker(), use_batched_grounding=True, batched_grounding_batch_size=16)
        info, seen = self.run_load(Sam31Backend(checkpoint=None, logger=LOG, grounding_batch=2),
                                   types.SimpleNamespace(model=model))

        self.assertEqual(seen["version"], "sam3.1")
        self.assertEqual(info["model_version"], "sam3.1")
        self.assertTrue(info["shared_conditioning_guard"]["installed"])
        self.assertEqual(model.batched_grounding_batch_size, 2)

    def test_sam3_session_request_carries_offload_state(self):
        requests = []

        class Predictor:
            def handle_request(self, request):
                requests.append(request)
                return {"session_id": "s"}

        backend = Sam3Backend(checkpoint=None, logger=LOG, offload_state=True)
        backend.predictor = Predictor()
        backend._start_session(Path("frames"))
        self.assertTrue(requests[0]["offload_state_to_cpu"])


class EntryScriptTests(unittest.TestCase):
    SCRIPTS = {"sam3": "sam3", "sam3.1": "sam31"}

    def test_every_script_has_help_and_the_right_model_options(self):
        for folder, slug in self.SCRIPTS.items():
            for kind in ("benchmark", "run", "healthcheck"):
                script = ROOT / folder / f"{kind}_{slug}.py"
                self.assertTrue(script.is_file(), script)
                done = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True,
                                      cwd=ROOT / folder, timeout=60)
                self.assertEqual(done.returncode, 0, done.stderr)
                self.assertIn(f"{kind}_{slug}.py", done.stdout)
                if kind == "benchmark":
                    self.assertEqual("--grounding-batch" in done.stdout, folder == "sam3.1")
                    self.assertEqual("--offload-state" in done.stdout, folder == "sam3")


if __name__ == "__main__":
    unittest.main()
