from __future__ import annotations

import logging
import sys
import types
import unittest
import weakref
from pathlib import Path

from samlocal.backend import Sam31Backend, WindowOOMError, configure_grounding_batch


class FakeOOM(RuntimeError):
    pass


class GpuTensor:
    """Stands in for a CUDA tensor; tests watch its lifetime through a weakref."""


class FakeCuda:
    OutOfMemoryError = FakeOOM

    def __init__(self) -> None:
        self.watch: weakref.ref | None = None
        self.alive_at_empty_cache: list[bool] = []

    def is_available(self) -> bool:
        return True

    def synchronize(self) -> None:
        pass

    def empty_cache(self) -> None:
        if self.watch is not None:
            self.alive_at_empty_cache.append(self.watch() is not None)


class FakePredictor:
    def __init__(self, cuda: FakeCuda, *, error: Exception) -> None:
        self.cuda = cuda
        self.error = error
        self.closed: list[str] = []

    def handle_request(self, request):
        if request["type"] == "start_session":
            return {"session_id": "s1"}
        if request["type"] == "close_session":
            self.closed.append(request["session_id"])
            return {"is_success": True}
        return {"outputs": {}}

    def handle_stream_request(self, request):
        activations = GpuTensor()  # held by the generator frame, like propagation tensors
        self.cuda.watch = weakref.ref(activations)
        yield {"frame_index": 0, "outputs": {}}
        raise self.error


class BackendMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cuda = FakeCuda()
        self.fake_torch = types.SimpleNamespace(cuda=self.cuda, OutOfMemoryError=FakeOOM)
        self.saved_torch = sys.modules.get("torch")
        sys.modules["torch"] = self.fake_torch

    def tearDown(self) -> None:
        if self.saved_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = self.saved_torch

    def make_backend(self, error: Exception) -> tuple[Sam31Backend, FakePredictor]:
        backend = Sam31Backend(checkpoint=None, logger=logging.getLogger("test"))
        backend.torch = self.fake_torch
        backend.predictor = FakePredictor(self.cuda, error=error)
        return backend, backend.predictor

    def test_oom_releases_generator_tensors_before_final_cleanup(self):
        backend, predictor = self.make_backend(FakeOOM("CUDA out of memory. Tried to allocate 1.27 GiB"))

        with self.assertRaises(WindowOOMError) as caught:
            backend.run_window(Path("unused"), 4, ["car"], frame_offset=0)

        self.assertEqual(str(caught.exception), "CUDA out of memory. Tried to allocate 1.27 GiB")
        self.assertIsNone(caught.exception.__cause__)
        self.assertTrue(caught.exception.__suppress_context__ or caught.exception.__context__ is None)
        self.assertEqual(predictor.closed, ["s1"])
        self.assertTrue(self.cuda.alive_at_empty_cache)
        self.assertFalse(self.cuda.alive_at_empty_cache[-1])

    def test_non_oom_error_is_rethrown_unchanged(self):
        backend, predictor = self.make_backend(RuntimeError("No points are provided; please add points first"))

        with self.assertRaisesRegex(RuntimeError, "No points are provided"):
            backend.run_window(Path("unused"), 4, ["car"], frame_offset=0)

        self.assertEqual(predictor.closed, ["s1"])

    def test_grounding_batch_must_be_positive(self):
        with self.assertRaises(ValueError):
            Sam31Backend(checkpoint=None, grounding_batch=0)


class GroundingBatchTests(unittest.TestCase):
    def test_overrides_upstream_chunk_size(self):
        model = types.SimpleNamespace(use_batched_grounding=True, batched_grounding_batch_size=16)

        info = configure_grounding_batch(model, 2)

        self.assertEqual(model.batched_grounding_batch_size, 2)
        self.assertTrue(info["applied"])
        self.assertEqual(info["upstream_batch_size"], 16)
        self.assertEqual(info["batch_size"], 2)

    def test_reports_when_model_has_no_batched_grounding(self):
        model = types.SimpleNamespace()

        info = configure_grounding_batch(model, 2)

        self.assertFalse(info["applied"])
        self.assertFalse(hasattr(model, "batched_grounding_batch_size"))


if __name__ == "__main__":
    unittest.main()
