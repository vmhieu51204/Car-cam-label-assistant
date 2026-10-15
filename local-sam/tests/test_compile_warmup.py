from __future__ import annotations

import types
import unittest

from samlocal.backend import Sam31Backend


class FakeModel:
    def __init__(self) -> None:
        self.batched_grounding_batch_size = 16  # upstream hard-coded chunk
        self.tracker = object()
        self.warm_up_batch: int | None = None
        self._warm_up_complete = True

    def warm_up_compilation(self) -> None:
        assert self._warm_up_complete is False
        self.warm_up_batch = self.batched_grounding_batch_size


class CompileWarmUpTest(unittest.TestCase):
    def _backend(self, compile_model: bool) -> tuple[Sam31Backend, FakeModel]:
        backend = Sam31Backend(checkpoint=None, compile_model=compile_model, grounding_batch=4)
        model = FakeModel()
        backend.predictor = types.SimpleNamespace(model=model)
        backend.torch = types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=lambda: None))
        return backend, model

    def test_predictor_is_built_without_its_own_warm_up(self) -> None:
        backend, _ = self._backend(True)
        props = types.SimpleNamespace(major=8)
        self.assertIs(backend.build_kwargs(props)["warm_up"], False)
        self.assertIs(backend.build_kwargs(props)["compile"], True)

    def test_warm_up_runs_after_grounding_batch_is_capped(self) -> None:
        backend, model = self._backend(True)
        info = backend.configure()
        self.assertEqual(model.warm_up_batch, 4)  # not upstream's 16, which ran out of memory on 16 GB
        self.assertTrue(model._warm_up_complete)
        self.assertIsNotNone(info["compile_warm_up_seconds"])

    def test_no_warm_up_without_compile(self) -> None:
        backend, model = self._backend(False)
        info = backend.configure()
        self.assertIsNone(model.warm_up_batch)
        self.assertIsNone(info["compile_warm_up_seconds"])


if __name__ == "__main__":
    unittest.main()
