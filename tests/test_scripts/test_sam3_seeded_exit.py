"""An exited object cannot be re-seeded from a stale box in a later window."""

from __future__ import annotations

import contextlib
import importlib
import sys
import types

import numpy as np


def test_seeded_tracker_does_not_reprompt_after_empty_mask(monkeypatch, tmp_path):
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    fake_torch.inference_mode = contextlib.nullcontext
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class Mask:
        def __init__(self, pixels):
            self.pixels = pixels

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.pixels

    class Tracker:
        def __init__(self):
            self.prompts = []
            self.states = 0

        def init_state(self, *, video_path):
            self.states += 1
            return {}

        def add_new_points_or_box(self, *, obj_id, **kwargs):
            self.prompts.append(obj_id)

        def propagate_in_video(self, state, **kwargs):
            visible = np.zeros((4, 4), dtype=bool)
            visible[1:3, 1:3] = True
            yield 0, [1], None, [Mask(visible)], [None]
            yield 1, [1], None, [Mask(np.zeros((4, 4), dtype=bool))], [None]

    tracker = Tracker()
    model = types.SimpleNamespace(tracker=tracker, detector=types.SimpleNamespace(backbone=object()))
    sam3 = types.ModuleType("sam3")
    sam3.__path__ = []
    builder = types.ModuleType("sam3.model_builder")
    builder.build_sam3_video_model = lambda **kwargs: model
    monkeypatch.setitem(sys.modules, "sam3", sam3)
    monkeypatch.setitem(sys.modules, "sam3.model_builder", builder)
    sys.modules.pop("kaggle.sam3.main", None)
    module = importlib.import_module("kaggle.sam3.main")
    monkeypatch.setattr(module, "CHUNK_FRAMES", 2)
    for index in range(4):
        (tmp_path / f"{index}.jpg").write_bytes(b"frame")
    (tmp_path / "chunks").mkdir()
    request = {
        "job_id": "test",
        "manifest_sha256": "a" * 64,
        "seed_sha256": "b" * 64,
        "end_frame_index": 3,
        "frames": [{"relative_path": f"{index}.jpg", "width": 4, "height": 4} for index in range(4)],
        "seed_set": {
            "seeds": [
                {
                    "object_id": "car-1",
                    "label_id": 1,
                    "label_name": "car",
                    "bbox": {"xtl": 1, "ytl": 1, "xbr": 3, "ybr": 3},
                }
            ]
        },
    }
    result = module._propagate(request, tmp_path, tmp_path / "checkpoint.pt", work_root=tmp_path / "chunks")
    observations = result["tracks"][0]["observations"]
    assert [item["outside"] for item in observations] == [False, True, True, True]
    assert tracker.prompts == [1]
    assert tracker.states == 1
    sys.modules.pop("kaggle.sam3.main", None)
