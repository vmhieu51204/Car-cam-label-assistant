from types import SimpleNamespace

import numpy as np
import pytest

from kaggle.dms_memory import SOURCE_COMMIT, DecoupledMemory, validate_evidence


class Masks:
    def __init__(self, values):
        self.values = np.asarray(values)

    def size(self, axis):
        return self.values.shape[axis]

    def __getitem__(self, key):
        return Masks(self.values[key])


class Model:
    def __init__(self):
        self.tracker = SimpleNamespace(use_memory_selection=True, mf_threshold=0.01)
        self.calls = []

    def _tracker_add_new_objects(
        self,
        frame_idx,
        num_frames,
        new_obj_ids,
        new_obj_masks,
        tracker_states_local,
        orig_vid_height,
        orig_vid_width,
        feature_cache,
    ):
        self.calls.append((new_obj_ids, new_obj_masks.values.copy(), feature_cache, frame_idx))
        tracker_states_local.append({"obj_ids": new_obj_ids})
        return tracker_states_local


def test_singleton_groups_preserve_ids_masks_and_cache():
    model = Model()
    patch = DecoupledMemory(model)
    patch.install()
    states, cache = [], {}
    model._tracker_add_new_objects(4, 20, [9, 42], Masks([[1, 2], [3, 4]]), states, 10, 20, cache)
    assert [s["obj_ids"] for s in states] == [[9], [42]]
    assert all(c[2] is cache and c[3] == 4 for c in model.calls)
    assert model.calls[1][1].tolist() == [[3, 4]]
    assert patch.evidence()["verified_singleton_groups"] == 2
    patch.restore()
    model._tracker_add_new_objects(5, 20, [51, 62], Masks([[1], [2]]), states, 10, 20, cache)
    assert states[-1]["obj_ids"] == [51, 62]


def test_empty_and_mismatched_masks():
    model = Model()
    patch = DecoupledMemory(model)
    patch.install()
    assert model._tracker_add_new_objects(0, 1, [], Masks([]), [], 1, 1, {}) == []
    with pytest.raises(ValueError):
        model._tracker_add_new_objects(0, 1, [1], Masks([]), [], 1, 1, {})
    with pytest.raises(RuntimeError):
        patch.install()


def test_requires_memory_selection_and_known_signature():
    model = Model()
    model.tracker.use_memory_selection = False
    with pytest.raises(NotImplementedError):
        DecoupledMemory(model)
    model._tracker_add_new_objects = lambda: None
    with pytest.raises(NotImplementedError):
        DecoupledMemory(model)


def test_evaluator_requires_effective_dms_evidence():
    item = {"predictions": [{}], "runtime": {}}
    with pytest.raises(ValueError):
        validate_evidence(item)
    item["runtime"]["memory_selection"] = {
        "policy": "dms_singleton",
        "author_commit": SOURCE_COMMIT,
        "verified_singleton_groups": 0,
    }
    with pytest.raises(ValueError):
        validate_evidence(item)
    item["runtime"]["memory_selection"]["verified_singleton_groups"] = 2
    validate_evidence(item)
