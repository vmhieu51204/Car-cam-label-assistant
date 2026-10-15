import importlib.util
import sys
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

path = Path(__file__).resolve().parents[2] / "integrations/cvat-autolabel/files/cvat/apps/engine/autolabel_stream.py"
spec = importlib.util.spec_from_file_location("stream_policy", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def operation(frame=0):
    return {
        "identity": "vm:fixture:track1",
        "label_id": 1,
        "frame": frame,
        "total_frames": 8,
        "points": [1, 2, 10, 12],
        "attributes": [{"spec_id": 4, "value": "vm:fixture:track1"}],
    }


def initial():
    data = {"tags": [], "shapes": [], "tracks": [], "intervals": []}
    data, state, _, _ = module.incremental_merge(data, {}, [operation()])
    data["tracks"][0]["id"] = 1
    return data, state


def test_review_strict_revision_conflicts_inside_cvat_transaction(monkeypatch):
    """A geometry save between API read and DB lock must invalidate acceptance."""
    current = {"tags": [], "shapes": [], "tracks": [], "intervals": []}
    models = SimpleNamespace(
        AutoLabelStreamState=SimpleNamespace(
            objects=SimpleNamespace(get_or_create=lambda **kwargs: (SimpleNamespace(state={}), False))
        ),
        AutoLabelStreamOperation=SimpleNamespace(
            objects=SimpleNamespace(filter=lambda **kwargs: SimpleNamespace(first=lambda: None))
        ),
    )
    monkeypatch.setitem(sys.modules, "cvat.apps.engine.models", models)
    monkeypatch.setitem(
        sys.modules, "cvat.apps.engine.autolabel_revision", SimpleNamespace(annotation_revision=lambda data: 123)
    )
    status, _response = module.handle_stream(
        job_id=1,
        data={
            "operation_id": "review-fixture",
            "base": current,
            "local": current,
            "expected_revision": 122,
            "require_revision": True,
        },
        mode="save",
        transaction=SimpleNamespace(atomic=nullcontext),
        job_model=SimpleNamespace(
            objects=SimpleNamespace(select_for_update=lambda: SimpleNamespace(get=lambda **kwargs: None))
        ),
        get_annotations=lambda identity: current,
        patch_annotations=lambda *args: pytest.fail("Stale review must not write"),
        put_annotations=lambda *args: pytest.fail("Stale review must not replace annotations"),
    )
    assert status == 409


def test_later_frame_keeps_edited_box_and_global_attributes():
    data, state = initial()
    data["tracks"][0]["shapes"][0]["points"] = [2, 3, 20, 23]
    data["tracks"][0]["label_id"] = 2
    result, _, creates, updates = module.incremental_merge(data, state, [operation(), operation(1)])
    assert not creates and len(updates) == 1
    assert result["tracks"][0]["shapes"][0]["points"] == [2, 3, 20, 23]
    assert result["tracks"][0]["label_id"] == 2
    assert not next(s for s in result["tracks"][0]["shapes"] if s["frame"] == 1)["outside"]


def test_replacing_automatic_outside_preserves_human_occlusion_and_z_order():
    data, state = initial()
    outside = data["tracks"][0]["shapes"][1]
    outside.update(occluded=True, z_order=7, attributes=[{"spec_id": 9, "value": "human"}])
    result, _, _, _ = module.incremental_merge(data, state, [operation(1)])
    shape = next(s for s in result["tracks"][0]["shapes"] if s["frame"] == 1)
    assert not shape["outside"] and shape["occluded"] and shape["z_order"] == 7
    assert shape["attributes"] == [{"spec_id": 9, "value": "human"}]


def test_restored_and_new_annotations_map_to_their_own_client_ids():
    base, state = initial()
    base["tracks"][0]["shapes"][0]["id"] = 91
    local, remote = deepcopy(base), deepcopy(base)
    local["tracks"][0]["clientID"] = 5
    local["tracks"][0]["shapes"][0]["occluded"] = True
    local["tracks"].append({"clientID": 6, "label_id": 2, "shapes": []})
    remote["tracks"] = []
    result = module.merge_save(base, local, remote, state, {"tracks/1/deleted": "local"})
    assert module.created_client_ids(result["tracks"], [{"id": 100}, {"id": 101}], local["tracks"]) == {
        "6": 100,
        "5": 101,
    }


def test_deleted_track_never_resurrects_and_manual_outside_stops_extension():
    data, state = initial()
    data["tracks"] = []
    result, state, creates, _ = module.incremental_merge(data, state, [operation(1)])
    assert not creates and state["tracks"]["vm:fixture:track1"]["deleted"]
    data, state = initial()
    data["tracks"][0]["shapes"][1]["points"] = [0, 0, 1, 1]
    result, _, _, _ = module.incremental_merge(data, state, [operation(2)])
    assert not any(s["frame"] == 2 for s in result["tracks"][0]["shapes"])


def test_deleted_keyframe_is_not_restored_by_replayed_model_output():
    data, state = initial()
    data["tracks"][0]["shapes"] = [s for s in data["tracks"][0]["shapes"] if s["frame"] != 0]
    result, state, creates, _ = module.incremental_merge(data, state, [operation()])
    assert not creates and not any(s["frame"] == 0 for s in result["tracks"][0]["shapes"])
    assert "0" in state["tracks"]["vm:fixture:track1"]["deleted_frames"]


def test_explicit_conflict_choice_merges_other_fields():
    base, state = initial()
    local, remote = deepcopy(base), deepcopy(base)
    local["tracks"][0]["shapes"][0]["points"] = [2, 3, 20, 23]
    remote["tracks"][0]["shapes"][0]["points"] = [4, 5, 24, 25]
    remote["tracks"][0]["label_id"] = 2
    path = "tracks/1/shapes/frames/0/points"
    mine = module.merge_save(base, local, remote, state, {path: "local"})
    theirs = module.merge_save(base, local, remote, state, {path: "remote"})
    assert mine["tracks"][0]["shapes"][0]["points"] == [2, 3, 20, 23]
    assert theirs["tracks"][0]["shapes"][0]["points"] == [4, 5, 24, 25]
    assert mine["tracks"][0]["label_id"] == 2


def test_save_merges_auto_future_with_human_edit_and_conflicts_same_field():
    base, state = initial()
    local = deepcopy(base)
    local["tracks"][0]["shapes"][0]["points"] = [2, 3, 20, 23]
    remote, new_state, _, _ = module.incremental_merge(base, state, [operation(1)])
    result = module.merge_save(base, local, remote, new_state)
    assert result["tracks"][0]["shapes"][0]["points"] == [2, 3, 20, 23]
    assert any(s["frame"] == 2 for s in result["tracks"][0]["shapes"])
    remote["tracks"][0]["shapes"][0]["points"] = [4, 5, 24, 25]
    with pytest.raises(module.MergeConflict):
        module.merge_save(base, local, remote, new_state)


def test_deletion_of_auto_extended_track_and_new_shape():
    base, state = initial()
    remote, new_state, _, _ = module.incremental_merge(base, state, [operation(1)])
    local = deepcopy(base)
    local["tracks"] = []
    result = module.merge_save(base, local, remote, new_state)
    assert not result["tracks"]
    local = deepcopy(base)
    local["shapes"].append({"frame": 2, "points": [1, 2, 4, 5], "type": "rectangle", "label_id": 1})
    assert len(module.merge_save(base, local, remote, new_state)["shapes"]) == 1
