import json
import sys
import zipfile
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from kaggle.native_memory import run_native, scenes
from kaggle.sam3_benchmark.main import validate_config
from scripts.build_kaggle_notebooks import build_notebook
from scripts.native_memory_trial import evaluate, prepare
from src.services.evaluation.phase0 import file_hash
from tests.test_scripts.test_phase0_cli import synthetic_protocol


class Model:
    def __init__(self, *, missing=False, duplicate=False):
        self.states = []
        self.calls = []
        self.missing = missing
        self.duplicate = duplicate

    def init_state(self, resource_path, **kwargs):
        state = dict(count=len(list(Path(resource_path).glob("*.jpg"))), **kwargs)
        self.states.append(state)
        return state

    def add_prompt(self, inference_state, **kwargs):
        inference_state.update(kwargs)

    def propagate_in_video(self, state, **kwargs):
        self.calls.append(kwargs)
        count = state["count"] - int(self.missing)
        indices = list(range(count))
        if self.duplicate:
            indices.insert(1, 0)
        for index in indices:
            # An object arrives late; disappearance/reappearance keeps its ID.
            ids = [1] if index in (1, 3, 4) else []
            mask = np.zeros((8, 8), dtype=bool)
            mask[2:6, 2:6] = True
            yield index, dict(out_obj_ids=ids, out_binary_masks=[mask] * len(ids), out_probs=[0.8] * len(ids))


def fixture(tmp_path, **kwargs):
    frames = []
    for i in range(10):
        path = tmp_path / f"{i}.jpg"
        path.write_bytes(b"image")
        frames.append(dict(source_frame_index=i * 3, relative_path=path.name, scene_cut_before=i == 5))
    cuda = SimpleNamespace(
        reset_peak_memory_stats=lambda: None,
        synchronize=lambda: None,
        max_memory_allocated=lambda: 123,
        get_device_properties=lambda _: SimpleNamespace(major=7),
    )
    backend = SimpleNamespace(
        model=Model(**kwargs),
        work_root=tmp_path / "work",
        numpy=np,
        torch=SimpleNamespace(
            cuda=cuda,
            float16="fp16",
            bfloat16="bf16",
            autocast=lambda *a, **k: nullcontext(),
            inference_mode=nullcontext,
        ),
        probe=lambda: {},
        recover=lambda: None,
    )
    row = dict(id="native", window_size=2, max_scene_frames=10, text_prompts=["car", "person"])
    return frames, backend, row


def test_native_state_lives_across_output_checkpoints_and_resets_at_scene(tmp_path):
    frames, backend, row = fixture(tmp_path)
    item, snapshots = {}, []
    run_native(
        backend,
        tmp_path,
        frames,
        row,
        item,
        save=lambda: snapshots.append(len(item["predictions"])),
        identity="input",
        model={"checkpoint_sha256": "model"},
        deadline=100,
        clock=lambda: 0,
    )
    assert len(backend.model.states) == 4  # exactly one per class/scene, not checkpoint
    assert len(backend.model.calls) == 4
    assert all(c["max_frame_num_to_track"] == 4 for c in backend.model.calls)
    assert all(s["offload_video_to_cpu"] and s["offload_state_to_cpu"] for s in backend.model.states)
    assert item["evaluated_source_frames"] == [i * 3 for i in range(10)]
    assert item["native_state_checkpoint"] is False
    ids = {p["track_id"] for p in item["predictions"]}
    assert len(ids) == 4
    assert len(item["native_checkpoints"]) == 12
    assert all(c["native_state_checkpoint"] is False for c in item["native_checkpoints"])
    assert snapshots
    assert not list((tmp_path / "work").glob("native-*"))


@pytest.mark.parametrize("kwargs, message", [({"missing": True}, "every scene"), ({"duplicate": True}, "out-of-order")])
def test_incomplete_native_scene_fails_with_partial_outputs(tmp_path, kwargs, message):
    frames, backend, row = fixture(tmp_path, **kwargs)
    item = {}
    with pytest.raises(ValueError, match=message):
        run_native(
            backend,
            tmp_path,
            frames,
            row,
            item,
            save=lambda: None,
            identity="input",
            model={},
            deadline=100,
            clock=lambda: 0,
        )
    assert item["evaluated_source_frames"] == []
    assert item["native_checkpoints"]


def test_native_timeout_preserves_outputs_not_memory(tmp_path):
    frames, backend, row = fixture(tmp_path)
    item = {}
    with pytest.raises(TimeoutError, match="replay scene"):
        run_native(
            backend,
            tmp_path,
            frames,
            row,
            item,
            save=lambda: None,
            identity="input",
            model={},
            deadline=0,
            clock=lambda: 0,
        )
    assert item["evaluated_source_frames"] == []


def test_native_scene_limit_does_not_silently_reset(tmp_path):
    frames, backend, row = fixture(tmp_path)
    row["max_scene_frames"] = 4
    with pytest.raises(ValueError, match="max_scene_frames"):
        run_native(
            backend,
            tmp_path,
            frames,
            row,
            {},
            save=lambda: None,
            identity="input",
            model={},
            deadline=100,
            clock=lambda: 0,
        )
    assert not backend.model.states


def test_scene_partition_and_invalid_policy():
    assert [len(s) for s in scenes([{}, {}, {"scene_cut_before": True}, {}])] == [2, 2]
    with pytest.raises(ValueError, match="tracking policy"):
        validate_config(
            dict(
                max_total_seconds=1,
                max_config_seconds=1,
                source_frame_limit=1,
                configurations=[dict(tracking_policy="unknown")],
            )
        )


def test_paired_payload_has_same_frames_prompts_and_no_ground_truth(tmp_path):
    for sequence, start in (("0013", 0), ("0020", 500)):
        directory = tmp_path / "dataset/training/image_02" / sequence
        directory.mkdir(parents=True)
        for i in range(100):
            Image.new("RGB", (32, 24), (i, 0, 0)).save(directory / f"{start + i:06d}.png")
    output = tmp_path / "trial"
    manifest = prepare(tmp_path / "dataset", output, count=100)
    assert manifest["ground_truth_uploaded"] is False
    with zipfile.ZipFile(output / "payload.zip") as archive:
        assert not any("label" in name or "ground-truth" in name for name in archive.namelist())
        config = json.loads(archive.read("benchmark-config.json"))
        a, b = config["configurations"]
        assert a["selection"] == b["selection"] == "dense"
        for key in ("text_prompts", "scene_cut_frames", "score_threshold_detection", "new_det_thresh"):
            assert a[key] == b[key]
        assert len(json.loads(archive.read("frames.json"))["frames"]) == 200
    (output / "submission.json").write_text("{}")
    with pytest.raises(ValueError, match="reserved experiment"):
        prepare(tmp_path / "dataset", output, count=100)


def test_notebook_bootstrap_can_import_native_adapter(monkeypatch):
    notebook = build_notebook("sam3_benchmark")
    module = ModuleType("native_trial_notebook")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    names = (
        "kaggle.runtime_probe",
        "kaggle.checkpoint_contract",
        "kaggle.native_memory",
        "kaggle.sam3_benchmark",
        "kaggle.sam3_benchmark.main",
    )
    previous = {name: sys.modules.get(name) for name in names}
    try:
        exec("".join(notebook["cells"][1]["source"]), module.__dict__)
        exec("".join(notebook["cells"][2]["source"]), module.__dict__)
        assert callable(sys.modules["kaggle.native_memory"].run_native)
        assert sys.modules["kaggle.sam3_benchmark.main"]._rle is module._rle
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def test_evaluator_uses_source_mapping_and_rejects_replaced_evidence(tmp_path):
    original = synthetic_protocol(tmp_path)
    definition = json.loads(original.read_text())
    other = deepcopy(definition["runs"][0])
    other["sequence"] = "0020"
    definition["runs"].append(other)
    (tmp_path / "eval").mkdir()
    protocol = tmp_path / "eval/protocol.json"
    protocol.write_text(json.dumps(definition))
    output = tmp_path / "trial"
    output.mkdir()
    (output / "payload.zip").write_bytes(b"frozen")
    configs = [dict(id="windows"), dict(id="native")]
    manifest = dict(
        count=5,
        config=dict(configurations=configs),
        image_hashes={},
        payload_sha256=file_hash(output / "payload.zip"),
        clips=[dict(sequence=s, source_start=0, payload_start=i * 5) for i, s in enumerate(("0013", "0020"))],
    )
    (output / "manifest.json").write_text(json.dumps(manifest))
    items = []
    for config in configs:
        rows = [
            dict(
                frame_index=i,
                window_id=i // 5,
                label="car",
                bbox=[10, 10, 20, 20],
                track_id=f"scene{i // 5}:" + (str(i % 5 // 2) if config["id"] == "windows" else "stable"),
            )
            for i in range(10)
        ]
        items.append(
            dict(
                id=config["id"],
                config=config,
                status="completed",
                evaluated_source_frames=list(range(10)),
                predictions=rows,
                window_predictions=rows,
                runtime={},
            )
        )
    result = dict(
        configurations=items,
        model_runtime=dict(
            checkpoint_sha256="checkpoint",
            environment=dict(versions=dict(sam3_commit="2345a4ad109ac29c569da749c91d84f10dc08c40")),
        ),
    )
    (output / "result.json").write_text(json.dumps(result))
    (output / "result-evidence.json").write_text(json.dumps(dict(result_sha256=file_hash(output / "result.json"))))
    report = evaluate(output, protocol)
    assert set(report["clips"]) == {"0013", "0020"}
    for clip in report["clips"].values():
        assert clip["native"]["thresholds"]["0.5"]["recall"] == 1
        assert clip["native"]["thresholds"]["0.5"]["tracking"]["id_switches"] == 0
    assert (output / "comparison.csv").is_file()
    (output / "result.json").write_text("{}")
    with pytest.raises(ValueError, match="result hash"):
        evaluate(output, protocol)
