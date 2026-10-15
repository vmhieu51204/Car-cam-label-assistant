from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from kaggle.sam3_benchmark.main import (
    Sam3Backend,
    locate_payload,
    run_benchmark,
    select_frames,
    validate_config,
    windows,
)
from scripts.build_kaggle_notebooks import build_notebook
from scripts.compare_sam3_configs import compare, write_csv
from scripts.eval_autolabel import read_boxes
from scripts.package_sam3_benchmark import package

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "eval/sample/ground_truth.coco.json"


def _payload(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    for index in range(4):
        Image.new("RGB", (100, 80), "gray").save(source / f"frame_{index:06d}.jpg")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "frames": [
                    {"frame_index": local, "source_frame_index": source_index}
                    for local, source_index in enumerate((0, 2, 3))
                ]
            }
        )
    )
    template = json.loads((ROOT / "kaggle/sam3_benchmark/benchmark-config.json").read_text())
    template["source_frame_limit"] = 4
    template["configurations"] = [
        {**template["configurations"][0], "window_size": 25},
        {**template["configurations"][0], "id": "fails", "window_size": 25},
        {**template["configurations"][5], "id": "mask", "window_size": 25},
    ]
    config = tmp_path / "config.json"
    config.write_text(json.dumps(template))
    archive = tmp_path / "payload.zip"
    report = package(frames_dir=source, manifest_path=manifest, ground_truth=SAMPLE, config_path=config, output=archive)
    assert report["source_frames_packaged"] == 3
    root = tmp_path / "payload"
    root.mkdir()
    with zipfile.ZipFile(archive) as source_zip:
        source_zip.extractall(root)
    return root


class FakeSam3:
    def __init__(self) -> None:
        self.calls = []

    def load(self, root: Path) -> None:
        assert (root / "seed.json").is_file()

    def probe(self) -> dict:
        return {"peak_cuda_allocated_bytes": 123, "peak_cuda_allocated_bytes_by_device": {"0": 123}}

    def begin_config(self) -> None:
        pass

    def run_window(self, root, batch, seeds, row, window_index, previous, previous_masks):
        self.calls.append(row["id"])
        if row["id"] == "fails":
            raise RuntimeError("simulated OOM")
        return {
            frame["source_frame_index"]: [
                {
                    "track_id": seed["track_id"],
                    "label": seed["label"],
                    "bbox": seed["bbox"],
                    "score": 1.0,
                    "outside": False,
                }
                for seed in seeds
            ]
            for frame in batch
        }

    def recover(self) -> None:
        pass

    def release(self) -> None:
        pass


def test_locate_payload_accepts_kaggle_expanded_or_preserved_zip(tmp_path: Path) -> None:
    # Kaggle may expand a nested upload to <dataset>/payload/ rather than keep the ZIP.
    unpacked = tmp_path / "dataset/payload"
    unpacked.mkdir(parents=True)
    for name in ("benchmark-config.json", "frames.json", "seed.json"):
        (unpacked / name).write_text("{}")
    assert locate_payload(tmp_path, tmp_path / "work") == unpacked

    archive_input = tmp_path / "archive-input"
    archive_input.mkdir()
    with zipfile.ZipFile(archive_input / "payload.zip", "w") as archive:
        for name in ("benchmark-config.json", "frames.json", "seed.json"):
            archive.writestr(name, "{}")
    with pytest.raises(RuntimeError, match="exactly one"):
        locate_payload(tmp_path, tmp_path / "work")
    for file in unpacked.iterdir():
        file.unlink()
    assert locate_payload(tmp_path, tmp_path / "work") == tmp_path / "work/sam3-benchmark-input"


def test_text_window_preserves_sam3_detection_score(tmp_path: Path) -> None:
    class FakeTextModel:
        def init_state(self, *, resource_path: str) -> dict:
            assert resource_path == str(tmp_path)
            return {}

        def add_prompt(self, *, inference_state: dict, frame_idx: int, text_str: str) -> None:
            assert frame_idx == 0 and text_str == "person"

        def propagate_in_video(self, state: dict, **_kwargs):
            mask = np.zeros((4, 4), dtype=bool)
            mask[1:3, 1:3] = True
            yield 0, {"out_obj_ids": [7], "out_binary_masks": [mask], "out_probs": [0.72]}

    backend = Sam3Backend()
    backend.model = FakeTextModel()
    backend.numpy = np
    result = backend._text_window(tmp_path, [{"source_frame_index": 0}], {"text_prompts": ["person"]}, 2)
    assert result[0][0]["score"] == pytest.approx(0.72)
    assert result[0][0]["bbox"] == [1.0, 1.0, 3.0, 3.0]
    assert backend.class_timings[0]["label"] == "person"
    assert backend.class_timings[0]["seconds"] >= 0


def test_sampler_rejects_unknown_model_variant() -> None:
    template = json.loads((ROOT / "kaggle/sam3_benchmark/benchmark-config.json").read_text())
    template["source_frame_limit"] = 2
    template["configurations"] = [{**template["configurations"][0], "model_variant": "sam3.2"}]
    with pytest.raises(ValueError, match="unsupported SAM model variant"):
        validate_config(template)


def test_frame_selection_and_windows() -> None:
    frames = [{"source_frame_index": index} for index in range(21)]
    row = {"selection": "uniform", "gap": 5}
    selected = select_frames(frames, [0, 3, 9], row)
    assert [frame["source_frame_index"] for frame in selected] == [0, 5, 10, 15, 20]
    assert [start for start, _ in windows(selected, 3, 1)] == [0, 2]


def test_nonoverlapping_windows_do_not_require_next_frame_in_current_window(tmp_path: Path) -> None:
    frames_dir = tmp_path / "source"
    frames_dir.mkdir()
    for index in range(30):
        Image.new("RGB", (100, 80), "gray").save(frames_dir / f"frame_{index:06d}.jpg")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"frames": [{"frame_index": index, "source_frame_index": index} for index in range(30)]})
    )
    config = json.loads((ROOT / "kaggle/sam3_benchmark/benchmark-config.json").read_text())
    config["source_frame_limit"] = 30
    config["configurations"] = [config["configurations"][0]]
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    archive = tmp_path / "payload.zip"
    package(frames_dir=frames_dir, manifest_path=manifest, ground_truth=SAMPLE, config_path=config_path, output=archive)
    root = tmp_path / "payload"
    root.mkdir()
    with zipfile.ZipFile(archive) as packed:
        packed.extractall(root)
    result = run_benchmark(root, tmp_path / "result.json", FakeSam3())
    assert result["configurations"][0]["status"] == "completed"
    assert result["configurations"][0]["evaluated_source_frames"] == list(range(30))


def test_overlap_predictions_keep_both_windows_for_stitching(tmp_path: Path) -> None:
    root = _payload(tmp_path)
    config_path = root / "benchmark-config.json"
    config = json.loads(config_path.read_text())
    config["configurations"] = [
        {**config["configurations"][0], "window_size": 2, "overlap": 1, "preserve_overlap": True}
    ]
    # This tiny synthetic fixture can use window size 2; production configs stay bounded.
    config_path.write_text(json.dumps(config))
    fake = FakeSam3()
    result = run_benchmark(root, tmp_path / "result.json", fake)
    item = result["configurations"][0]
    assert item["status"] == "completed"
    assert len(item["predictions"]) == 3
    assert len(item["window_predictions"]) == 4
    assert {row["window_id"] for row in item["window_predictions"] if row["frame_index"] == 2} == {0, 1}


def test_config_rejects_invalid_overlap() -> None:
    config = json.loads((ROOT / "kaggle/sam3_benchmark/benchmark-config.json").read_text())
    config["configurations"][0]["overlap"] = 50
    try:
        validate_config(config)
    except ValueError as error:
        assert "overlap" in str(error)
    else:
        raise AssertionError("invalid overlap was accepted")


def test_config_rejects_text_mode_without_known_label_prompts() -> None:
    config = json.loads((ROOT / "kaggle/sam3_benchmark/benchmark-config.json").read_text())
    text_mode = next(row for row in config["configurations"] if row["mode"] == "text_prompt")
    text_mode["text_prompts"] = []
    try:
        validate_config(config)
    except ValueError as error:
        assert "text_prompts" in str(error)
    else:
        raise AssertionError("empty text prompts were accepted")


def test_config_accepts_proposed_chunk_sizes_and_rejects_unbounded_windows() -> None:
    config = json.loads((ROOT / "kaggle/sam3_benchmark/benchmark-config.json").read_text())
    config["configurations"][0]["window_size"] = 200
    validate_config(config)
    config["configurations"][0]["window_size"] = 251
    with pytest.raises(ValueError, match="window_size"):
        validate_config(config)


def test_sam3_benchmark_partial_failure_and_comparison(tmp_path: Path) -> None:
    root = _payload(tmp_path)
    fake = FakeSam3()
    output = tmp_path / "result.json"
    result = run_benchmark(root, output, fake)
    assert [item["status"] for item in result["configurations"]] == ["completed", "failed", "completed"]
    assert result == json.loads(output.read_text())
    assert result["configurations"][0]["evaluated_source_frames"] == [0, 2, 3]
    rows = compare(result, read_boxes(SAMPLE))
    assert rows[0]["evaluated_frames"] == 3
    assert rows[0]["vram_peak_bytes"] == 123
    write_csv(rows, tmp_path / "comparison.csv")
    assert "precision_iou_05" in (tmp_path / "comparison.csv").read_text()


def test_comparison_uses_baseline_on_identical_frames() -> None:
    ground_truth = read_boxes(SAMPLE)
    baseline_predictions = [
        {"frame_index": box.frame, "label": box.label, "bbox": list(box.xyxy), "track_id": box.track_id}
        for box in ground_truth
    ]
    config = {"selection": "optical", "window_size": 50, "overlap": 0, "reprompt": "box_last", "mode": "box_seed"}
    result = {
        "configurations": [
            {
                "id": "optical-current",
                "status": "completed",
                "config": config,
                "evaluated_source_frames": [0, 1, 2, 3],
                "predictions": baseline_predictions,
            },
            {
                "id": "sparse",
                "status": "completed",
                "config": {**config, "selection": "uniform"},
                "evaluated_source_frames": [0, 2],
                "predictions": [item for item in baseline_predictions if item["frame_index"] == 0],
            },
        ]
    }
    rows = compare(result, ground_truth)
    assert rows[1]["common_baseline_frames"] == 2
    assert rows[1]["frame_coverage_vs_baseline"] == 0.5
    assert rows[1]["baseline_recall_iou_05_common"] == 1
    assert rows[1]["recall_delta_vs_baseline_common"] < 0


def test_comparison_uses_partial_timeout_results_on_common_frames() -> None:
    ground_truth = read_boxes(SAMPLE)
    boxes = [
        {"frame_index": box.frame, "label": box.label, "bbox": list(box.xyxy), "track_id": box.track_id}
        for box in ground_truth
    ]
    config = {"selection": "optical", "window_size": 50, "overlap": 0, "reprompt": "box_last", "mode": "box_seed"}
    rows = compare(
        {
            "configurations": [
                {
                    "id": "optical-current",
                    "status": "timed_out",
                    "config": config,
                    "evaluated_source_frames": [0, 1],
                    "predictions": boxes[:1],
                },
                {
                    "id": "text",
                    "status": "timed_out",
                    "config": {**config, "mode": "text_prompt"},
                    "evaluated_source_frames": [0],
                    "predictions": boxes[:1],
                },
            ]
        },
        ground_truth,
    )
    assert rows[1]["baseline_id"] == "optical-current"
    assert rows[1]["common_baseline_frames"] == 1
    assert rows[1]["frame_coverage_vs_baseline"] == 0.5
    assert rows[1]["recall_iou_05_common"] is not None


def test_sam3_benchmark_records_unrun_configs_when_total_budget_expires(tmp_path: Path) -> None:
    root = _payload(tmp_path)
    config_path = root / "benchmark-config.json"
    config = json.loads(config_path.read_text())
    config["max_total_seconds"] = 0.5
    config_path.write_text(json.dumps(config))
    fake = FakeSam3()
    ticks = iter(range(100))
    result = run_benchmark(root, tmp_path / "result.json", fake, clock=lambda: next(ticks))
    assert [item["status"] for item in result["configurations"]] == ["skipped_budget"] * 3
    assert fake.calls == []
    assert result == json.loads((tmp_path / "result.json").read_text())


def test_uploadable_sam3_benchmark_notebook_matches_builder() -> None:
    notebook = build_notebook("sam3_benchmark")
    committed = json.loads((ROOT / "kaggle/notebooks/autolabel_sam3_benchmark.ipynb").read_text())
    assert notebook == committed
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "sam3-benchmark-notebook-cell", "exec")


def test_text_checkpoint_survives_failure_and_skips_verified_windows(tmp_path):
    root = _payload(tmp_path)
    config_path = root / "benchmark-config.json"
    config = json.loads(config_path.read_text())
    config["configurations"] = [
        {
            **config["configurations"][0],
            "id": "resume",
            "window_size": 2,
            "overlap": 1,
            "preserve_overlap": True,
            "mode": "text_prompt",
            "seed_type": "none",
            "reprompt": "none",
            "text_prompts": ["car"],
        }
    ]
    config_path.write_text(json.dumps(config))

    class Interrupted(FakeSam3):
        def run_window(self, root, batch, seeds, row, window_index, previous, previous_masks):
            if window_index == 1:
                raise RuntimeError("simulated second-window OOM")
            return super().run_window(root, batch, seeds, row, window_index, previous, previous_masks)

    first = run_benchmark(root, tmp_path / "first.json", Interrupted())
    item = first["configurations"][0]
    assert item["status"] == "failed"
    assert len(item["window_checkpoints"]) == 1
    (root / "resume.json").write_text(json.dumps({"windows": item["window_checkpoints"]}))
    second_backend = FakeSam3()
    second = run_benchmark(root, tmp_path / "second.json", second_backend)["configurations"][0]
    assert second["status"] == "completed"
    assert second["resumed_windows"] == [0]
    assert len(second_backend.calls) == 1
    assert second["window_checkpoints"][0] == item["window_checkpoints"][0]
    assert second["evaluated_source_frames"] == [0, 2, 3]
    corrupted = item["window_checkpoints"][0]
    corrupted["predictions"][0]["bbox"][0] += 1
    (root / "resume.json").write_text(json.dumps({"windows": [corrupted]}))
    broken = run_benchmark(root, tmp_path / "broken.json", FakeSam3())["configurations"][0]
    assert broken["status"] == "failed"
    assert "hash mismatch" in broken["error"]


def test_scene_windows_never_overlap_across_cut():
    frames = [{"source_frame_index": i, "scene_cut_before": i == 4} for i in range(9)]
    batches = windows(frames, 3, 1)
    assert [[x["source_frame_index"] for x in batch] for _, batch in batches] == [
        [0, 1, 2],
        [2, 3],
        [4, 5, 6],
        [6, 7, 8],
    ]
