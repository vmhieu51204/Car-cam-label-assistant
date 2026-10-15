from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import pytest

from scripts.evaluate_benchmark import cmd_manual, cmd_report
from src.services.evaluation.phase0 import compare_baselines, file_hash, write_baseline


def test_manual_cli_fixture_has_actual_negative_frame_and_errors(tmp_path: Path) -> None:
    fixture = Path(__file__).parents[1] / "fixtures/evaluation/phase0"
    args = argparse.Namespace(
        ground_truth=fixture / "manual-gt.json",
        predictions=fixture / "predictions.json",
        output=tmp_path / "manual.json",
    )
    cmd_manual(args)
    metrics = json.loads(args.output.read_text())["thresholds"]["0.5"]
    assert metrics["recall"] == 0.8
    assert metrics["fp_per_frame"] == pytest.approx(1 / 6)
    assert metrics["tracking"]["idf1"] == 0.6


def test_old_report_contract_and_undefined_rates_csv(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    path.write_text(
        json.dumps(
            {
                "evaluations": [
                    {
                        "config_id": "one",
                        "view": "raw",
                        "status": "completed",
                        "scopes": [
                            {
                                "scope": {"scope_id": "fixed"},
                                "groups": [
                                    {
                                        "group_id": "micro:none:all",
                                        "aggregation": "micro",
                                        "diagnostic": {
                                            "tp": 0,
                                            "fp": 0,
                                            "fn": 0,
                                            "precision": {"value": None},
                                            "recall": {"value": None},
                                            "f1": {"value": None},
                                        },
                                    }
                                ],
                            }
                        ],
                    }
                ]
            }
        )
    )
    cmd_report(argparse.Namespace(report=path, output=tmp_path / "out"))
    with (tmp_path / "out/report.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1 and rows[0]["precision"] == ""


def synthetic_protocol(root: Path) -> Path:
    mapping = {"source_fps": 10, "frames": [{"frame_index": i, "source_frame_index": 2 * i} for i in range(3)]}
    predictions = [
        {"frame_index": i, "track_id": "car:1", "label": "car", "bbox": [10, 10, 20, 20], "score": 0.9}
        for i in range(3)
    ]
    configuration = {
        "id": "baseline",
        "status": "completed",
        "config": {"mode": "text_prompt"},
        "evaluated_source_frames": [0, 1, 2],
        "predictions": predictions,
        "window_predictions": [{"window_id": 0, **x} for x in predictions],
    }
    (root / "result.json").write_text(json.dumps({"configurations": [configuration]}))
    (root / "mapping.json").write_text(json.dumps(mapping))
    (root / "labels.txt").write_text("\n".join(f"{i} 1 Car 0 0 0 10 10 20 20 1 1 1 0 0 1 0" for i in range(5)))
    run = {
        "sequence": "0013",
        "native_frame_count": 5,
        "config_id": "baseline",
        "result": "result.json",
        "mapping": "mapping.json",
        "labels": "labels.txt",
        "class_map": {"Car": "car", "Pedestrian": "person"},
        "sha256": {
            name: file_hash(root / f"{name}.{'txt' if name == 'labels' else 'json'}")
            for name in ("result", "mapping", "labels")
        },
    }
    path = root / "protocol.json"
    path.write_text(json.dumps({"schema": "autolabel-phase0-protocol/v1", "seed": 42, "runs": [run]}))
    return path


def test_stored_baseline_views_mapping_hash_and_repeatability(tmp_path: Path) -> None:
    protocol = synthetic_protocol(tmp_path)
    input_before = {p.name: file_hash(p) for p in tmp_path.glob("*.json")}
    first = write_baseline(protocol, root=tmp_path, output=tmp_path / "before")
    second = write_baseline(protocol, root=tmp_path, output=tmp_path / "after")
    assert first == second
    assert input_before == {p.name: file_hash(p) for p in tmp_path.glob("*.json")}
    views = first["runs"][0]["views"]
    assert views["direct_sampled"]["raw"]["annotated_frames"] == 3
    assert views["interpolated_only"]["raw"]["annotated_frames"] == 2
    assert views["interpolated_timeline"]["raw"]["thresholds"]["0.5"]["recall"] == 1
    assert first["services_called"] == [] and first["annotation_change"] is False
    assert first["runs"][0]["runtime"]["peak_gpu_memory_bytes"] is None
    assert (tmp_path / "after/slices.csv").exists()
    with (tmp_path / "after/profile.csv").open(newline="") as stream:
        stages = list(csv.DictReader(stream))
    assert any(row["run"] == "0013" and row["stage"] == "model_load" and row["seconds"] == "" for row in stages)
    assert first["runs"][0]["views"]["direct_sampled"]["kitti_style"]["ignored_predictions"] == {}
    (tmp_path / "result.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum mismatch"):
        write_baseline(protocol, root=tmp_path, output=tmp_path / "bad")


def test_old_artifact_missing_overlap_is_not_fabricated(tmp_path: Path) -> None:
    protocol = synthetic_protocol(tmp_path)
    result = json.loads((tmp_path / "result.json").read_text())
    del result["configurations"][0]["window_predictions"]
    (tmp_path / "result.json").write_text(json.dumps(result))
    raw = json.loads(protocol.read_text())
    raw["runs"][0]["sha256"]["result"] = file_hash(tmp_path / "result.json")
    protocol.write_text(json.dumps(raw))
    before = copy.deepcopy(result)
    report = write_baseline(protocol, root=tmp_path, output=tmp_path / "out")
    assert report["runs"][0]["missing_views"] == ["stitched_sampled", "interpolated_timeline"]
    assert json.loads((tmp_path / "result.json").read_text()) == before


def test_equivalence_detects_metric_and_availability_changes(tmp_path: Path) -> None:
    protocol = synthetic_protocol(tmp_path)
    write_baseline(protocol, root=tmp_path, output=tmp_path / "before")
    write_baseline(protocol, root=tmp_path, output=tmp_path / "after")
    a, b = (tmp_path / side / "baseline.json" for side in ("before", "after"))
    assert compare_baselines(a, b)["metrics_unchanged"]
    data = json.loads(b.read_text())
    metrics = data["runs"][0]["views"]["direct_sampled"]["raw"]["thresholds"]["0.5"]
    metrics["tracking"]["idf1"] = None
    b.write_text(json.dumps(data))
    assert not compare_baselines(a, b)["metrics_unchanged"]
    data["runs"][0]["prediction_sha256"] = "changed"
    b.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="artifacts changed"):
        compare_baselines(a, b)


def test_equivalence_tracks_profile_population_and_export_hash(tmp_path: Path) -> None:
    protocol = synthetic_protocol(tmp_path)
    export = tmp_path / "cvat.xml"
    export.write_text(
        '<annotations><version>1.1</version><track id="1" label="car"><box frame="0" xtl="10" ytl="10" xbr="20" ybr="20" outside="0"/></track></annotations>'
    )
    config = json.loads(protocol.read_text())
    config["runs"][0]["cvat_export"] = "cvat.xml"
    protocol.write_text(json.dumps(config))
    report = write_baseline(protocol, root=tmp_path, output=tmp_path / "a")
    assert report["runs"][0]["input_hashes"]["cvat_export"] == file_hash(export)
    a, b = tmp_path / "a/baseline.json", tmp_path / "b.json"
    report["runs"][0]["views"]["direct_sampled"]["kitti_style"] = None
    b.write_text(json.dumps(report))
    assert not compare_baselines(a, b)["metrics_unchanged"]
    del report["runs"][0]["views"]["cvat_export"]
    b.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="view population changed"):
        compare_baselines(a, b)
    export.write_text(export.read_text() + "\n")
    write_baseline(protocol, root=tmp_path, output=tmp_path / "changed")
    with pytest.raises(ValueError, match="artifacts changed"):
        compare_baselines(a, tmp_path / "changed/baseline.json")


def test_old_artifact_without_track_ids_keeps_box_rates_only(tmp_path: Path) -> None:
    protocol = synthetic_protocol(tmp_path)
    path = tmp_path / "result.json"
    data = json.loads(path.read_text())
    item = data["configurations"][0]
    del item["window_predictions"]
    for box in item["predictions"]:
        box["track_id"] = None
    path.write_text(json.dumps(data))
    config = json.loads(protocol.read_text())
    config["runs"][0]["sha256"]["result"] = file_hash(path)
    protocol.write_text(json.dumps(config))
    report = write_baseline(protocol, root=tmp_path, output=tmp_path / "out")
    rates = report["runs"][0]["views"]["direct_sampled"]["raw"]["thresholds"]["0.5"]
    assert rates["recall"] == 1
    assert rates["tracking"]["idf1"] is None
    assert rates["tracking"]["availability"] == "missing_track_ids"
