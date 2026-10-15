from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.eval_autolabel import Box
from src.services.evaluation.geometry import (
    IMAGE_QUOTAS,
    box_geometry,
    direct_ledger,
    distributions,
    other_gt_evidence,
    recall_rows,
    select_images,
    validate_misses,
    write_geometry_analysis,
)
from src.services.evaluation.geometry import (
    _write_geometry_analysis as write_geometry_fixture,
)
from src.services.evaluation.misses import analyze_direct, write_miss_analysis
from src.services.evaluation.phase0 import file_hash, load_frozen_run
from tests.test_scripts.test_phase0_cli import synthetic_protocol
from tests.test_services.test_phase05_misses import miss_fixture


@pytest.mark.parametrize(
    "pred,label",
    [
        ([2, 2, 8, 8], "pred_inside_gt"),
        ([-5, -5, 15, 15], "gt_inside_pred"),
        ([5, 0, 15, 10], "shifted"),
        ([0, 0, 10, 10], "shifted"),
    ],
)
def test_geometry_labels_and_equal_area(pred: list[float], label: str) -> None:
    assert box_geometry([0, 0, 10, 10], pred)["geometry"] == label


def test_geometry_formulas_signed_centers_and_containment_boundary() -> None:
    data = box_geometry([0, 0, 10, 20], [5, 0, 25, 10])
    assert data["pred_area_over_gt_area"] == 1
    assert data["intersection_over_pred_area"] == 0.25
    assert data["intersection_over_gt_area"] == 0.25
    assert data["pred_width_over_gt_width"] == 2
    assert data["pred_height_over_gt_height"] == 0.5
    assert data["pred_aspect_over_gt_aspect"] == 4
    assert data["center_dx_over_gt_width"] == 1
    assert data["center_dy_over_gt_height"] == -0.25
    assert data["center_distance_normalized"] == pytest.approx(np.hypot(1, -0.25))
    assert box_geometry([1, 0, 21, 10], [0, 0, 10, 5])["geometry"] == "pred_inside_gt"
    assert box_geometry([1.001, 0, 21, 10], [0, 0, 10, 5])["geometry"] == "shifted"
    assert box_geometry([0, 0, 10, 5], [1, 0, 21, 10])["geometry"] == "gt_inside_pred"
    assert box_geometry([0, 0, 10, 10], None)["pred_area_over_gt_area"] is None
    with pytest.raises(ValueError, match="positive"):
        box_geometry([0, 0, 0, 10], [0, 0, 10, 10])


@pytest.mark.parametrize("native_class", ["Car", "Van", "Cyclist", "Misc"])
def test_other_gt_covers_unprompted_classes_with_native_ties(tmp_path: Path, native_class: str) -> None:
    protocol = synthetic_protocol(tmp_path)
    loaded = load_frozen_run(json.loads(protocol.read_text())["runs"][0], tmp_path)
    gt = loaded.native_rows[0]
    other = gt.model_copy(update={"track_id": 8, "native_class": native_class, "bbox_xyxy": (20, 10, 30, 20)})
    dontcare = gt.model_copy(update={"track_id": -1, "native_class": "DontCare", "bbox_xyxy": (20, 10, 30, 20)})
    row = {
        "annotation_id": "kitti/0013/0",
        "sequence": "0013",
        "frame": 0,
        "best_prediction_box": [20, 10, 30, 20],
        "best_iou": 0,
    }
    probe = replace(loaded, native_rows=(gt, dontcare, other, other.model_copy(update={"track_id": 9})))
    result = other_gt_evidence(probe, row)
    assert result["prediction_belongs_to_other_gt"] is True
    assert result["other_gt_id"] == "8" and result["other_gt_ordinal"] == 2
    assert result["other_gt_native_class"] == native_class
    assert result["other_gt_iou"] == result["other_gt_iou_delta"] == 1
    assert other_gt_evidence(probe, {**row, "best_iou": 1})["prediction_belongs_to_other_gt"] is False
    assert other_gt_evidence(probe, {**row, "best_iou": 1 - 1e-12})["prediction_belongs_to_other_gt"] is True
    assert other_gt_evidence(replace(loaded, native_rows=(gt, dontcare)), row)["other_gt_id"] is None
    assert other_gt_evidence(probe, {**row, "best_prediction_box": None})["other_gt_id"] is None


def test_csv_validated_against_artifact_and_unchanged_columns(tmp_path: Path) -> None:
    protocol = miss_fixture(tmp_path)
    write_miss_analysis(protocol, root=tmp_path, output=tmp_path / "old", bootstrap_repeats=10)
    loaded = load_frozen_run(json.loads(protocol.read_text())["runs"][0], tmp_path)
    _, expected = analyze_direct(loaded)
    path = tmp_path / "old/misses.csv"
    assert "frame_sampled" in validate_misses(path, expected)
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    rows[0]["best_prediction_ordinal"] = "99"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="best_prediction_ordinal"):
        validate_misses(path, expected)
    with pytest.raises(ValueError, match="population"):
        validate_misses(path, expected[:-1])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(reversed(fields)))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="columns"):
        validate_misses(path, expected)


def test_csv_extra_cells_rejected_and_public_seed_fixed(tmp_path: Path) -> None:
    protocol = miss_fixture(tmp_path)
    write_miss_analysis(protocol, root=tmp_path, output=tmp_path / "old", bootstrap_repeats=10)
    loaded = load_frozen_run(json.loads(protocol.read_text())["runs"][0], tmp_path)
    _, expected = analyze_direct(loaded)
    path = tmp_path / "old/misses.csv"
    lines = path.read_text().splitlines()
    lines[1] += ",EXTRA"
    path.write_text("\n".join(lines))
    with pytest.raises(ValueError, match="row shape"):
        validate_misses(path, expected)
    with pytest.raises(ValueError, match="fixed seed 42"):
        write_geometry_analysis(protocol, root=tmp_path, misses_csv=path, output=tmp_path / "new", seed=7)
    assert not (tmp_path / "new").exists()


def test_ledger_best_prediction_ties_mapping_and_one_to_one_recall(tmp_path: Path) -> None:
    protocol = miss_fixture(tmp_path)
    loaded = load_frozen_run(json.loads(protocol.read_text())["runs"][0], tmp_path)
    ledger, misses = direct_ledger(loaded)
    assert len(ledger) == 6 and len(misses) == 4
    assert all(r["geometry"] == "not_applicable" for r in misses)
    assert next(r for r in ledger if r["frame"] == 2)["frame_sampled"] == 1
    assert all(r["other_gt_iou_delta"] == 0 for r in ledger)
    recall = recall_rows(loaded)
    assert all(r["tp"] == 2 and r["gt"] == 6 and r["fn"] == 4 for r in recall)
    assert all(r["recall"] == 1 / 3 for r in recall)
    records, missing = direct_ledger(replace(loaded, direct=[]))
    assert len(missing) == 6 and all(r["best_iou"] == 0 for r in records)
    assert all(r["distribution_area_ratio"] is None for r in records)
    probe = replace(
        loaded, direct=[Box(0, "person", (10, 10, 20, 20), "first"), Box(0, "car", (10, 10, 20, 20), "second")]
    )
    records, _ = direct_ledger(probe)
    assert records[0]["best_prediction_track_id"] == "first"


def test_unoccluded_distribution_missing_prediction_and_strata() -> None:
    common = {
        "sequence": "s",
        "class": "car",
        "occlusion": "none",
        "reason": "poor_localization",
        "truncation": 0,
        "prediction_belongs_to_other_gt": False,
    }
    rows = [
        {**common, "best_iou": 0, "best_prediction_box": None, "distribution_area_ratio": None},
        {**common, "best_iou": 0.5, "best_prediction_box": [0, 0, 1, 1], "distribution_area_ratio": 1},
        {
            **common,
            "truncation": 1,
            "prediction_belongs_to_other_gt": True,
            "best_iou": 1,
            "best_prediction_box": [0, 0, 1, 1],
            "distribution_area_ratio": 4,
        },
    ]
    stats, hist = distributions(rows)
    group = [
        r
        for r in stats
        if r["sequence"] == r["class"] == r["other_gt_flag"] == r["truncation_group"] == "all"
        and r["subset"] == "all_none"
    ]
    iou = next(r for r in group if r["metric"] == "iou")
    ratio = next(r for r in group if r["metric"] == "area_ratio")
    assert iou["n"] == 3 and iou["median"] == 0.5 and iou["missing_prediction"] == 1
    assert ratio["n"] == 2 and ratio["median"] == 2.5
    assert any(
        r["other_gt_flag"] == "true" and r["truncation_group"] == "positive" and r["samples"] == 1 for r in stats
    )
    assert (
        sum(
            r["count"]
            for r in hist
            if all(
                r[k] == iou[k] for k in ("sequence", "class", "subset", "other_gt_flag", "truncation_group", "metric")
            )
        )
        == 3
    )


def quota_rows() -> list[dict]:
    rows = []
    for index, (sequence, reason, geometry, occlusion, count) in enumerate(IMAGE_QUOTAS):
        for i in range(count + 1):
            rows.append(
                {
                    "sequence": sequence,
                    "reason": reason,
                    "geometry": geometry,
                    "occlusion": occlusion if occlusion != "any" else "none",
                    "frame": index * 10 + i,
                    "annotation_id": f"{sequence}/{index}/{i}",
                }
            )
    return rows


def test_image_quotas_clip_occlusion_reproducible_without_replacement() -> None:
    rows = quota_rows()
    selected = select_images(rows)
    assert selected == select_images(list(reversed(rows)))
    assert len(selected) == len({r["annotation_id"] for r in selected}) == 30
    assert Counter(r["sequence"] for r in selected) == {"0013": 15, "0020": 15}
    assert Counter(r["reason"] for r in selected) == {
        "no_overlap": 10,
        "poor_localization": 15,
        "assignment_conflict": 5,
    }
    for label in ("pred_inside_gt", "gt_inside_pred", "shifted"):
        assert Counter(r["occlusion"] for r in selected if r["geometry"] == label) == {"partial": 3, "heavy": 2}
    with pytest.raises(ValueError, match='"available": 0'):
        select_images([r for r in rows if r["geometry"] != "gt_inside_pred"])


def test_report_protection_and_resume_preserve_originals(tmp_path: Path) -> None:
    protocol = miss_fixture(tmp_path)
    write_miss_analysis(protocol, root=tmp_path, output=tmp_path / "old", bootstrap_repeats=10)
    originals = {p: file_hash(p) for p in tmp_path.rglob("*") if p.is_file()}
    args = {
        "misses_csv": tmp_path / "old/misses.csv",
        "root": tmp_path,
        "quotas": (("0013", "assignment_conflict", "not_applicable", "heavy", 2),),
    }
    report = write_geometry_fixture(protocol, output=tmp_path / "new", **args)
    assert report["schema"] == "autolabel-phase06-geometry-fixture/v1"
    assert report == write_geometry_fixture(protocol, output=tmp_path / "new", **args)
    assert sum(r["count"] for r in report["cross"]) == 4
    assert all(file_hash(p) == h for p, h in originals.items())
    assert len(report["images"]) == 2
    with Image.open(tmp_path / "new" / report["images"][0]["path"]) as overlay:
        assert overlay.size == (32, 32)
        colors = {tuple(c) for c in np.asarray(overlay).reshape(-1, 3)}
        assert (0, 200, 0) in colors and (255, 0, 0) in colors
    with pytest.raises(ValueError, match="phase-0.5"):
        write_geometry_fixture(protocol, output=tmp_path / "old", **args)
    for destination in (tmp_path / "old/geometry", tmp_path / "images/geometry"):
        with pytest.raises(ValueError, match="overlaps"):
            write_geometry_fixture(protocol, output=destination, **args)
        assert not destination.exists()
    with pytest.raises(ValueError, match="fixed seed 42"):
        write_geometry_fixture(protocol, output=tmp_path / "new", seed=7, **args)
    with pytest.raises(ValueError, match="fixed seed 42"):
        write_geometry_fixture(protocol, output=tmp_path / "different", seed=7, **args)
    assert not (tmp_path / "different").exists()
    with pytest.raises(ValueError, match="incompatible"):
        write_geometry_fixture(
            protocol, output=tmp_path / "new", quotas=(), misses_csv=args["misses_csv"], root=tmp_path
        )
    (tmp_path / "new/images/unrelated.png").write_bytes(b"keep")
    with pytest.raises(ValueError, match="unrelated"):
        write_geometry_fixture(protocol, output=tmp_path / "new", **args)
    with pytest.raises(ValueError, match="source image directory"):
        write_geometry_fixture(protocol, output=tmp_path, **args)


def test_shortage_missing_source_and_mismatched_csv_fail_before_write(tmp_path: Path) -> None:
    protocol = miss_fixture(tmp_path)
    write_miss_analysis(protocol, root=tmp_path, output=tmp_path / "old", bootstrap_repeats=10)
    args = {"misses_csv": tmp_path / "old/misses.csv", "root": tmp_path, "output": tmp_path / "new"}
    with pytest.raises(ValueError, match="insufficient image strata"):
        write_geometry_analysis(protocol, **args)
    assert not (tmp_path / "new").exists()
    (tmp_path / "images/000000.png").unlink()
    with pytest.raises(FileNotFoundError, match="source image"):
        write_geometry_fixture(
            protocol, quotas=(("0013", "assignment_conflict", "not_applicable", "heavy", 2),), **args
        )
    assert not (tmp_path / "new").exists()


def thirty_image_protocol(root: Path) -> Path:
    runs = []
    boxes = {
        "no_overlap": [40, 10, 60, 30],
        "assignment_conflict": [10, 10, 30, 30],
        "pred_inside_gt": [14, 14, 26, 26],
        "gt_inside_pred": [0, 0, 40, 40],
        "shifted": [20, 10, 40, 30],
    }
    for sequence in ("0013", "0020"):
        folder = root / sequence
        folder.mkdir()
        protocol = synthetic_protocol(folder)
        run = json.loads(protocol.read_text())["runs"][0]
        result = json.loads((folder / "result.json").read_text())
        item = result["configurations"][0]
        predictions, labels = [], []
        frame = 0
        images = folder / "images"
        images.mkdir()
        for seq, reason, geometry, occlusion, count in IMAGE_QUOTAS:
            if seq != sequence:
                continue
            for _ in range(count + 1):
                box = boxes[reason if reason != "poor_localization" else geometry]
                track = 2 * frame + 1
                occluded = 1 if occlusion == "partial" else 2 if occlusion == "heavy" else 0
                for gt_id in [track, track + 1] if reason == "assignment_conflict" else [track]:
                    labels.append(f"{frame} {gt_id} Car 0 {occluded} 0 10 10 30 30 1 1 1 0 0 1 0")
                predictions.append({"frame_index": frame, "track_id": str(track), "label": "car", "bbox": box})
                Image.new("RGB", (64, 64), "white").save(images / f"{frame:06d}.png")
                frame += 1
        item.update(predictions=predictions, evaluated_source_frames=list(range(frame)), window_predictions=[])
        (folder / "result.json").write_text(json.dumps(result))
        (folder / "mapping.json").write_text(
            json.dumps({"frames": [{"frame_index": i, "source_frame_index": i} for i in range(frame)]})
        )
        (folder / "labels.txt").write_text("\n".join(labels))
        run.update(sequence=sequence, native_frame_count=frame, image_dir=f"{sequence}/images")
        for field in ("result", "mapping", "labels"):
            run[field] = f"{sequence}/{Path(run[field]).name}"
            run["sha256"][field] = file_hash(root / run[field])
        runs.append(run)
    protocol = root / "protocol.json"
    protocol.write_text(json.dumps({"schema": "autolabel-phase0-protocol/v1", "seed": 42, "runs": runs}))
    return protocol


def test_full_stratified_report_thirty_native_images_and_recall_baseline(tmp_path: Path) -> None:
    protocol = thirty_image_protocol(tmp_path)
    write_miss_analysis(protocol, root=tmp_path, output=tmp_path / "old", bootstrap_repeats=10)
    before = {p: file_hash(p) for p in tmp_path.rglob("*") if p.is_file()}
    report = write_geometry_analysis(
        protocol, root=tmp_path, misses_csv=tmp_path / "old/misses.csv", output=tmp_path / "geometry"
    )
    assert len(report["images"]) == len(report["rendered_image_sha256"]) == 30
    assert Counter(r["sequence"] for r in report["images"]) == {"0013": 15, "0020": 15}
    assert all(file_hash(p) == checksum for p, checksum in before.items())
    with (tmp_path / "old/misses.csv").open(newline="") as stream:
        original = list(csv.DictReader(stream))
    with (tmp_path / "geometry/misses-geometry.csv").open(newline="") as stream:
        enhanced = list(csv.DictReader(stream))
    assert [{k: r[k] for k in original[0]} for r in enhanced] == original
    for run in json.loads(protocol.read_text())["runs"]:
        records, _ = analyze_direct(load_frozen_run(run, tmp_path))
        for threshold, field in ((0.5, "matched_05"), (0.7, "matched_07")):
            row = next(
                r
                for r in report["recall"]
                if r["sequence"] == run["sequence"] and r["class"] == "all" and r["iou"] == threshold
            )
            assert row["tp"] == sum(r[field] for r in records)
    assert report == write_geometry_analysis(
        protocol, root=tmp_path, misses_csv=tmp_path / "old/misses.csv", output=tmp_path / "geometry"
    )


def test_phase06_cli_dispatch_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import sys

    from scripts.evaluate_benchmark import main

    protocol = thirty_image_protocol(tmp_path)
    write_miss_analysis(protocol, root=tmp_path, output=tmp_path / "old", bootstrap_repeats=10)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_benchmark",
            "phase0-geometry",
            "--protocol",
            str(protocol),
            "--misses",
            str(tmp_path / "old/misses.csv"),
            "--root",
            str(tmp_path),
            "--output",
            str(tmp_path / "new"),
        ],
    )
    main()
    assert json.loads(capsys.readouterr().out)["services_called"] == []
    assert (tmp_path / "new/sample-manifest.json").is_file()


def test_relative_root_default_images_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    protocol = miss_fixture(artifact)
    config = json.loads(protocol.read_text())
    del config["runs"][0]["image_dir"]
    protocol.write_text(json.dumps(config))
    image_dir = tmp_path / "image_02/0013"
    image_dir.mkdir(parents=True)
    for frame in range(5):
        Image.new("RGB", (32, 32), "white").save(image_dir / f"{frame:06d}.png")
    write_miss_analysis(protocol, root=artifact, output=tmp_path / "old", bootstrap_repeats=10)
    monkeypatch.chdir(tmp_path)
    report = write_geometry_fixture(
        Path("artifact/protocol.json"),
        root=Path("artifact"),
        misses_csv=Path("old/misses.csv"),
        output=Path("new"),
        quotas=(("0013", "assignment_conflict", "not_applicable", "heavy", 2),),
    )
    assert len(report["images"]) == 2
