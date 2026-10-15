from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.services.evaluation.misses import analyze_direct, bootstrap_recall, miss_reason, write_miss_analysis
from src.services.evaluation.phase0 import file_hash, load_frozen_run, write_baseline
from tests.test_scripts.test_phase0_cli import synthetic_protocol


@pytest.mark.parametrize(
    "overlap,same_class,expected",
    [
        (0.0999, True, "no_overlap"),
        (0.1, True, "poor_localization"),
        (0.4999, False, "poor_localization"),
        (0.5, False, "wrong_class"),
        (0.5, True, "assignment_conflict"),
    ],
)
def test_reason_boundaries(overlap: float, same_class: bool, expected: str) -> None:
    assert miss_reason(overlap, same_class=same_class) == expected


def miss_fixture(root: Path) -> Path:
    protocol = synthetic_protocol(root)
    result = json.loads((root / "result.json").read_text())
    # Frames 0 and 2 each have a duplicated GT extent. One prediction can match
    # only GT 1; GT 2 must be an assignment conflict. At 4, a wrong-class box.
    result["configurations"][0]["predictions"][-1]["label"] = "person"
    (root / "result.json").write_text(json.dumps(result))
    lines = []
    for frame in range(5):
        for track in (1, 2):
            lines.append(f"{frame} {track} Car 1 {0 if track == 1 else 2} 0 10 10 20 20 1 1 1 0 0 1 0")
    (root / "labels.txt").write_text("\n".join(lines))
    images = root / "images"
    images.mkdir()
    for frame in range(5):
        Image.new("RGB", (32, 32), "white").save(images / f"{frame:06d}.png")
    config = json.loads(protocol.read_text())
    run = config["runs"][0]
    run["image_dir"] = "images"
    for field in ("result", "labels"):
        run["sha256"][field] = file_hash(root / run[field])
    protocol.write_text(json.dumps(config))
    return protocol


def test_miss_mapping_all_classes_conflicts_cross_tables_and_images(tmp_path: Path) -> None:
    protocol = miss_fixture(tmp_path)
    before = {
        p: file_hash(p) for p in [tmp_path / "result.json", tmp_path / "labels.txt", *tmp_path.glob("images/*.png")]
    }
    report = write_miss_analysis(protocol, root=tmp_path, output=tmp_path / "out", bootstrap_repeats=50, image_count=30)
    assert report["population"]["gt"] == 6
    assert report["population"]["misses"] == 4
    assert report["reason_counts"] == {"assignment_conflict": 2, "wrong_class": 2}
    import csv

    with (tmp_path / "out/misses.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["frame"] == "0" and rows[1]["frame"] == "2"
    assert rows[1]["frame_sampled"] == "1"
    assert rows[0]["best_prediction_matched_gt_id"] == "1"
    assert rows[0]["truncation"] == "1.0" and rows[0]["occlusion"] == "heavy"
    assert rows[-1]["best_prediction_class"] == "person"
    for name in ("occlusion-size", "occlusion-class"):
        with (tmp_path / f"out/{name}.csv").open(newline="") as stream:
            cross = list(csv.DictReader(stream))
        selected = [x for x in cross if x["sequence"] == "0013" and x["iou"] == "0.5"]
        assert sum(int(x["gt"]) for x in selected) == 6
        assert sum(int(x["tp"]) for x in selected) == 2
        assert any(x["recall"] == "" for x in selected)
    assert len(report["images"]) == 4
    with Image.open(tmp_path / "out" / report["images"][0]["path"]) as overlay:
        assert overlay.size == (32, 32)
        # Coincident red/green boxes are inset/styled to keep both visible.
        colors = {tuple(x) for x in np.asarray(overlay.convert("RGB")).reshape(-1, 3)}
        assert (0, 200, 0) in colors and (255, 0, 0) in colors
    assert all(file_hash(p) == checksum for p, checksum in before.items())
    repeated = write_miss_analysis(
        protocol, root=tmp_path, output=tmp_path / "again", bootstrap_repeats=50, image_count=30
    )
    assert repeated["images"] == report["images"]
    assert repeated["bootstrap"] == report["bootstrap"]
    same_output = write_miss_analysis(
        protocol, root=tmp_path, output=tmp_path / "out", bootstrap_repeats=50, image_count=30
    )
    assert same_output == report
    with pytest.raises(ValueError, match="different image sample"):
        write_miss_analysis(protocol, root=tmp_path, output=tmp_path / "out", bootstrap_repeats=50, image_count=2)
    assert len(list((tmp_path / "out/images").glob("*.png"))) == 4
    assert json.loads((tmp_path / "out/summary.json").read_text()) == report


def test_bootstrap_clusters_preserve_track_length_and_sequence_namespace() -> None:
    # One long perfect track and one short fully missed track: micro recall 10/11,
    # not the unweighted mean .5. Repeated draws retain all 10 observations.
    rows = [
        {
            "sequence": "a",
            "gt_id": "1",
            "class": "car",
            "occlusion": "none",
            "size": "small",
            "matched_05": True,
            "matched_07": True,
        }
        for _ in range(10)
    ]
    rows += [
        {
            "sequence": "a",
            "gt_id": "2",
            "class": "car",
            "occlusion": "none",
            "size": "small",
            "matched_05": False,
            "matched_07": False,
        }
    ]
    rows += [
        {
            "sequence": "b",
            "gt_id": "1",
            "class": "person",
            "occlusion": "unknown",
            "size": "large",
            "matched_05": False,
            "matched_07": False,
        }
    ]
    report = bootstrap_recall(rows, repeats=200, seed=42)
    assert report == bootstrap_recall(rows, repeats=200, seed=42)
    overall = next(x for x in report["overall"] if x["sequence"] == "a" and x["iou"] == 0.5)
    assert overall["recall"] == pytest.approx(10 / 11)
    assert overall["support_tracks"] == 2
    assert overall["ci_low"] == 0 and overall["ci_high"] == 1
    combined = next(x for x in report["overall"] if x["sequence"] == "all" and x["iou"] == 0.5)
    assert combined["support_tracks"] == 3
    group = next(
        x
        for x in report["occlusion_class"]
        if x["sequence"] == "all" and x["class"] == "person" and x["occlusion"] == "unknown"
    )
    assert group["ci_status"] == "insufficient_tracks" and group["ci_low"] is None
    assert report["method"] == "stratified_GT_track_percentile_bootstrap"
    assert np.isfinite(overall["ci_high"])


def test_missing_images_and_input_hash_fail_without_fake_evidence(tmp_path: Path) -> None:
    protocol = miss_fixture(tmp_path)
    (tmp_path / "images/000000.png").unlink()
    with pytest.raises(FileNotFoundError, match="source image"):
        write_miss_analysis(protocol, root=tmp_path, output=tmp_path / "out", bootstrap_repeats=10)
    assert not (tmp_path / "out/summary.json").exists()
    (tmp_path / "result.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        write_miss_analysis(protocol, root=tmp_path, output=tmp_path / "bad", bootstrap_repeats=10)


def test_phase0_report_unchanged_after_shared_reader_refactor(tmp_path: Path) -> None:
    protocol = synthetic_protocol(tmp_path)
    first = write_baseline(protocol, root=tmp_path, output=tmp_path / "a")
    second = write_baseline(protocol, root=tmp_path, output=tmp_path / "b")
    assert first == second


def test_absent_prediction_ties_and_unknown_occlusion(tmp_path: Path) -> None:
    protocol = synthetic_protocol(tmp_path)
    run = json.loads(protocol.read_text())["runs"][0]
    loaded = load_frozen_run(run, tmp_path)
    rows, misses = analyze_direct(replace(loaded, direct=[], occlusions={}))
    assert len(misses) == 3
    assert all(
        x["best_iou"] == 0 and x["best_prediction_class"] is None and x["reason"] == "no_overlap" for x in misses
    )
    assert all(x["occlusion"] == "unknown" for x in rows)
    # Tie order stays artifact order even when classes differ.
    from scripts.eval_autolabel import Box

    changed = [Box(0, "person", (10, 10, 20, 20), "first"), Box(0, "truck", (10, 10, 20, 20), "second")]
    _, misses = analyze_direct(replace(loaded, direct=changed))
    assert misses[0]["best_prediction_track_id"] == "first"
    assert misses[0]["best_prediction_class"] == "person"


def test_size_boundaries_and_bootstrap_empty_replicates(tmp_path: Path) -> None:
    protocol = synthetic_protocol(tmp_path)
    loaded = load_frozen_run(json.loads(protocol.read_text())["runs"][0], tmp_path)
    sizes = ((0, 0, 31, 32), (0, 0, 32, 32), (0, 0, 96, 96))
    sampled_gt = [box for box in loaded.gt if int(box.frame.frame_uid) in loaded.source]
    gt = tuple(box.model_copy(update={"bbox_xyxy": sizes[i]}) for i, box in enumerate(sampled_gt))
    rows, _ = analyze_direct(replace(loaded, gt=gt, direct=[]))
    assert [x["size"] for x in rows] == ["small", "medium", "large"]
    # Two supporting car tracks out of four tracks; some resamples contain only
    # person tracks, so that cell's zero denominators must not become recall 0.
    records = [
        {
            "sequence": "s",
            "gt_id": str(i),
            "class": "car" if i < 2 else "person",
            "occlusion": "none",
            "size": "small",
            "matched_05": True,
            "matched_07": True,
        }
        for i in range(4)
    ]
    result = bootstrap_recall(records, repeats=1000)
    group = next(
        x
        for x in result["occlusion_class"]
        if x["sequence"] == "s" and x["class"] == "car" and x["occlusion"] == "none"
    )
    assert 0 < group["bootstrap_zero_denominator"] < 1000
    assert group["bootstrap_valid"] + group["bootstrap_zero_denominator"] == 1000
    assert group["ci_low"] == group["ci_high"] == 1
    with pytest.raises(ValueError, match="track IDs"):
        bootstrap_recall([{**records[0], "gt_id": None}])


def test_thirty_image_sample_is_reproducible_without_replacement(tmp_path: Path) -> None:
    protocol = miss_fixture(tmp_path)
    source = list(range(0, 61, 2))
    (tmp_path / "mapping.json").write_text(
        json.dumps({"frames": [{"frame_index": i, "source_frame_index": f} for i, f in enumerate(source)]})
    )
    data = json.loads((tmp_path / "result.json").read_text())
    item = data["configurations"][0]
    item["evaluated_source_frames"] = list(range(len(source)))
    item["predictions"] = []
    (tmp_path / "result.json").write_text(json.dumps(data))
    (tmp_path / "labels.txt").write_text("\n".join(f"{f} 1 Car 0 0 0 10 10 20 20 1 1 1 0 0 1 0" for f in range(61)))
    for f in source:
        Image.new("RGB", (32, 32), "white").save(tmp_path / f"images/{f:06d}.png")
    config = json.loads(protocol.read_text())
    run = config["runs"][0]
    run["native_frame_count"] = 61
    for field in ("result", "mapping", "labels"):
        run["sha256"][field] = file_hash(tmp_path / run[field])
    protocol.write_text(json.dumps(config))
    a = write_miss_analysis(protocol, root=tmp_path, output=tmp_path / "a", bootstrap_repeats=10)
    b = write_miss_analysis(protocol, root=tmp_path, output=tmp_path / "b", bootstrap_repeats=10)
    assert len(a["images"]) == len({x["annotation_id"] for x in a["images"]}) == 30
    assert a["images"] == b["images"]
    assert len(list((tmp_path / "a/images").glob("*.png"))) == 30


def test_excluded_class_counts_use_sampled_frames_without_changing_baseline(tmp_path: Path) -> None:
    protocol = synthetic_protocol(tmp_path)
    labels = tmp_path / "labels.txt"
    labels.write_text(
        labels.read_text() + "\n" + "\n".join(f"{f} 3 Van 0 0 0 0 0 10 10 1 1 1 0 0 1 0" for f in range(5))
    )
    config = json.loads(protocol.read_text())
    config["runs"][0]["sha256"]["labels"] = file_hash(labels)
    protocol.write_text(json.dumps(config))
    loaded = load_frozen_run(config["runs"][0], tmp_path)
    assert loaded.excluded_classes == {"Van": 5}
    assert loaded.excluded_sampled_classes == {"Van": 3}


def test_output_cannot_overwrite_optional_cvat_export(tmp_path: Path) -> None:
    protocol = miss_fixture(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    export = output / "misses.csv"
    export.write_text("<annotations><version>1.1</version></annotations>")
    before = file_hash(export)
    config = json.loads(protocol.read_text())
    config["runs"][0]["cvat_export"] = "out/misses.csv"
    config["runs"][0]["sha256"]["cvat_export"] = before
    protocol.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="overwrite an input"):
        write_miss_analysis(protocol, root=tmp_path, output=output, bootstrap_repeats=10)
    assert file_hash(export) == before
    assert not (output / "summary.json").exists()
