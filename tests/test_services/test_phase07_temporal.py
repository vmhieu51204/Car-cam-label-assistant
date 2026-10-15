from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from scripts.eval_autolabel import Box
from src.services.evaluation.geometry import GEOMETRY_FIELDS, OTHER_FIELDS, direct_ledger
from src.services.evaluation.misses import MISS_FIELDS, write_diagnostic_csv
from src.services.evaluation.phase0 import canonical_predictions, file_hash, load_frozen_run
from src.services.evaluation.temporal import (
    Anchor,
    bridge_ledger,
    bridge_tables,
    eligibility,
    eligibility_tables,
    expand_at_prediction_center,
    gt_blind_gap_fill,
    interpolate,
    nearest_pair,
    size_memory_rows,
    stitched_evidence,
    validate_geometry_csv,
    write_temporal_analysis,
)
from tests.test_scripts.test_phase0_cli import synthetic_protocol


def temporal_fixture(root: Path, *, missing: bool = False, split_ids: bool = False) -> tuple[Path, Path]:
    protocol = synthetic_protocol(root)
    config = json.loads(protocol.read_text())
    result = json.loads((root / "result.json").read_text())
    item = result["configurations"][0]
    if missing:
        del item["predictions"][1]
    else:
        item["predictions"][1]["bbox"] = [12, 12, 18, 18]
    if split_ids:
        item["predictions"][-1]["track_id"] = "car:2"
    item["window_predictions"] = [
        {"window_id": int(split_ids and p["frame_index"] == 2), **p} for p in item["predictions"]
    ]
    (root / "result.json").write_text(json.dumps(result))
    config["runs"][0]["sha256"]["result"] = file_hash(root / "result.json")
    protocol.write_text(json.dumps(config))
    loaded = load_frozen_run(config["runs"][0], root)
    _, misses = direct_ledger(loaded)
    geometry = root / "geometry/misses-geometry.csv"
    geometry.parent.mkdir()
    write_diagnostic_csv(geometry, misses, [*MISS_FIELDS, *GEOMETRY_FIELDS, "geometry", *OTHER_FIELDS])
    return protocol, geometry


def loaded_fixture(root: Path, **kwargs: bool):
    protocol, _ = temporal_fixture(root, **kwargs)
    return load_frozen_run(json.loads(protocol.read_text())["runs"][0], root)


def test_source_time_interpolation_and_no_extrapolation() -> None:
    assert interpolate((0, 0, 10, 10), (10, 10, 20, 20), 0, 10, 2) == (2, 2, 12, 12)
    with pytest.raises(ValueError, match="bracketing"):
        interpolate((0, 0, 10, 10), (10, 10, 20, 20), 0, 10, 10)


def test_bridgeable_uses_sampled_distance_both_sides_and_k_boundary(tmp_path: Path) -> None:
    loaded = loaded_fixture(tmp_path)
    prediction = canonical_predictions(loaded.direct, loaded.keys, "fixture")[0]
    a, b = Anchor(0, 0, {}, prediction, "a"), Anchor(20, 100, {}, prediction, "a")
    assert nearest_pair([a, b], 10, 10) == (a, b)
    assert nearest_pair([a, b], 10, 5) == (None, None)
    assert nearest_pair([a], 10, 10) == (a, None)
    assert nearest_pair([a, b], 0, 10) == (None, None)


def test_oracle_vs_same_id_and_gt_blind_no_replacing_poor_boxes(tmp_path: Path) -> None:
    loaded = loaded_fixture(tmp_path)
    ledger, _ = direct_ledger(loaded)
    rows, meta = bridge_ledger(loaded, ledger)
    assert len(rows) == 3 and all(r["bridgeable"] for r in rows)
    assert all(r["oracle_iou"] == r["id_gated_iou"] == 1 for r in rows)
    assert all(r["gt_blind_fill_recovered_05"] is False for r in rows)
    assert meta["stitched_available"] and meta["direct_ids_unmapped"] == 0
    assert all(r["fill_candidates"] == 0 for r in meta["gt_blind"])
    table = next(
        r
        for r in bridge_tables(rows)
        if r["sequence"] == "all" and r["dimension"] == "overall" and r["k"] == 2 and r["iou"] == 0.5
    )
    assert table["bridgeable_fraction"] == table["oracle_gap_fill_recall_bridgeable"] == 1
    assert table["identity_lost_recoveries"] == 0


def test_id_disagreement_isolates_loss_without_changing_endpoints(tmp_path: Path) -> None:
    loaded = loaded_fixture(tmp_path, split_ids=True)
    ledger, _ = direct_ledger(loaded)
    rows, _ = bridge_ledger(loaded, ledger)
    assert all(r["oracle_iou"] == 1 and r["same_stitched_id"] is False and r["id_gated_iou"] is None for r in rows)
    table = next(
        r
        for r in bridge_tables(rows)
        if r["sequence"] == "all" and r["dimension"] == "overall" and r["k"] == 2 and r["iou"] == 0.5
    )
    assert table["identity_lost_recoveries"] == 1 and table["id_gated_recovered"] == 0


def test_actual_gap_candidate_and_one_to_one_matching(tmp_path: Path) -> None:
    loaded = loaded_fixture(tmp_path, missing=True)
    ledger, _ = direct_ledger(loaded)
    rows, meta = bridge_ledger(loaded, ledger)
    assert all(r["gt_blind_fill_recovered_05"] is True for r in rows)
    assert all(r["original_tp"] == 2 and r["with_fill_tp"] == 3 and r["fill_candidates"] == 1 for r in meta["gt_blind"])
    duplicated = (
        *loaded.gt,
        loaded.gt[2].model_copy(update={"annotation_id": "duplicate", "native_track_id": "2", "source_ordinal": 50}),
    )
    # One synthesized box must not recover both coincident GT objects.
    probe = replace(loaded, gt=duplicated)
    extra_ledger = [
        {**r, "annotation_id": "duplicate", "gt_id": "2", "matched_05": False} for r in ledger if r["frame"] == 2
    ]
    _, evidence = bridge_ledger(probe, [*ledger, *extra_ledger])
    assert all(r["fill_matched"] == 1 for r in evidence["gt_blind"])


def test_scene_cut_blocks_both_temporal_variants(tmp_path: Path) -> None:
    loaded = loaded_fixture(tmp_path, missing=True)
    records = [{**r, "scene_cut_before": i == 1} for i, r in enumerate(loaded.records)]
    probe = replace(loaded, records=records)
    ledger, _ = direct_ledger(probe)
    rows, evidence = bridge_ledger(probe, ledger)
    assert all(not r["bridgeable"] and r["cut_between_anchors"] and r["oracle_iou"] is None for r in rows)
    assert all(r["fill_candidates"] == 0 for r in evidence["gt_blind"])


def test_absent_stitched_artifact_is_unavailable_not_raw_id_fallback(tmp_path: Path) -> None:
    loaded = loaded_fixture(tmp_path)
    probe = replace(loaded, item={k: v for k, v in loaded.item.items() if k != "window_predictions"})
    ledger, _ = direct_ledger(probe)
    rows, evidence = bridge_ledger(probe, ledger)
    assert evidence["stitched_available"] is False
    assert all(
        r["bridgeable"] and r["same_stitched_id"] is None and r["gt_blind_fill_recovered_05"] is None for r in rows
    )


def test_gt_blind_candidate_generation_does_not_read_gt(tmp_path: Path) -> None:
    loaded = loaded_fixture(tmp_path, missing=True)
    boxes = [Box(0, "car", (0, 0, 10, 10), "global"), Box(4, "car", (10, 0, 20, 10), "global")]
    expected = gt_blind_gap_fill(loaded, boxes, 2)
    assert expected == gt_blind_gap_fill(replace(loaded, gt=()), boxes, 2)
    assert expected[0].frame == 2 and expected[0].xyxy == (5, 0, 15, 10)
    assert not gt_blind_gap_fill(loaded, [boxes[0], Box(4, "car", boxes[1].xyxy, "different")], 2)


def test_size_memory_oracle_expands_at_prediction_center_and_normalizes_scale(tmp_path: Path) -> None:
    loaded = loaded_fixture(tmp_path)
    ledger, _ = direct_ledger(loaded)
    rows = size_memory_rows(loaded, ledger)
    assert len(rows) == 3 and all(r["anchor_side"] == "past" for r in rows)
    assert all(
        r["gt_scale_oracle"] == r["width_ratio_scale_normalized"] == r["height_ratio_scale_normalized"] == 1
        for r in rows
    )
    assert all(r["gt_current_size_oracle_iou"] == r["gt_anchor_size_normalized_oracle_iou"] == 1 for r in rows)
    assert expand_at_prediction_center([0, 0, 20, 10], 10, 10) == (0, 0, 20, 10)
    assert expand_at_prediction_center([5, 5, 15, 15], 20, 20) == (0, 0, 20, 20)
    # Current GT 20x20, anchor GT 10x10: the oracle scale is 2, while
    # scale-normalized shape ratios remain 1; no GT center is used.
    changed = [{**r, "gt_box": [5, 5, 25, 25]} if r["frame"] == 2 else r for r in ledger]
    rows = size_memory_rows(loaded, changed)
    assert all(r["gt_scale_oracle"] == 2 and r["width_ratio_scale_normalized"] == 1 for r in rows)


@pytest.mark.parametrize(
    "height,occlusion,truncation,tier,height_group",
    [
        (14.9, "none", 0, "small_lt25", "lt15"),
        (15, "heavy", 1, "small_lt25", "15_to_lt25"),
        (25, "partial", 0, "partial_untruncated_ge25", "25_to_lt40"),
        (40, "heavy", 0, "heavy_untruncated_ge25", "ge40"),
        (40, "none", 1, "truncated_ge25", "ge40"),
        (40, "unknown", None, "truncation_unknown_ge25", "ge40"),
    ],
)
def test_eligibility_boundaries_unadjusted_raw_population(
    height: float, occlusion: str, truncation: float | None, tier: str, height_group: str
) -> None:
    actual = eligibility({"box_height": height, "occlusion": occlusion, "truncation": truncation})
    assert actual[:2] == (tier, height_group)


def test_track_bootstrap_namespace_reproducibility_and_low_support() -> None:
    common = {"occlusion": "none", "class": "car", "box_height": 40, "truncation": 0}
    ledger = [{**common, "sequence": "a", "gt_id": "1", "matched_05": True} for _ in range(10)]
    ledger += [
        {**common, "sequence": "a", "gt_id": "2", "matched_05": False},
        {**common, "sequence": "b", "gt_id": "1", "matched_05": False},
    ]
    table = eligibility_tables(ledger, repeats=200)
    assert table == eligibility_tables(ledger, repeats=200)
    total = next(r for r in table if r["sequence"] == "all" and r["dimension"] == "overall")
    assert total["recall"] == 10 / 12 and total["support_tracks"] == 3
    single = next(r for r in table if r["sequence"] == "b" and r["dimension"] == "overall")
    assert single["ci_status"] == "insufficient_tracks" and single["ci_low"] is None
    with pytest.raises(ValueError, match="positive"):
        eligibility_tables(ledger, repeats=0)
    with pytest.raises(ValueError, match="GT ID"):
        eligibility_tables([{**ledger[0], "gt_id": None}])


def test_temporal_report_input_hashes_preservation_and_output_guard(tmp_path: Path) -> None:
    protocol, geometry = temporal_fixture(tmp_path)
    before = {p: file_hash(p) for p in tmp_path.rglob("*") if p.is_file()}
    report = write_temporal_analysis(
        protocol, geometry_csv=geometry, root=tmp_path, output=tmp_path / "out", bootstrap_repeats=100
    )
    assert report["population"] == {"gt": 3, "misses": 1}
    assert report["inputs_unchanged"] and report["services_called"] == [] and report["annotation_change"] is False
    assert report == write_temporal_analysis(
        protocol, geometry_csv=geometry, root=tmp_path, output=tmp_path / "out", bootstrap_repeats=100
    )
    assert all(file_hash(p) == h for p, h in before.items())
    with pytest.raises(ValueError, match="overlaps"):
        write_temporal_analysis(
            protocol, geometry_csv=geometry, root=tmp_path, output=geometry.parent / "child", bootstrap_repeats=10
        )
    with pytest.raises(ValueError, match="incompatible"):
        write_temporal_analysis(
            protocol, geometry_csv=geometry, root=tmp_path, output=tmp_path / "out", bootstrap_repeats=10
        )
    geometry.write_text(geometry.read_text().replace("pred_inside_gt", "shifted"))
    loaded = load_frozen_run(json.loads(protocol.read_text())["runs"][0], tmp_path)
    with pytest.raises(ValueError, match="mismatch"):
        validate_geometry_csv(geometry, direct_ledger(loaded)[1])


def test_class_aware_matching_does_not_make_wrong_class_an_anchor(tmp_path: Path) -> None:
    loaded = loaded_fixture(tmp_path)
    wrong = [Box(b.frame, "person", b.xyxy, b.track_id) for b in loaded.direct]
    probe = replace(loaded, direct=wrong)
    ledger, _ = direct_ledger(probe)
    rows, _ = bridge_ledger(probe, ledger)
    assert not any(r["bridgeable"] for r in rows)
    assert Counter(r["reason"] for r in rows)["wrong_class"] == 6


def test_initial_global_id_mapping_and_explicit_outside_rejection(tmp_path: Path) -> None:
    loaded = loaded_fixture(tmp_path)
    _, identities, _ = stitched_evidence(loaded)
    assert len(set(identities.values())) == 1 and None not in identities.values()
    outside = [{**r, "outside": True} for r in loaded.item["window_predictions"]]
    with pytest.raises(ValueError, match="outside"):
        stitched_evidence(replace(loaded, item={**loaded.item, "window_predictions": outside}))
