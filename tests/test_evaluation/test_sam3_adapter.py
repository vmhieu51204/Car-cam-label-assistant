"""Tests for SAM3 prediction normalization adapter."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from src.models import evaluation as e
from src.services.evaluation.predictions.sam3 import load_sam3_predictions

HASH = "a" * 64


@pytest.fixture
def test_manifest() -> e.BenchmarkManifest:
    sequence = e.SequenceKey(dataset_id="kitti", split="training", sequence_uid="0013")
    keys = tuple(e.FrameKey(sequence=sequence, frame_uid=str(i)) for i in range(3))
    profile = e.ReferenceProfile(
        profile_id="kitti-raw",
        revision="kitti-raw/v1",
        designation="raw",
        classes=("car", "person"),
        class_map=(
            e.ClassMapping(native_class="Car", class_name="car"),
            e.ClassMapping(native_class="Pedestrian", class_name="person"),
        ),
        eligibility_policy="native-targets/v1",
        geometry_policy="native-unclipped/v1",
    )
    bindings = tuple(
        e.FrameBinding(
            mapping_id="payload/v1",
            domain="payload",
            index=i,
            frame=key,
            upstream=(e.BindingStep(mapping_id="native/v1", domain="native_export", index=i),),
        )
        for i, key in enumerate(keys)
    )
    options = e.PredictionNormalizationOptions(
        views=("raw", "stitched"),
        reset_boundaries=(
            e.ResetBoundary(config_id="cfg-1", frame=keys[1], mapping_id="payload/v1", source="payload_declared"),
        ),
    )
    return e.BenchmarkManifest(
        protocol_revision="fixture/v1",
        reference_revision="fixture/v1",
        reference_sha256=HASH,
        sequences=(sequence,),
        planned_frames=keys,
        cohort="text_discovery",
        seed_set=e.SeedSet(revision="none/v1"),
        profile=profile,
        bindings=bindings,
        normalization=options,
        cadence=(
            e.CadencePolicy(
                sequence=sequence,
                source_fps=10.0,
                source_fps_provenance="kitti_dataset_fallback",
                input_fps=10.0,
                target_fps=5.0,
                sampling_policy="uniform/v1",
                endpoint_policy="include_final_source_frame",
            ),
        ),
        evaluator_revision="fixture",
    )


def test_load_benchmark_all_views_scored(tmp_path: Path, test_manifest: e.BenchmarkManifest) -> None:
    result_doc = {
        "schema": "autolabel-sam3-benchmark/v1",
        "completed": True,
        "configurations": [
            {
                "id": "cfg-1",
                "status": "completed",
                "config": {"mode": "text_prompt"},
                "selected_source_frames": [0, 1, 2],
                "evaluated_source_frames": [0, 1, 2],
                "window_predictions": [
                    {
                        "window_id": 0,
                        "frame_index": 0,
                        "track_id": 1,
                        "label": "car",
                        "bbox": [10.0, 15.0, 50.0, 80.0],
                        "score": 0.95,
                    },
                    {
                        "window_id": 0,
                        "frame_index": 1,
                        "track_id": 1,
                        "label": "car",
                        "bbox": [12.0, 16.0, 52.0, 82.0],
                        "score": 0.92,
                    },
                    {
                        "window_id": 1,
                        "frame_index": 1,
                        "track_id": 1,
                        "label": "car",
                        "bbox": [13.0, 17.0, 53.0, 83.0],
                        "score": 0.91,
                    },
                    {
                        "window_id": 1,
                        "frame_index": 2,
                        "track_id": 1,
                        "label": "car",
                        "bbox": [15.0, 18.0, 55.0, 85.0],
                        "score": 0.90,
                    },
                ],
            }
        ],
    }
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result_doc), encoding="utf-8")

    runs = load_sam3_predictions(result_path, manifest=test_manifest, options=test_manifest.normalization)
    assert len(runs) == 2
    raw_run = next(r for r in runs if r.view == "raw")
    stitched_run = next(r for r in runs if r.view == "stitched")

    assert raw_run.status == "completed"
    assert raw_run.raw_status == "completed"
    assert len(raw_run.processing) == 3
    assert all(p.state == "processed" and p.evidence == "wrapper_declared" for p in raw_run.processing)

    # In raw run, frame 1 takes first window observation
    f1_raw = [b for b in raw_run.boxes if b.frame.frame_uid == "1"]
    assert len(f1_raw) == 1
    assert f1_raw[0].source.window_id == "0"
    assert f1_raw[0].score.value == 0.92
    assert f1_raw[0].score.semantics == "probability"

    # In stitched run, stitching assigns a stitched_track_id
    assert all(b.stitched_track_id is not None for b in stitched_run.boxes)
    assert raw_run.content_sha256 is not None
    assert stitched_run.content_sha256 is not None


def test_completed_empty_run(tmp_path: Path, test_manifest: e.BenchmarkManifest) -> None:
    result_doc = {
        "schema": "autolabel-sam3-benchmark/v1",
        "completed": True,
        "configurations": [
            {
                "id": "cfg-1",
                "status": "completed",
                "config": {"mode": "text_prompt"},
                "selected_source_frames": [0, 1, 2],
                "evaluated_source_frames": [0, 1, 2],
                "predictions": [],
            }
        ],
    }
    result_path = tmp_path / "empty_result.json"
    result_path.write_text(json.dumps(result_doc), encoding="utf-8")

    runs = load_sam3_predictions(result_path, manifest=test_manifest, options=test_manifest.normalization)
    assert len(runs) == 2
    for run in runs:
        assert run.status == "completed"
        assert len(run.boxes) == 0
        assert len(run.processing) == 3
        assert all(p.state == "processed" for p in run.processing)


def test_failed_partial_run_retains_observations(tmp_path: Path, test_manifest: e.BenchmarkManifest) -> None:
    result_doc = {
        "schema": "autolabel-sam3-benchmark/v1",
        "completed": False,
        "configurations": [
            {
                "id": "cfg-1",
                "status": "failed",
                "error": "CUDA out of memory",
                "config": {"mode": "text_prompt"},
                "selected_source_frames": [0, 1, 2],
                "evaluated_source_frames": [0],
                "predictions": [
                    {
                        "frame_index": 0,
                        "track_id": 1,
                        "label": "car",
                        "bbox": [10.0, 15.0, 50.0, 80.0],
                        "score": 0.88,
                    }
                ],
            }
        ],
    }
    result_path = tmp_path / "failed_result.json"
    result_path.write_text(json.dumps(result_doc), encoding="utf-8")

    runs = load_sam3_predictions(result_path, manifest=test_manifest, options=test_manifest.normalization)
    assert len(runs) == 2
    for run in runs:
        assert run.status == "failed"
        assert run.raw_status == "failed"
        assert run.failure_reason == "CUDA out of memory"
        assert len(run.boxes) == 1
        # Evaluated frame 0 is processed, frames 1 and 2 are failed/unprocessed
        p0 = next(p for p in run.processing if p.frame.frame_uid == "0")
        assert p0.state == "processed"
        p1 = next(p for p in run.processing if p.frame.frame_uid == "1")
        assert p1.state == "unknown"
        assert p1.evidence == "unavailable"


def test_outside_rows_separated_into_track_states(tmp_path: Path, test_manifest: e.BenchmarkManifest) -> None:
    result_doc = {
        "schema": "autolabel-sam3-benchmark/v1",
        "completed": True,
        "configurations": [
            {
                "id": "cfg-1",
                "status": "completed",
                "config": {"mode": "text_prompt"},
                "selected_source_frames": [0, 1],
                "evaluated_source_frames": [0, 1],
                "predictions": [
                    {
                        "frame_index": 0,
                        "track_id": 1,
                        "label": "car",
                        "bbox": [10.0, 15.0, 50.0, 80.0],
                        "score": 0.9,
                        "outside": False,
                    },
                    {
                        "frame_index": 1,
                        "track_id": 1,
                        "label": "car",
                        "bbox": [10.0, 15.0, 50.0, 80.0],
                        "outside": True,
                    },
                ],
            }
        ],
    }
    result_path = tmp_path / "outside_result.json"
    result_path.write_text(json.dumps(result_doc), encoding="utf-8")

    runs = load_sam3_predictions(result_path, manifest=test_manifest, options=test_manifest.normalization)
    raw_run = next(r for r in runs if r.view == "raw")
    assert len(raw_run.boxes) == 1
    assert len(raw_run.track_states) == 1
    assert raw_run.track_states[0].outside is True
    assert raw_run.track_states[0].score.semantics == "missing"


def test_skipped_budget_and_scoreless(tmp_path: Path, test_manifest: e.BenchmarkManifest) -> None:
    result_doc = {
        "schema": "autolabel-sam3-benchmark/v1",
        "completed": True,
        "configurations": [
            {
                "id": "cfg-1",
                "status": "skipped_budget",
                "error": "time limit reached",
                "predictions": [],
            }
        ],
    }
    result_path = tmp_path / "skipped_result.json"
    result_path.write_text(json.dumps(result_doc), encoding="utf-8")

    runs = load_sam3_predictions(result_path, manifest=test_manifest, options=test_manifest.normalization)
    assert len(runs) == 2
    assert all(r.status == "skipped_budget" and r.raw_status == "skipped_budget" for r in runs)


def test_mask_rle_validation_and_reference(tmp_path: Path, test_manifest: e.BenchmarkManifest) -> None:
    result_doc = {
        "schema": "autolabel-sam3-benchmark/v1",
        "completed": True,
        "configurations": [
            {
                "id": "cfg-1",
                "status": "completed",
                "selected_source_frames": [0],
                "evaluated_source_frames": [0],
                "predictions": [
                    {
                        "frame_index": 0,
                        "track_id": 1,
                        "label": "car",
                        "bbox": [10.0, 10.0, 20.0, 20.0],
                        "mask_rle": {"width": 100, "height": 100, "counts": [9900, 100]},
                    }
                ],
            }
        ],
    }
    result_path = tmp_path / "mask_result.json"
    result_path.write_text(json.dumps(result_doc), encoding="utf-8")

    runs = load_sam3_predictions(result_path, manifest=test_manifest, options=test_manifest.normalization)
    raw_run = next(r for r in runs if r.view == "raw")
    assert len(raw_run.boxes) == 1
    assert raw_run.boxes[0].mask is not None
    assert raw_run.boxes[0].mask.width == 100
    assert raw_run.boxes[0].mask.height == 100


def test_corrupt_mask_rle_rejected(tmp_path: Path, test_manifest: e.BenchmarkManifest) -> None:
    result_doc = {
        "schema": "autolabel-sam3-benchmark/v1",
        "completed": True,
        "configurations": [
            {
                "id": "cfg-1",
                "status": "completed",
                "predictions": [
                    {
                        "frame_index": 0,
                        "track_id": 1,
                        "label": "car",
                        "bbox": [10.0, 10.0, 20.0, 20.0],
                        "mask_rle": {"width": 100, "height": 100, "counts": [50]},  # sum != 10000
                    }
                ],
            }
        ],
    }
    result_path = tmp_path / "bad_mask.json"
    result_path.write_text(json.dumps(result_doc), encoding="utf-8")

    with pytest.raises(ValueError, match="corrupt MaskRLE"):
        load_sam3_predictions(result_path, manifest=test_manifest, options=test_manifest.normalization)


def test_endpoint_mismatch_with_payload_zip(tmp_path: Path, test_manifest: e.BenchmarkManifest) -> None:
    # Payload zip has frames 0, 1, 2, 339 (where 339 is not in bindings)
    payload_path = tmp_path / "payload.zip"
    with zipfile.ZipFile(payload_path, "w") as archive:
        archive.writestr(
            "frames.json",
            json.dumps({"frames": [{"original_source_index": 0}, {"original_source_index": 339}]}),
        )
        archive.writestr("benchmark-config.json", json.dumps({"configurations": []}))

    result_doc = {
        "schema": "autolabel-sam3-benchmark/v1",
        "completed": True,
        "configurations": [{"id": "cfg-1", "status": "completed", "predictions": []}],
    }
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result_doc), encoding="utf-8")

    with pytest.raises(ValueError, match="payload frames contain unrepresented endpoint"):
        load_sam3_predictions(
            result_path, manifest=test_manifest, payload=payload_path, options=test_manifest.normalization
        )


def test_stitch_comparison_schema(tmp_path: Path, test_manifest: e.BenchmarkManifest) -> None:
    doc = {
        "schema": "autolabel-stitch-comparison/v1",
        "config_id": "cfg-stitch",
        "tracks": [
            {
                "track_id": "global-1",
                "label": "car",
                "observations": [
                    {"frame": 0, "bbox": [10.0, 10.0, 50.0, 50.0], "score": 0.8},
                    {"frame": 1, "bbox": [12.0, 12.0, 52.0, 52.0], "score": 0.82},
                ],
            }
        ],
    }
    result_path = tmp_path / "stitch.json"
    result_path.write_text(json.dumps(doc), encoding="utf-8")

    runs = load_sam3_predictions(result_path, manifest=test_manifest, options=test_manifest.normalization)
    assert len(runs) == 2
    assert all(r.status == "completed" for r in runs)
    assert len(runs[0].boxes) == 2


def test_track_set_schema(tmp_path: Path, test_manifest: e.BenchmarkManifest) -> None:
    doc = {
        "schema": "autolabel-track-set/v1",
        "job_id": "00000000-0000-0000-0000-000000000000",
        "manifest_sha256": HASH,
        "seed_sha256": HASH,
        "tracks": [
            {
                "track_id": "track-1",
                "label_name": "car",
                "observations": [
                    {"frame_index": 0, "bbox": [10.0, 10.0, 50.0, 50.0]},
                ],
            }
        ],
    }
    result_path = tmp_path / "track_set.json"
    result_path.write_text(json.dumps(doc), encoding="utf-8")

    runs = load_sam3_predictions(result_path, manifest=test_manifest, options=test_manifest.normalization)
    assert len(runs) == 2
    raw_run = next(r for r in runs if r.view == "raw")
    assert len(raw_run.boxes) == 1
    assert raw_run.boxes[0].score.semantics == "fixed_visible"
    assert raw_run.boxes[0].score.value == 1.0


def test_interpolation_view_produced(tmp_path: Path, test_manifest: e.BenchmarkManifest) -> None:
    options = e.PredictionNormalizationOptions(
        views=("raw", "stitched", "interpolated"),
        interpolation=e.InterpolationPolicy(algorithm="video_optimization_linear", revision="opt-linear/v1"),
    )
    result_doc = {
        "schema": "autolabel-sam3-benchmark/v1",
        "completed": True,
        "configurations": [
            {
                "id": "cfg-1",
                "status": "completed",
                "config": {"mode": "text_prompt"},
                "selected_source_frames": [0, 2],
                "evaluated_source_frames": [0, 2],
                "predictions": [
                    {"frame_index": 0, "track_id": 1, "label": "car", "bbox": [10.0, 10.0, 20.0, 20.0], "score": 0.9},
                    {"frame_index": 2, "track_id": 1, "label": "car", "bbox": [30.0, 30.0, 40.0, 40.0], "score": 0.9},
                ],
            }
        ],
    }
    result_path = tmp_path / "interp_result.json"
    result_path.write_text(json.dumps(result_doc), encoding="utf-8")

    runs = load_sam3_predictions(result_path, manifest=test_manifest, options=options)
    assert len(runs) == 3
    interp_run = next(r for r in runs if r.view == "interpolated")
    # Frame 1 should be interpolated
    interpolated_box = next((b for b in interp_run.boxes if b.origin == "interpolated"), None)
    assert interpolated_box is not None
    assert interpolated_box.frame.frame_uid == "1"
    assert interpolated_box.score.semantics == "missing"
