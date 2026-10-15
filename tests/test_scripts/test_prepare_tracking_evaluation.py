import csv
import hashlib
import json
import zipfile

import pytest
from PIL import Image

from kaggle.sam3_benchmark.main import load_payload
from scripts.prepare_tracking_evaluation import inventory_sequence, prepare, split_sequences
from src.services.evaluation.tracking_payload import materialize, split_windows


def dataset(tmp_path, *, sequences=("0001",), frames=3):
    training = tmp_path / "dataset/training"
    (training / "label_02").mkdir(parents=True)
    for name in sequences:
        directory = training / "image_02" / name
        directory.mkdir(parents=True)
        for i in range(frames):
            Image.new("RGB", (24, 20), (i * 20 % 256, 30, 40)).save(directory / f"{i:06d}.png")
        (training / "label_02" / f"{name}.txt").write_text(
            "0 0 Car 0 0 0 0 0 10 10 1 1 1 0 0 10 0\n"
            "2 0 Car 1 2 0 0 0 10 10 1 1 1 0 0 10 0\n"
            "1 1 Pedestrian 0 1 0 0 0 5 15 1 1 1 0 0 10 0\n"
            "1 -1 DontCare -1 -1 -10 0 0 5 5 -1 -1 -1 -1000 -1000 -1000 -10\n"
        )
    return training


def test_inventory_hashes_distributions_and_source_gt_gaps(tmp_path):
    training = dataset(tmp_path)
    row = inventory_sequence(training, "0001")
    assert row["classes"]["Car"]["interrupted_gt_tracks"] == 1
    assert row["classes"]["Car"]["occlusion"] == {"none": 1, "heavy": 1}
    assert row["dontcare_object_frames"] == 1
    Image.new("RGB", (24, 20), "red").save(training / "image_02/0001/000000.png")
    changed = inventory_sequence(training, "0001")
    assert changed["image_names_sha256"] == row["image_names_sha256"]
    assert changed["images_content_sha256"] != row["images_content_sha256"]


def test_prepare_csv_splits_priority_and_input_protection(tmp_path):
    names = [f"{i:04d}" for i in range(21)]
    training = dataset(tmp_path, sequences=names)
    path = tmp_path / "protocol.json"
    p = prepare(training.parent, tmp_path / "out", path)
    assert p["splits"] == split_sequences(list(reversed(names)))
    assert p["held_out_history_status"] == "unverified"
    assert all(g["sequence"] in p["splits"]["dev"] for g in p["gpu_plan"])
    assert all(r["scene_diversity"].startswith("unavailable") for r in p["priorities"])
    with (tmp_path / "out/inventory.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 42 and json.loads(rows[0]["truncation_native"])
    assert "occlusion_heavy" in rows[0] and "height_lt25" in rows[0]
    before = path.read_bytes()
    Image.new("RGB", (24, 20), "red").save(training / "image_02/0001/000000.png")
    with pytest.raises(ValueError, match="hash mismatch"):
        prepare(training.parent, tmp_path / "out", path)
    assert path.read_bytes() == before


def test_materialized_all_frames_obeys_notebook_contract_and_has_no_gt(tmp_path):
    training = dataset(tmp_path)
    inventory = inventory_sequence(training, "0001")
    out = tmp_path / "prepared"
    kernels = materialize(training, inventory, out)
    assert len(kernels) == 1 and kernels[0]["source_frames"] == [0, 1, 2]
    archive = out / kernels[0]["archive_name"]
    before = archive.read_bytes()
    assert materialize(training, inventory, out) == kernels
    assert archive.read_bytes() == before
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    with zipfile.ZipFile(archive) as stream:
        assert set(stream.namelist()) == {
            "frames.json",
            "benchmark-config.json",
            "seed.json",
            *(f"frames/frame_{i:06d}.jpg" for i in range(3)),
        }
        stream.extractall(extracted)
    config, frames, optical, seeds = load_payload(extracted)
    assert config["configurations"][0]["sampling_strategy"] == "all_frames" and seeds == []
    assert [f["original_source_index"] for f in frames] == optical == [0, 1, 2]
    assert len(kernels[0]["source_hashes"]) == 3
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == kernels[0]["archive_sha256"]
    archive.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="archive hash mismatch"):
        materialize(training, inventory, out)


def test_payload_source_and_output_protection(tmp_path):
    training = dataset(tmp_path)
    row = inventory_sequence(training, "0001")
    out = tmp_path / "foreign"
    out.mkdir()
    (out / "keep.txt").write_text("keep")
    with pytest.raises(ValueError, match="existing payload output"):
        materialize(training, row, out)
    assert (out / "keep.txt").read_text() == "keep"
    row["images"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source image hash mismatch"):
        materialize(training, row, tmp_path / "new")
    assert not (tmp_path / "new").exists()


def test_window_aligned_splitting_preserves_overlap_and_bounds():
    spans = split_windows([100] * 120, max_bytes=9600)
    assert spans == [(0, 95), (90, 120)]
    assert set(range(120)) == set(i for a, b in spans for i in range(a, b))
    assert spans[0][1] - spans[1][0] == 5
    timed = split_windows([1] * 120, session_seconds=900)
    assert all((b - a) * 598.39 / 100 * 1.25 + 180 <= 900 for a, b in timed)
    with pytest.raises(ValueError, match="cannot drop frames"):
        split_windows([100] * 50, max_bytes=4999)
    with pytest.raises(ValueError, match="budget"):
        split_windows([1] * 50, session_seconds=21601)


def test_interrupted_archive_publication_recovers_without_overwriting(tmp_path, monkeypatch):
    from pathlib import Path

    training = dataset(tmp_path)
    row = inventory_sequence(training, "0001")
    output = tmp_path / "prepared"
    replace = Path.replace

    def interrupted(path, target):
        if path.name == "manifest.json":
            raise OSError("simulated publication interruption")
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", interrupted)
    with pytest.raises(OSError, match="interruption"):
        materialize(training, row, output)
    archive = (output / "kernel-000.zip").read_bytes()
    assert (output / "preparation.json").exists() and not (output / "manifest.json").exists()
    monkeypatch.setattr(Path, "replace", replace)
    assert materialize(training, row, output)[0]["frames"] == 3
    assert (output / "kernel-000.zip").read_bytes() == archive
