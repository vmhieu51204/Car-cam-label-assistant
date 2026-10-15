from __future__ import annotations

import json
import zipfile

import pytest
from PIL import Image

from kaggle.sam3_benchmark.main import load_payload
from scripts.package_sam3_sequence import package_sequence


def test_unlabeled_sequence_package_runs_text_only_without_ground_truth(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    for frame in range(6):
        Image.new("RGB", (32, 24), (frame * 10, 0, 0)).save(images / f"{frame:06d}.png")
    archive_path = tmp_path / "payload.zip"
    report = package_sequence(images=images, output=archive_path, source_fps=10, target_fps=5, classes=("car",))
    assert report["source_indices"] == [0, 2, 4, 5]
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extracted)
    config, frames, optical, seeds = load_payload(extracted)
    assert config["configurations"][0]["mode"] == "text_prompt"
    assert seeds == []
    assert optical == [0, 1, 2, 3]
    assert [item["original_source_index"] for item in frames] == [0, 2, 4, 5]
    assert json.loads((extracted / "seed.json").read_text()) == {"seeds": []}


def test_unlabeled_package_rejects_oversize_before_output(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    for frame in range(2):
        Image.new("RGB", (32, 24), "red").save(images / f"{frame:06d}.png")
    output = tmp_path / "payload.zip"
    with pytest.raises(ValueError, match="max_bytes"):
        package_sequence(images=images, output=output, source_fps=5, max_bytes=1)
    assert not output.exists()
