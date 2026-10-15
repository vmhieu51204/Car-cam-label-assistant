"""KITTI preflight must reject missing or misaligned multimodal inputs."""

import numpy as np
import pytest
from PIL import Image

from src.services.geometry.kitti_tracking import load_sequence, missing_paths


def _fixture(root, *, directory_oxts=False):
    base = root / "training"
    for folder in ("image_02/0013", "velodyne/0013", "calib"):
        (base / folder).mkdir(parents=True)
    for index in range(2):
        Image.new("RGB", (16, 12), "gray").save(base / "image_02/0013" / f"{index:06d}.png")
        np.array([[1, 2, 3, 0.5]], dtype="<f4").tofile(base / "velodyne/0013" / f"{index:06d}.bin")
    (base / "calib/0013.txt").write_text(
        "P2: " + " ".join(map(str, range(12))) + "\n"
        "R_rect: " + " ".join(map(str, range(9))) + "\n"
        "Tr_velo_cam: " + " ".join(map(str, range(12))) + "\n"
    )
    packet = " ".join(["0"] * 30)
    if directory_oxts:
        folder = base / "oxts/0013"
        folder.mkdir(parents=True)
        for index in range(2):
            (folder / f"{index:06d}.txt").write_text(packet + "\n")
    else:
        (base / "oxts").mkdir()
        (base / "oxts/0013.txt").write_text(packet + "\n" + packet + "\n")


@pytest.mark.parametrize("directory_oxts", [False, True])
def test_loads_aligned_native_style_sequence_with_declared_timebase(tmp_path, directory_oxts):
    _fixture(tmp_path, directory_oxts=directory_oxts)
    sequence = load_sequence(tmp_path, "0013", source_fps=10, fps_source="operator-fallback")
    assert len(sequence.frames) == 2
    assert sequence.frames[1].timestamp_s == 0.1
    assert sequence.frames[0].lidar.name == "000000.bin"
    assert sequence.calibration["P2"].shape == (3, 4)
    assert sequence.fps_source == "operator-fallback"


def test_reports_exact_missing_modalities_before_inference(tmp_path):
    _fixture(tmp_path)
    (tmp_path / "training/calib/0013.txt").unlink()
    (tmp_path / "training/oxts/0013.txt").unlink()
    missing = missing_paths(tmp_path, "0013")
    assert "training\\calib\\0013.txt" in missing[0] or "training/calib/0013.txt" in missing[0]
    assert "oxts" in missing[1]
    with pytest.raises(FileNotFoundError, match="KITTI sequence is incomplete"):
        load_sequence(tmp_path, "0013", source_fps=10, fps_source="fallback")


def test_rejects_misaligned_lidar_and_pose(tmp_path):
    _fixture(tmp_path)
    (tmp_path / "training/velodyne/0013/000001.bin").unlink()
    with pytest.raises(ValueError, match="frame counts differ"):
        load_sequence(tmp_path, "0013", source_fps=10, fps_source="fallback")
    np.array([[1, 2, 3, 0.5]], dtype="<f4").tofile(tmp_path / "training/velodyne/0013/000001.bin")
    (tmp_path / "training/oxts/0013.txt").write_text("0 0 0\n")
    with pytest.raises(ValueError, match="frame counts differ"):
        load_sequence(tmp_path, "0013", source_fps=10, fps_source="fallback")
