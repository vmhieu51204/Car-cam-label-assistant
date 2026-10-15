import pytest

from src.services.vm.regions import expand_tracks


def test_moving_privacy_track_uses_timestamps_and_no_extrapolation():
    track = {"track_id": "face", "keyframes": [{"frame": 1, "box": [0, 0, 4, 4]}, {"frame": 3, "box": [8, 0, 12, 4]}]}
    regions = expand_tracks([track], [0, 0.1, 0.3, 0.5, 0.6], 64, 64)
    assert [r["start"] for r in regions] == [1, 2, 3]
    assert regions[1]["box"] == pytest.approx([4, 0, 8, 4])
    assert all(r["track_id"] == "face" for r in regions)
    track["keyframes"].reverse()
    with pytest.raises(ValueError, match="ordered"):
        expand_tracks([track], [0, 0.1, 0.3, 0.5, 0.6], 64, 64)
