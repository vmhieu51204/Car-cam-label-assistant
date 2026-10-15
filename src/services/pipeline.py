"""Run the local mock detector and tracker with explicit provenance."""

from time import perf_counter

from src.config import Settings
from src.models.perception import DetectRequest, Track
from src.services.detection.mock import generate_mock
from src.services.tracking import Tracker


def detect_tracks(settings: Settings, request: DetectRequest) -> tuple[dict, list[Track]]:
    """Produce reproducible GT-derived predictions, never detector quality evidence."""
    started = perf_counter()
    result = generate_mock(settings, request.scene, request.max_frames)
    tracker = Tracker(settings.match_giou, settings.max_missed)
    for frame in result.frames:
        tracker.step(frame)
    metadata = {
        "data_source": "mock",
        "dataset": settings.data_version,
        "scene": request.scene,
        "sample_tokens": [frame.sample_token for frame in result.frames],
        "seed": settings.mock_seed,
        "position_std": settings.mock_position_std,
        "miss_rate": settings.mock_miss_rate,
        "false_rate": settings.mock_false_rate,
        "class_error_rate": settings.mock_class_error_rate,
        "yaw_std": settings.mock_yaw_std,
        "match_giou": settings.match_giou,
        "max_missed": settings.max_missed,
        "frames": len(result.frames),
        "duration_ms": round((perf_counter() - started) * 1000, 3),
        "user_verification": "unverified",
        "coordinate_frame": "nuscenes_global",
        "limitations": "GT + noise; upright yaw approximation; no quality or AL evidence",
        **result.stats,
    }
    return metadata, tracker.tracks
