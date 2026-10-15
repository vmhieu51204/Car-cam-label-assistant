"""Precompute tracked predictions from explicit mock or real detector inputs."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Settings
from src.models.perception import Prediction, Track
from src.services.detection.centerpoint import CenterPointResult, load_centerpoint_output
from src.services.detection.mock import MockResult, generate_mock
from src.services.tracking import Tracker

JsonObject = dict[str, object]


def serialize_predictions(tracks: list[Track]) -> list[JsonObject]:
    """Serialize tracks through the public prediction contract."""
    rows = []
    for track in tracks:
        for item in track.observations:
            row = {
                "frame_token": item.sample_token,
                "frame_index": item.frame_index,
                "box3d": {
                    "center": item.box.center,
                    "size": item.box.size_lwh,
                    "size_order": "lwh",
                    "yaw": item.box.yaw,
                    "coordinate_frame": item.box.coordinate_frame,
                },
                "class": track.label,
                "score": item.confidence,
                "track_id": track.track_id,
                "source": "auto",
                "data_source": track.data_source,
                "review_status": "pending",
            }
            rows.append(Prediction.model_validate(row).model_dump(by_alias=True))
    return sorted(rows, key=lambda row: (int(str(row["frame_index"])), str(row["track_id"])))


class MockPrecompute:
    """Serialize one mock scene while keeping oracle diagnostics separate."""

    def __init__(self, settings: Settings, scene: str, output: Path) -> None:
        self.settings = settings
        self.scene = scene
        self.output = output

    def run(self) -> JsonObject:
        """Generate, track, serialize, and return explicit mock provenance."""
        result = generate_mock(self.settings, self.scene)
        tracker = Tracker(self.settings.match_giou, self.settings.max_missed)
        for frame in result.frames:
            tracker.step(frame)
        predictions = serialize_predictions(tracker.tracks)
        metadata = self._metadata(result, len(tracker.tracks))
        self._write(metadata, predictions, result.audit)
        return metadata

    def _metadata(self, result: MockResult, track_count: int) -> JsonObject:
        return {
            "data_source": "mock",
            "dataset": self.settings.data_version,
            "scene": self.scene,
            "seed": self.settings.mock_seed,
            "user_verification": "unverified",
            "mock_miss_rate": self.settings.mock_miss_rate,
            "mock_false_rate": self.settings.mock_false_rate,
            "mock_class_error_rate": self.settings.mock_class_error_rate,
            "frames": len(result.frames),
            "tracks": track_count,
            **result.stats,
            "warning": "GT-derived simulation. Not detector or AL quality evidence.",
        }

    def _write(self, metadata: JsonObject, predictions: list[JsonObject], audit: list[dict]) -> None:
        self.output.mkdir(parents=True, exist_ok=True)
        payload = {"metadata": metadata, "predictions": predictions}
        (self.output / f"{self.scene}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        audit_dir = self.output / "mock-audit"
        audit_dir.mkdir(exist_ok=True)
        audit_payload = {"metadata": metadata, "diagnostics": audit}
        (audit_dir / f"{self.scene}.json").write_text(json.dumps(audit_payload, indent=2), encoding="utf-8")


class ModelPrecompute:
    """Transform and track an isolated real CenterPoint output."""

    def __init__(self, settings: Settings, scene: str, model_output: Path, output: Path) -> None:
        self.settings = settings
        self.scene = scene
        self.model_output = model_output
        self.output = output

    def run(self) -> JsonObject:
        """Validate real provenance, convert LiDAR boxes, track, and serialize."""
        result = load_centerpoint_output(self.settings, self.model_output, self.scene)
        tracker = Tracker(self.settings.match_giou, self.settings.max_missed, data_source="real")
        for frame in result.frames:
            tracker.step(frame)
        metadata = self._metadata(result, len(tracker.tracks))
        payload = {"metadata": metadata, "predictions": serialize_predictions(tracker.tracks)}
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / f"{self.scene}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return metadata

    def _metadata(self, result: CenterPointResult, track_count: int) -> JsonObject:
        return {
            "data_source": "real",
            "dataset": self.settings.data_version,
            "scene": self.scene,
            "detector": result.model,
            "coordinate_frame": "nuscenes_global",
            "user_verification": "unverified",
            "tracks": track_count,
            **result.stats,
        }


def precompute(settings: Settings, scene: str, output: Path) -> JsonObject:
    """Precompute one mock scene without exposing oracle-only fields."""
    return MockPrecompute(settings, scene, output).run()


def precompute_model(settings: Settings, scene: str, model_output: Path, output: Path) -> JsonObject:
    """Precompute one real CenterPoint scene from a versioned interchange file."""
    return ModelPrecompute(settings, scene, model_output, output).run()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mock", "model"], default="mock")
    parser.add_argument("--scene", help="Default: every scene in the configured dataset")
    parser.add_argument("--output", type=Path, default=Path("predictions"))
    parser.add_argument("--model-output", type=Path, help="Raw CenterPoint JSON produced by the GPU worker")
    args = parser.parse_args()
    if args.mode == "model" and (not args.scene or args.model_output is None):
        parser.error("--mode model requires --scene and --model-output; no mock fallback is allowed")
    return args


def _scene_names(settings: Settings, selected: str | None) -> list[str]:
    scenes = json.loads((settings.data_root / settings.data_version / "scene.json").read_text(encoding="utf-8"))
    return [selected] if selected else sorted(str(scene["name"]) for scene in scenes)


def _print_summary(name: str, metadata: JsonObject) -> None:
    print(
        f"[{metadata['data_source']}] {name}: frames={metadata['frames']} "
        f"boxes={metadata['predicted_boxes']} tracks={metadata['tracks']}",
        flush=True,
    )


def main() -> None:
    """Run deterministic mock precompute for one or all configured scenes."""
    args = _arguments()
    settings = Settings()
    summary = []
    if args.mode == "model":
        metadata = precompute_model(settings, args.scene, args.model_output, args.output)
        summary.append(metadata)
        _print_summary(args.scene, metadata)
    else:
        for name in _scene_names(settings, args.scene):
            metadata = precompute(settings, name, args.output)
            summary.append(metadata)
            _print_summary(name, metadata)
    reports = Path("reports")
    reports.mkdir(exist_ok=True)
    (reports / f"precompute-{args.mode}-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
