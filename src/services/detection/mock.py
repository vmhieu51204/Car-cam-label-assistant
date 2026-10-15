"""GT-derived simulation only; never report these outputs as model evidence."""

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np

from src.config import Settings
from src.models.perception import Box3D, Detection, Frame
from src.services.geometry.boxes import iou3d

LABELS = ("car", "truck", "pedestrian")
AnnotationRow = tuple[int, dict, str]


def category_label(category: str) -> str | None:
    """Map the supported nuScenes categories used by the prototype."""
    if category in {"vehicle.car", "vehicle.truck"}:
        return category.split(".")[1]
    return "pedestrian" if category.startswith("human.pedestrian.") else None


@dataclass
class MockResult:
    frames: list[Frame]
    stats: dict
    audit: list[dict]


class SceneReader:
    """Read metadata without importing GPU or nuScenes visualization dependencies."""

    def __init__(self, settings: Settings) -> None:
        self.root = settings.data_root / settings.data_version

    def table(self, name: str) -> list[dict]:
        """Load one nuScenes metadata table."""
        return json.loads((self.root / f"{name}.json").read_text(encoding="utf-8"))

    def samples(self, scene_name: str, max_frames: int | None) -> list[dict]:
        """Walk the scene's linked sample chain, rejecting malformed cycles."""
        scenes = {row["name"]: row for row in self.table("scene")}
        if scene_name not in scenes:
            raise ValueError(f"Unknown scene: {scene_name}")
        samples = {row["token"]: row for row in self.table("sample")}
        token = scenes[scene_name]["first_sample_token"]
        selected: list[dict] = []
        seen: set[str] = set()
        while token and (max_frames is None or len(selected) < max_frames):
            if token in seen:
                raise ValueError("Cycle in sample chain")
            seen.add(token)
            sample = samples[token]
            selected.append(sample)
            token = sample["next"]
        return selected

    def annotations(self, samples: list[dict]) -> list[AnnotationRow]:
        """Return supported annotations ordered exactly as the previous backend."""
        instances = {row["token"]: row for row in self.table("instance")}
        categories = {row["token"]: row["name"] for row in self.table("category")}
        annotations: dict[str, list[tuple[dict, str]]] = {}
        for row in self.table("sample_annotation"):
            label = category_label(categories[instances[row["instance_token"]]["category_token"]])
            if label:
                annotations.setdefault(row["sample_token"], []).append((row, label))
        return [
            (index, ann, label)
            for index, sample in enumerate(samples)
            for ann, label in sorted(annotations.get(sample["token"], []), key=lambda pair: pair[0]["token"])
        ]


def quota(count: int, rate: float) -> int:
    """Use round-half-up error quotas rather than stochastic percentages."""
    return int(math.floor(count * rate + 0.5))


class MockGenerator:
    """Keep random state local so seeds reproduce predictions across entry points."""

    def __init__(self, settings: Settings, scene_name: str) -> None:
        self.settings, self.scene_name = settings, scene_name
        seed = int.from_bytes(hashlib.sha256(scene_name.encode()).digest()[:4], "little")
        self.rng = np.random.default_rng([settings.mock_seed, seed])

    def generate(self, max_frames: int | None) -> MockResult:
        """Apply misses, class errors, geometric noise and clearly marked false boxes."""
        reader = SceneReader(self.settings)
        samples = reader.samples(self.scene_name, max_frames)
        rows = reader.annotations(samples)
        missed, changed = self._selection(len(rows))
        frames = self._frames(samples)
        audit = self._retained(frames, rows, (missed, changed))
        spurious = self._false_boxes(frames, rows, audit)
        stats = {
            "eligible_gt": len(rows),
            "dropped_boxes": len(missed),
            "spurious_boxes": spurious,
            "class_changed_boxes": len(changed),
            "predicted_boxes": sum(len(frame.detections) for frame in frames),
            "supported_classes": list(LABELS),
        }
        return MockResult(frames, stats, audit)

    def _selection(self, count: int) -> tuple[set[int], set[int]]:
        missed = set(self.rng.choice(count, quota(count, self.settings.mock_miss_rate), replace=False).tolist())
        retained = [index for index in range(count) if index not in missed]
        changed = set(
            self.rng.choice(retained, quota(len(retained), self.settings.mock_class_error_rate), replace=False).tolist()
        )
        return missed, changed

    def _frames(self, samples: list[dict]) -> list[Frame]:
        return [
            Frame(
                scene=self.scene_name,
                sample_token=sample["token"],
                frame_index=index,
                timestamp_s=sample["timestamp"] / 1_000_000,
                detections=[],
            )
            for index, sample in enumerate(samples)
        ]

    def _retained(
        self, frames: list[Frame], rows: list[AnnotationRow], selection: tuple[set[int], set[int]]
    ) -> list[dict]:
        missed, changed = selection
        audit = []
        for row_index, (index, annotation, label) in enumerate(rows):
            if row_index in missed:
                continue
            detection, overlap = self._detection(annotation, label, row_index in changed)
            audit.append(
                {
                    "frame_index": index,
                    "detection_index": len(frames[index].detections),
                    "spurious": False,
                    "class_changed": row_index in changed,
                    "oracle_iou": overlap,
                }
            )
            frames[index].detections.append(detection)
        return audit

    def _detection(self, annotation: dict, label: str, wrong_class: bool) -> tuple[Detection, float]:
        w, x, y, z = annotation["rotation"]
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        width, length, height = annotation["size"]
        gt_box = Box3D(center=tuple(annotation["translation"]), size_lwh=(length, width, height), yaw=yaw)
        center = np.asarray(gt_box.center) + self.rng.normal(0, self.settings.mock_position_std, 3)
        size = np.asarray(gt_box.size_lwh) * self.rng.uniform(0.95, 1.05, 3)
        predicted_box = Box3D(
            center=tuple(center), size_lwh=tuple(size), yaw=yaw + self.rng.normal(0, self.settings.mock_yaw_std)
        )
        predicted_label = str(self.rng.choice([name for name in LABELS if name != label])) if wrong_class else label
        overlap = iou3d(gt_box, predicted_box)
        quality = overlap * (0.25 if wrong_class else 1.0)
        score = float(np.clip(0.10 + 0.85 * quality + self.rng.normal(0, 0.06), 0.01, 0.99))
        return Detection(label=predicted_label, box=predicted_box, confidence=score), overlap

    def _false_boxes(self, frames: list[Frame], rows: list[AnnotationRow], audit: list[dict]) -> int:
        count = quota(len(rows), self.settings.mock_false_rate)
        # Easy negatives are deliberate and must never support real quality claims.
        far_x = max((ann["translation"][0] + max(ann["size"]) for _, ann, _ in rows), default=0) + 30
        for index in range(count):
            frame_index = int(self.rng.integers(len(frames)))
            false_box = Box3D(
                center=(far_x + 6 * index, float(self.rng.uniform(-20, 20)), 1),
                size_lwh=(4, 2, 2),
                yaw=float(self.rng.uniform(-math.pi, math.pi)),
            )
            audit.append(
                {
                    "frame_index": frame_index,
                    "detection_index": len(frames[frame_index].detections),
                    "spurious": True,
                    "class_changed": False,
                    "oracle_iou": 0.0,
                }
            )
            frames[frame_index].detections.append(
                Detection(
                    label=str(self.rng.choice(LABELS)), box=false_box, confidence=float(self.rng.uniform(0.05, 0.35))
                )
            )
        return count


def generate_mock(settings: Settings, scene_name: str, max_frames: int | None = None) -> MockResult:
    """Generate mock global upright boxes; oracle audit stays outside prediction schema."""
    return MockGenerator(settings, scene_name).generate(max_frames)


def mock_frames(settings: Settings, scene_name: str, max_frames: int) -> list[Frame]:
    """Compatibility entry point for scene-limited mock requests."""
    return generate_mock(settings, scene_name, max_frames).frames
