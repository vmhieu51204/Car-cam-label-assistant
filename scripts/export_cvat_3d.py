"""Build a pending CVAT 3D cuboid-track draft without contacting CVAT.

The fork serializes cuboids as position XYZ, Euler XYZ, scale XYZ, then seven
reserved values. LiDAR↔CVAT visual axis mapping still requires a real task check.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _points(box: dict) -> list[float]:
    center = list(map(float, box["center_lidar_xyz_m"]))
    size = list(map(float, box["size_lwh_m"]))
    yaw = float(box["yaw_lidar_rad"])
    if (
        len(center) != 3
        or len(size) != 3
        or not all(math.isfinite(item) for item in (*center, *size, yaw))
        or min(size) <= 0
    ):
        raise ValueError("invalid LiDAR cuboid")
    return center + [0.0, 0.0, yaw] + size + [0.0] * 7


def build_cvat_3d_draft(draft: dict, labels: list[dict], *, source_frame_count: int) -> dict:
    """Resolve CVAT label/attribute IDs and preserve outside markers."""
    if draft.get("schema") != "autolabel-lidar-3d-draft/v1" or source_frame_count < 1:
        raise ValueError("expected a LiDAR 3D draft and positive source frame count")
    by_name = {label["name"]: label for label in labels}
    if len(by_name) != len(labels):
        raise ValueError("CVAT labels have duplicate names")
    converted = []
    for track in draft["tracks"]:
        label = by_name.get(track["label"])
        if label is None:
            raise ValueError(f"CVAT 3D task lacks label {track['label']}")
        specs = {item["name"]: item for item in label.get("attributes", [])}
        if not {"autolabel_track_id", "review_status"} <= specs.keys():
            raise ValueError("CVAT 3D label needs autolabel_track_id and review_status attributes")
        if specs["review_status"].get("mutable") is not True:
            raise ValueError("CVAT 3D review_status must be mutable for human review")
        observations = sorted(track["observations"], key=lambda row: row["source_frame_index"])
        if len({row["source_frame_index"] for row in observations}) != len(observations):
            raise ValueError("duplicate source frame in 3D track")
        shapes = []
        previously_inside = False
        for observation in observations:
            frame = int(observation["source_frame_index"])
            if not 0 <= frame < source_frame_count:
                raise ValueError("3D track frame outside CVAT task")
            box = observation["box_lidar"]
            if box is None and not previously_inside:
                continue
            shape = {
                "type": "cuboid",
                "frame": frame,
                "points": _points(box) if box is not None else shapes[-1]["points"],
                "outside": box is None,
                "occluded": False,
                "keyframe": True,
                "z_order": 0,
                "rotation": 0.0,
                "source": "auto",
                "attributes": [{"spec_id": int(specs["review_status"]["id"]), "value": "pending"}],
            }
            shapes.append(shape)
            previously_inside = box is not None
        if not shapes:
            continue
        converted.append(
            {
                "frame": shapes[0]["frame"],
                "label_id": int(label["id"]),
                "group": 0,
                "source": "auto",
                "attributes": [{"spec_id": int(specs["autolabel_track_id"]["id"]), "value": track["track_id"]}],
                "shapes": shapes,
            }
        )
    return {
        "schema": "autolabel-cvat-3d-draft/v1",
        "dimension": "3d",
        "coordinate_frame": draft["coordinate_frame"],
        "source_frame_count": source_frame_count,
        "review_status": "pending",
        "mapping_visually_verified": False,
        "tracks": converted,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--cvat-labels", type=Path, required=True, help="Read-only GET /api/jobs/<id>/labels snapshot")
    parser.add_argument("--source-frame-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw_labels = json.loads(args.cvat_labels.read_text(encoding="utf-8"))
    labels = raw_labels.get("labels") if isinstance(raw_labels, dict) else raw_labels
    result = build_cvat_3d_draft(
        json.loads(args.draft.read_text(encoding="utf-8")), labels, source_frame_count=args.source_frame_count
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "tracks": len(result["tracks"])}))


if __name__ == "__main__":
    main()
