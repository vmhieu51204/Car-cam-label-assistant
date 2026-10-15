"""Review saved 3D objects without changing geometry or racing editor saves."""

import hashlib
from copy import deepcopy

from src.services.vm.storage import encode


def revision(annotations: dict) -> int:
    content = {name: annotations.get(name, []) for name in ("tags", "shapes", "tracks", "intervals")}
    return int(hashlib.sha256(encode(content)).hexdigest()[:13], 16)


def review_objects(annotations: dict, labels: list[dict]) -> list[dict]:
    by_id = {label["id"]: label for label in labels}
    result = []
    for kind in ("shapes", "tracks"):
        for obj in annotations.get(kind, []):
            label = by_id[obj["label_id"]]
            attrs = {item["id"]: item["name"] for item in label["attributes"]}
            states = []
            inherited = {item["spec_id"]: item["value"] for item in obj.get("attributes", [])}
            identity = next((value for key, value in inherited.items() if attrs[key] == "autolabel_3d_track_id"), "")
            for shape in sorted(obj.get("shapes", [obj]), key=lambda shape: shape["frame"]):
                inherited.update({item["spec_id"]: item["value"] for item in shape.get("attributes", [])})
                if not shape.get("outside"):
                    states.append(
                        next((value for key, value in inherited.items() if attrs[key] == "review_status"), "pending")
                    )
            state = states[0] if states and len(set(states)) == 1 else "pending"
            result.append(
                {
                    "object_id": f"{kind}:{obj['id']}",
                    "label": label["name"],
                    "frame": obj["frame"],
                    "track_id": identity,
                    "review_status": state,
                    "observations": len(states),
                }
            )
    return result


def reviewed_annotations(annotations: dict, labels: list[dict], object_id: str, status: str) -> dict:
    local = deepcopy(annotations)
    kind, identity = object_id.split(":")
    obj = next((row for row in local[kind] if row["id"] == int(identity)), None)
    if obj is None:
        raise ValueError("Saved object no longer exists")
    label = next(label for label in labels if label["id"] == obj["label_id"])
    spec = next((item["id"] for item in label["attributes"] if item["name"] == "review_status"), None)
    if spec is None:
        raise ValueError("This label has no review_status attribute")
    for shape in obj.get("shapes", [obj]):
        shape["attributes"] = [item for item in shape.get("attributes", []) if item["spec_id"] != spec] + [
            {"spec_id": spec, "value": status}
        ]
    return local
