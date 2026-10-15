"""CPU-only checkpoint integrity contract shared by backend and GPU notebook."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any


def checkpoint_identity(frames: list[dict[str, Any]], row: dict[str, Any], code_sha256: str = "") -> str:
    return hashlib.sha256(json.dumps([frames, row, code_sha256], sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def model_identity(probe: dict[str, Any]) -> dict[str, Any]:
    identity = {"sam3_commit": probe.get("environment", {}).get("versions", {}).get("sam3_commit"),
                "checkpoint_sha256": probe.get("checkpoint_sha256")}
    if probe.get("model_variant") not in {None, "sam3"}:
        identity["model_variant"] = probe["model_variant"]
    return identity


def validate_checkpoint(entry: dict[str, Any], *, identity: str, window_id: int,
                        frame_ids: list[int], model: dict[str, Any] | None = None) -> None:
    body = {key: value for key, value in entry.items() if key != "output_sha256"}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if (entry.get("schema") != "autolabel-sam3-window/v1" or entry.get("input_sha256") != identity
            or entry.get("window_id") != window_id or entry.get("frame_ids") != frame_ids
            or entry.get("output_sha256") != digest or (model is not None and entry.get("model") != model)):
        raise ValueError("window checkpoint provenance or hash mismatch")
    if not isinstance(entry.get("predictions"), list):
        raise ValueError("invalid checkpoint predictions")
    for prediction in entry["predictions"]:
        if prediction.get("frame_index") not in frame_ids or prediction.get("window_id") != window_id:
            raise ValueError("checkpoint prediction outside window")
        box = prediction.get("bbox")
        if (not isinstance(prediction.get("track_id"), str) or not prediction["track_id"]
                or not isinstance(prediction.get("label"), str) or not prediction["label"]
                or not isinstance(box, list) or len(box) != 4
                or any(type(x) not in {int, float} or not math.isfinite(x) or x < 0 for x in box)
                or box[0] >= box[2] or box[1] >= box[3]):
            raise ValueError("invalid checkpoint box or track")
        raw = prediction.get("mask_rle")
        if raw:
            h, w, counts = raw["height"], raw["width"], raw["counts"]
            digest = hashlib.sha256(json.dumps([h, w, counts], separators=(",", ":")).encode("ascii")).hexdigest()
            if (type(h) is not int or type(w) is not int or min(h, w) < 1
                    or not isinstance(counts, list) or any(type(x) is not int or x < 0 for x in counts)
                    or sum(counts) != h * w or digest != raw["sha256"]):
                raise ValueError("checkpoint mask checksum mismatch")

