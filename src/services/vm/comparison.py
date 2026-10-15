"""Offline batch/stream comparison that refuses different media or model profiles."""

import json
from collections import Counter
from pathlib import Path

from src.services.vm.contracts import file_sha256
from src.services.vm.storage import encode


def compare(job_root: Path, batch: dict) -> dict:
    manifest = job_root / "public" / "input-manifest.json"
    profile = json.loads((job_root / "profile.json").read_text())
    if (
        batch.get("schema") != "autolabel-redacted-batch-reference/v1"
        or batch.get("input_manifest_sha256") != file_sha256(manifest)
        or batch.get("profile") != profile
    ):
        raise ValueError("Compare the same verified redacted frames and frozen profile")
    stream = []
    for path in sorted((job_root / "checkpoints").glob("*.json")):
        if "-boundary" in path.stem or "-partial" in path.stem:
            continue
        checkpoint = json.loads(path.read_text())
        for entry in checkpoint["events"]:
            event_path = job_root / entry["path"]
            if file_sha256(event_path) != entry["sha256"]:
                raise ValueError("Streaming comparison checkpoint checksum differs")
            event = json.loads(event_path.read_text())
            event["observations"] = [
                {**r, "track_id": checkpoint["mapping"].get(r["track_id"], r["track_id"])}
                for r in event["observations"]
            ]
            stream.append(event)

    def key(event: dict) -> bytes:
        return encode({k: event[k] for k in ("window", "frame", "class", "image_sha256", "observations")})

    left, right = Counter(key(e) for e in stream), Counter(key(e) for e in batch["events"])
    if Counter((e["window"], e["frame"], e["class"], e["image_sha256"]) for e in stream) != Counter(
        (e["window"], e["frame"], e["class"], e["image_sha256"]) for e in batch["events"]
    ):
        raise ValueError("Batch reference class/frame/image coverage differs")
    return {
        "schema": "autolabel-redacted-comparison/v1",
        "exact_event_equivalence": left == right,
        "stream_events": len(stream),
        "batch_events": len(batch["events"]),
        "different_stream_events": sum((left - right).values()),
        "different_batch_events": sum((right - left).values()),
        "input_manifest_sha256": file_sha256(manifest),
        "quality_after_blur": "not_evaluated",
        "note": "Exact output comparison on redacted media; it is not a quality or privacy guarantee",
    }
