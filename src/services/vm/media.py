"""One fail-closed gate for every consumer of privacy output."""

import json
from pathlib import Path

from src.services.vm.contracts import file_sha256
from src.services.vm.storage import VmStore


def verified_media(store: VmStore, job: dict) -> tuple[Path, dict]:
    video = store.directory(job["job_id"]) / "public" / "redacted.mp4"
    receipt = json.loads(video.with_suffix(".privacy.json").read_text())
    profile = job["privacy_profile"]
    model = receipt.get("model", {})
    expected = {
        "face_sha256": profile["privacy_face_sha256"],
        "plate_sha256": profile["privacy_plate_sha256"],
        "source_commit": profile["privacy_source_commit"],
    }
    if (
        receipt.get("status") != "complete"
        or receipt["source_sha256"] != job["source_sha256"]
        or model != expected
        or receipt.get("manual_regions") != job["manual_regions"]
        or file_sha256(video) != receipt["output_sha256"]
        or file_sha256(video.with_suffix(".regions.jsonl")) != receipt["regions_sha256"]
    ):
        raise ValueError("Verified redacted media identity differs")
    return video, receipt
