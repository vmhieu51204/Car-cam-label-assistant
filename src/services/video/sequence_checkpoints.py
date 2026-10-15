"""Durable Kaggle window checkpoints; reuse only identical code, input and model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from kaggle.checkpoint_contract import checkpoint_identity, validate_checkpoint
from src.models.video import FrameManifest
from src.services.storage.repository import Store
from src.services.video.chunks import ChunkClaim, ChunkStore, plan_chunks


class SequenceCheckpoints:
    def __init__(
        self, store: Store, directory: Path, manifest: FrameManifest, frames: list[dict], config: dict, code_hash: str
    ) -> None:
        self.store = ChunkStore(store)
        self.job_id = manifest.job_id
        self.directory = directory / "sequence-checkpoints"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.plans = plan_chunks(
            manifest.frame_count,
            window_size=config["window_size"],
            overlap=config["overlap"],
            scene_cut_frames=tuple(config.get("scene_cut_frames", [])),
        )
        self.identity = checkpoint_identity(frames, config, code_hash)
        self.config = config
        self.frames = frames
        self.store.register(
            self.job_id,
            self.plans,
            model_version=f"{config.get('model_variant', 'sam3')}-notebook@{code_hash}",
            input_sha256=self.identity,
        )
        self.claims: dict[int, ChunkClaim] = {}

    def resume(self) -> list[dict]:
        result = []
        for record in self.store.list(self.job_id):
            if record.status != "done":
                continue
            encoded = (self.directory / f"window-{record.index:04d}.json").read_bytes()
            if hashlib.sha256(encoded).hexdigest() != record.output_sha256:
                raise ValueError("stored window file hash mismatch")
            entry = json.loads(encoded)
            self.validate(entry)
            result.append(entry)
        return result

    def validate(self, entry: dict) -> None:
        index = entry.get("window_id")
        if type(index) is not int or not 0 <= index < len(self.plans):
            raise ValueError("checkpoint window outside plan")
        plan = self.plans[index]
        validate_checkpoint(
            entry, identity=self.identity, window_id=index, frame_ids=list(range(plan.start, plan.stop))
        )
        if entry.get("config_id") != self.config["id"]:
            raise ValueError("checkpoint configuration mismatch")
        for prediction in entry["predictions"]:
            frame = self.frames[prediction["frame_index"]]
            if (
                prediction["label"] not in self.config["text_prompts"]
                or prediction["bbox"][2] > frame["width"]
                or prediction["bbox"][3] > frame["height"]
            ):
                raise ValueError("checkpoint class or box outside input")

    def claim_pending(self) -> None:
        while (claim := self.store.claim(self.job_id, max_attempts=3, lease_seconds=21_600)) is not None:
            self.claims[claim.index] = claim
        if any(row.status != "done" and row.index not in self.claims for row in self.store.list(self.job_id)):
            self.fail_remaining("chunk retry limit or active claim prevents submission")
            raise ValueError("chunk retry limit or active claim prevents submission")

    def ingest(self, entries: list[dict]) -> None:
        if len({entry.get("window_id") for entry in entries}) != len(entries):
            raise ValueError("duplicate window checkpoint")
        for entry in entries:
            self.validate(entry)
        for entry in entries:
            index = entry["window_id"]
            if index not in self.claims:
                # A resumed window must agree byte-for-byte with its verified local copy.
                previous = json.loads((self.directory / f"window-{index:04d}.json").read_bytes())
                if previous != entry:
                    raise ValueError("resumed checkpoint changed")
                continue
            encoded = json.dumps(entry, sort_keys=True, separators=(",", ":")).encode()
            path = self.directory / f"window-{index:04d}.json"
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(encoded)
            temporary.replace(path)
            self.store.complete(self.claims.pop(index), output_sha256=hashlib.sha256(encoded).hexdigest())

    def fail_remaining(self, reason: str) -> None:
        for claim in self.claims.values():
            try:
                self.store.fail(claim, reason)
            except ValueError:
                pass  # Parent cancellation revoked the chunk claim.
        self.claims.clear()
        self.write_frame_status()

    def write_frame_status(self) -> None:
        """Coverage from verified chunk completion, never from box presence."""
        records = {row.index: row for row in self.store.list(self.job_id)}
        rows = []
        for frame in self.frames:
            index = frame["source_frame_index"]
            windows = [p.index for p in self.plans if p.start <= index < p.stop]
            states = [records[w].status for w in windows]
            status = (
                "processed"
                if states and all(s == "done" for s in states)
                else ("failed" if "failed" in states else "pending")
            )
            rows.append(
                {
                    "frame_index": index,
                    "source_frame_index": frame["original_source_index"],
                    "window_ids": windows,
                    "status": status,
                    "errors": [getattr(records[w], "error", None) for w in windows if records[w].status == "failed"],
                }
            )
        payload = {
            "schema": "autolabel-frame-processing/v1",
            "checkpoint_identity": self.identity,
            "frames": rows,
            "annotation_provenance": "model coverage; empty detections are valid",
        }
        path = self.directory.parent / "frame-processing.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(path)
