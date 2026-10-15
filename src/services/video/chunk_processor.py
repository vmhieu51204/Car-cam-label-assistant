"""Run a whole sampled timeline with one model load and durable chunk artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from src.services.video.chunks import ChunkStore, plan_chunks
from src.services.video.stitching import (
    ChunkObservation,
    ChunkTrack,
    MaskRLE,
    StitchConfig,
    StitchResult,
    stitch_tracks,
)


class ChunkBackend(Protocol):
    """SAM3 GPU adapter or a deterministic offline fake with the same boundary."""

    version: str

    def load(self) -> None: ...

    def run_chunk(self, *, index: int, frames: list[int], labels: tuple[str, ...]) -> list[ChunkTrack]: ...

    def close(self) -> None: ...


def _write_json(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)
    return hashlib.sha256(encoded).hexdigest()


def _read_tracks(path: Path, *, expected_hash: str, input_hash: str, model_version: str) -> list[ChunkTrack]:
    encoded = path.read_bytes()
    if hashlib.sha256(encoded).hexdigest() != expected_hash:
        raise ValueError("stored chunk artifact hash mismatch")
    payload = json.loads(encoded)
    if (
        payload.get("schema") != "autolabel-chunk-output/v1"
        or payload.get("input_sha256") != input_hash
        or payload.get("model_version") != model_version
    ):
        raise ValueError("stored chunk artifact has different provenance")
    tracks = []
    for row in payload["tracks"]:
        observations = []
        for item in row["observations"]:
            raw = item.get("mask")
            mask = (
                MaskRLE(height=raw["height"], width=raw["width"], counts=tuple(raw["counts"]), sha256=raw["sha256"])
                if raw is not None
                else None
            )
            observations.append(
                ChunkObservation(frame=item["frame"], bbox=tuple(item["bbox"]), mask=mask, score=item.get("score"))
            )
        tracks.append(
            ChunkTrack(
                window_id=row["window_id"],
                local_track_id=row["local_track_id"],
                label=row["label"],
                observations=observations,
            )
        )
    return tracks


class ChunkProcessor:
    """A single call processes all chunks; a retry reuses verified completed artifacts."""

    def __init__(
        self,
        *,
        store: ChunkStore,
        artifact_root: Path,
        backend: ChunkBackend,
        window_size: int = 50,
        overlap: int = 5,
        max_attempts: int = 2,
        stitch: StitchConfig | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self.store = store
        self.artifact_root = artifact_root
        self.backend = backend
        self.window_size = window_size
        self.overlap = overlap
        self.max_attempts = max_attempts
        self.stitch = stitch or StitchConfig()

    def run(
        self,
        job_id: str,
        *,
        frame_count: int,
        input_sha256: str,
        labels: tuple[str, ...],
        scene_cut_frames: tuple[int, ...] = (),
    ) -> StitchResult:
        error = None
        try:
            return self._run(
                job_id,
                frame_count=frame_count,
                input_sha256=input_sha256,
                labels=labels,
                scene_cut_frames=scene_cut_frames,
            )
        except Exception as failure:
            error = f"{type(failure).__name__}: {failure}"
            raise
        finally:
            records = self.store.list(job_id)
            if records:
                _write_json(
                    self.artifact_root / job_id / "frame-processing.json",
                    {
                        "schema": "autolabel-frame-processing/v1",
                        "source_sha256": input_sha256,
                        "run_error": error,
                        "frames": [
                            {
                                "frame_index": frame,
                                "window_ids": [r.index for r in records if r.start <= frame < r.stop],
                                "status": "processed"
                                if error is None
                                and all(r.status == "done" for r in records if r.start <= frame < r.stop)
                                else "failed",
                                "errors": ([error] if error else [])
                                + [r.error for r in records if r.start <= frame < r.stop and r.status != "done"],
                            }
                            for frame in range(frame_count)
                        ],
                    },
                )

    def _run(
        self,
        job_id: str,
        *,
        frame_count: int,
        input_sha256: str,
        labels: tuple[str, ...],
        scene_cut_frames: tuple[int, ...],
    ) -> StitchResult:
        if not labels or len(set(labels)) != len(labels) or any(not label.strip() for label in labels):
            raise ValueError("unique nonempty text prompts are required")
        plans = plan_chunks(
            frame_count, window_size=self.window_size, overlap=self.overlap, scene_cut_frames=scene_cut_frames
        )
        run_input_sha256 = hashlib.sha256(
            json.dumps(
                {
                    "source_sha256": input_sha256,
                    "labels": labels,
                    "window_size": self.window_size,
                    "overlap": self.overlap,
                    "model_version": self.backend.version,
                    **({"scene_cut_frames": scene_cut_frames} if scene_cut_frames else {}),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        self.store.register(job_id, plans, model_version=self.backend.version, input_sha256=run_input_sha256)
        directory = self.artifact_root / job_id
        if any(item.status in {"pending", "failed", "running"} for item in self.store.list(job_id)):
            try:
                self.backend.load()
                while (claim := self.store.claim(job_id, max_attempts=self.max_attempts)) is not None:
                    try:
                        tracks = self.backend.run_chunk(
                            index=claim.index, frames=list(range(claim.start, claim.stop)), labels=labels
                        )
                        if any(
                            track.window_id != claim.index
                            or any(not claim.start <= obs.frame < claim.stop for obs in track.observations)
                            for track in tracks
                        ):
                            raise ValueError("backend returned tracks outside the claimed chunk")
                        digest = _write_json(
                            directory / f"chunk-{claim.index:04d}.json",
                            {
                                "schema": "autolabel-chunk-output/v1",
                                "input_sha256": claim.input_sha256,
                                "model_version": claim.model_version,
                                "window_id": claim.index,
                                "start": claim.start,
                                "stop": claim.stop,
                                "tracks": [asdict(track) for track in tracks],
                            },
                        )
                        self.store.complete(claim, output_sha256=digest)
                    except Exception as error:
                        self.store.fail(claim, f"{type(error).__name__}: {error}")
            finally:
                self.backend.close()
        records = self.store.list(job_id)
        if any(item.status != "done" or item.output_sha256 is None for item in records):
            failures = [(item.index, item.error) for item in records if item.status != "done"]
            raise RuntimeError(f"sequence has incomplete chunks: {failures}")
        raw = [
            track
            for item in records
            for track in _read_tracks(
                directory / f"chunk-{item.index:04d}.json",
                expected_hash=item.output_sha256 or "",
                input_hash=item.input_sha256,
                model_version=item.model_version,
            )
        ]
        result = stitch_tracks(
            raw,
            self.stitch,
            reset_windows={p.index for p in plans if p.start in scene_cut_frames},
            planned_window_ids=[p.index for p in plans],
        )
        _write_json(
            directory / "stitched.json",
            {
                "schema": "autolabel-stitched-tracks/v1",
                "input_sha256": run_input_sha256,
                "source_sha256": input_sha256,
                "model_version": self.backend.version,
                "tracks": [asdict(track) for track in result.tracks],
                "decisions": [asdict(item) for item in result.decisions],
                "boundary_logs": result.boundary_logs,
                "decision_log_enabled": self.stitch.decision_log_enabled,
                "runtime": getattr(self.backend, "runtime", None),
            },
        )
        return result
