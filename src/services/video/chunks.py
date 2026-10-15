"""Bounded video windows with durable, independently retryable SQLite claims."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from src.services.storage.repository import Store


@dataclass(frozen=True)
class ChunkPlan:
    index: int
    start: int  # inclusive manifest frame index
    stop: int  # exclusive manifest frame index


@dataclass(frozen=True)
class ChunkRecord(ChunkPlan):
    job_id: str
    status: str
    attempts: int
    model_version: str
    input_sha256: str
    output_sha256: str | None
    error: str | None


@dataclass(frozen=True)
class ChunkClaim(ChunkRecord):
    token: str


def plan_chunks(
    frame_count: int, *, window_size: int = 50, overlap: int = 5, scene_cut_frames: tuple[int, ...] = ()
) -> list[ChunkPlan]:
    """Cover the full sampled timeline; preserve overlap in both chunk outputs.

    The default is an uncalibrated T4 engineering choice from KITTI 0020;
    full-video quality and peak VRAM remain scene dependent.
    """
    if frame_count < 1 or window_size < 2 or not 0 <= overlap < window_size:
        raise ValueError("invalid frame count, window size, or overlap")
    result: list[ChunkPlan] = []
    if list(scene_cut_frames) != sorted(set(scene_cut_frames)) or any(
        not 0 < cut < frame_count for cut in scene_cut_frames
    ):
        raise ValueError("scene cuts must be unique ordered interior frames")
    boundaries = (0, *scene_cut_frames, frame_count)
    for left, right in zip(boundaries, boundaries[1:]):
        start = left
        while start < right:
            stop = min(right, start + window_size)
            result.append(ChunkPlan(len(result), start, stop))
            if stop == right:
                break
            start = stop - overlap
    return result


def _record(row: tuple) -> ChunkRecord:
    return ChunkRecord(
        index=row[1],
        start=row[2],
        stop=row[3],
        job_id=row[0],
        status=row[4],
        attempts=row[5],
        model_version=row[6],
        input_sha256=row[7],
        output_sha256=row[8],
        error=row[9],
    )


_COLUMNS = (
    "job_id, chunk_index, start_frame, stop_frame, status, attempts, model_version, input_sha256, output_sha256, error"
)


class ChunkStore:
    """One chunk may fail or expire without discarding previously completed chunks."""

    def __init__(self, store: Store) -> None:
        self.store = store

    def register(self, job_id: str, plans: list[ChunkPlan], *, model_version: str, input_sha256: str) -> None:
        if not plans or not model_version or len(input_sha256) != 64:
            raise ValueError("invalid chunk registration")
        if [plan.index for plan in plans] != list(range(len(plans))):
            raise ValueError("chunk indices must be contiguous")
        now = datetime.now(UTC).isoformat()
        with self.store.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                f"SELECT {_COLUMNS} FROM video_chunks WHERE job_id=? ORDER BY chunk_index", (job_id,)
            ).fetchall()
            if existing:
                old = [_record(row) for row in existing]
                if [(item.index, item.start, item.stop) for item in old] != [
                    (item.index, item.start, item.stop) for item in plans
                ] or any(item.model_version != model_version or item.input_sha256 != input_sha256 for item in old):
                    raise ValueError("existing chunk plan has different inputs or model version")
                return
            connection.executemany(
                """INSERT INTO video_chunks
                   (job_id, chunk_index, start_frame, stop_frame, status, model_version, input_sha256, updated_at)
                   VALUES (?, ?, ?, ?, 'pending', ?, ?, ?)""",
                [(job_id, plan.index, plan.start, plan.stop, model_version, input_sha256, now) for plan in plans],
            )

    def list(self, job_id: str) -> list[ChunkRecord]:
        with self.store.connect() as connection:
            rows = connection.execute(
                f"SELECT {_COLUMNS} FROM video_chunks WHERE job_id=? ORDER BY chunk_index", (job_id,)
            ).fetchall()
        return [_record(row) for row in rows]

    def claim(self, job_id: str, *, max_attempts: int = 3, lease_seconds: float = 3600) -> ChunkClaim | None:
        if max_attempts < 1 or lease_seconds <= 0:
            raise ValueError("invalid retry or lease policy")
        now = datetime.now(UTC)
        with self.store.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            parent = connection.execute("SELECT status FROM video_jobs WHERE job_id=?", (job_id,)).fetchone()
            if parent is None:
                raise KeyError(job_id)
            if parent[0] in {"cancelled", "completed"}:
                return None
            connection.execute(
                """UPDATE video_chunks SET status='failed', claim_token=NULL,
                   claim_expires_at=NULL, error='claim expired', updated_at=?
                   WHERE job_id=? AND status='running' AND claim_expires_at<=?""",
                (now.isoformat(), job_id, now.isoformat()),
            )
            row = connection.execute(
                f"""SELECT {_COLUMNS} FROM video_chunks WHERE job_id=?
                    AND status IN ('pending', 'failed') AND attempts<?
                    ORDER BY chunk_index LIMIT 1""",
                (job_id, max_attempts),
            ).fetchone()
            if row is None:
                return None
            item = _record(row)
            token = str(uuid4())
            connection.execute(
                """UPDATE video_chunks SET status='running', attempts=attempts+1,
                   claim_token=?, claim_expires_at=?, error=NULL, updated_at=?
                   WHERE job_id=? AND chunk_index=?""",
                (token, (now + timedelta(seconds=lease_seconds)).isoformat(), now.isoformat(), job_id, item.index),
            )
            return ChunkClaim(
                index=item.index,
                start=item.start,
                stop=item.stop,
                job_id=job_id,
                status="running",
                attempts=item.attempts + 1,
                model_version=item.model_version,
                input_sha256=item.input_sha256,
                output_sha256=None,
                error=None,
                token=token,
            )

    def complete(self, claim: ChunkClaim, *, output_sha256: str) -> None:
        if len(output_sha256) != 64:
            raise ValueError("invalid chunk output hash")
        with self.store.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            changed = connection.execute(
                """UPDATE video_chunks SET status='done', output_sha256=?,
                   claim_token=NULL, claim_expires_at=NULL, updated_at=?
                   WHERE job_id=? AND chunk_index=? AND status='running' AND claim_token=?""",
                (output_sha256, datetime.now(UTC).isoformat(), claim.job_id, claim.index, claim.token),
            )
            if changed.rowcount != 1:
                raise ValueError("chunk claim is no longer valid")

    def fail(self, claim: ChunkClaim, error: str) -> None:
        with self.store.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            changed = connection.execute(
                """UPDATE video_chunks SET status='failed', error=?, claim_token=NULL,
                   claim_expires_at=NULL, updated_at=?
                   WHERE job_id=? AND chunk_index=? AND status='running' AND claim_token=?""",
                (error[:500], datetime.now(UTC).isoformat(), claim.job_id, claim.index, claim.token),
            )
            if changed.rowcount != 1:
                raise ValueError("chunk claim is no longer valid")

    def cancel(self, job_id: str) -> None:
        with self.store.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """UPDATE video_chunks SET status='cancelled', claim_token=NULL,
                   claim_expires_at=NULL, updated_at=?
                   WHERE job_id=? AND status IN ('pending', 'running', 'failed')""",
                (datetime.now(UTC).isoformat(), job_id),
            )
