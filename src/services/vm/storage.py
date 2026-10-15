"""Additive VM tables, atomic artifacts and per-process exclusive workers."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4


def encode(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def atomic_json(path: Path, value: Any) -> str:
    data = encode(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(data).hexdigest()


@contextmanager
def worker_lock(path: Path) -> Iterator[None]:
    """OS releases the lock on crashes; no stale-file deletion is needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0)
        handle.write(b"0")
        handle.flush()
        handle.seek(0)
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            getattr(fcntl, "flock")(handle.fileno(), getattr(fcntl, "LOCK_EX") | getattr(fcntl, "LOCK_NB"))
        try:
            yield
        finally:
            handle.seek(0)
            if sys.platform == "win32":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                getattr(fcntl, "flock")(handle.fileno(), getattr(fcntl, "LOCK_UN"))


class VmStore:
    def __init__(self, path: Path, root: Path):
        self.path, self.root = path, root
        path.parent.mkdir(parents=True, exist_ok=True)
        root.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
              CREATE TABLE IF NOT EXISTS vm_jobs (
                id TEXT PRIMARY KEY, owner INTEGER NOT NULL, status TEXT NOT NULL,
                payload TEXT NOT NULL, credentials TEXT NOT NULL, updated REAL NOT NULL);
              CREATE TABLE IF NOT EXISTS vm_events (
                job TEXT NOT NULL, event TEXT NOT NULL, payload TEXT NOT NULL,
                PRIMARY KEY(job,event));
              CREATE TABLE IF NOT EXISTS vm_vlm_profiles (
                owner INTEGER NOT NULL, version TEXT NOT NULL, payload TEXT NOT NULL,
                current INTEGER NOT NULL, PRIMARY KEY(owner,version));
            """)

    def connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        return db

    def directory(self, job: str) -> Path:
        if str(UUID(job)) != job:
            raise ValueError("Invalid VM job ID")
        return self.root / job

    def create(self, owner: int, payload: dict, credentials: str) -> dict:
        job = str(uuid4())
        directory = self.directory(job)
        directory.mkdir(mode=0o700)
        for name in ("private", "public", "events", "checkpoints"):
            (directory / name).mkdir(mode=0o700)
        payload = {
            **payload,
            "job_id": job,
            "status": "uploading",
            "stage": "uploading",
            "privacy_frames": 0,
            "delivered_frames": 0,
            "processed_frames": 0,
            "complete_frames": 0,
            "published_frames": 0,
            "total_frames": 0,
        }
        with self.connect() as db:
            db.execute(
                "INSERT INTO vm_jobs VALUES(?,?,?,?,?,julianday('now'))",
                (job, owner, "uploading", encode(payload).decode(), credentials),
            )
        return payload

    def get(self, job: str, owner: int | None = None) -> dict:
        with self.connect() as db:
            row = db.execute("SELECT * FROM vm_jobs WHERE id=?", (job,)).fetchone()
        if row is None or (owner is not None and owner != row["owner"]):
            raise KeyError("VM job not found")
        return {**json.loads(row["payload"]), "owner": row["owner"]}

    def credentials(self, job: str) -> str:
        with self.connect() as db:
            row = db.execute("SELECT credentials FROM vm_jobs WHERE id=?", (job,)).fetchone()
        if row is None:
            raise KeyError("VM job not found")
        return str(row["credentials"])

    def renew_credentials(self, job: str, owner: int, credentials: str) -> None:
        self.get(job, owner)
        with self.connect() as db:
            db.execute("UPDATE vm_jobs SET credentials=? WHERE id=? AND owner=?", (credentials, job, owner))

    def review(self, job: str, owner: int, track: str, state: str) -> None:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT payload FROM vm_jobs WHERE id=? AND owner=?", (job, owner)).fetchone()
            if row is None:
                raise KeyError("VM job not found")
            payload = json.loads(row[0])
            payload.setdefault("review_states", {})[track] = state
            db.execute("UPDATE vm_jobs SET payload=? WHERE id=?", (encode(payload).decode(), job))

    def update(self, job: str, **values: Any) -> dict:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT payload,status FROM vm_jobs WHERE id=?", (job,)).fetchone()
            if row is None:
                raise KeyError("VM job not found")
            payload = {**json.loads(row["payload"]), **values}
            if row["status"] == "cancelled" and values.get("status") != "queued":
                raise ValueError("VM job cancelled")
            db.execute(
                "UPDATE vm_jobs SET payload=?,status=?,updated=julianday('now') WHERE id=?",
                (encode(payload).decode(), payload["status"], job),
            )
        return payload

    def next(self) -> dict | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT id FROM vm_jobs WHERE status IN ('queued','running') ORDER BY updated LIMIT 1"
            ).fetchone()
        return self.get(row["id"]) if row else None

    def event(self, job: str, event: str, payload: dict) -> bool:
        with self.connect() as db:
            previous = db.execute("SELECT payload FROM vm_events WHERE job=? AND event=?", (job, event)).fetchone()
            value = encode(payload).decode()
            if previous:
                if previous[0] != value:
                    raise ValueError("Immutable frame event changed on retry")
                return False
            db.execute("INSERT INTO vm_events VALUES(?,?,?)", (job, event, value))
        return True
