import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from src.models.perception import QueueQuery, TelemetryRequest, Track
from src.models.video import ComponentVersion, VideoJob, VideoJobStatus

SCHEMA_VERSION = 13
VIDEO_JOB_COLUMNS = """job_id, status, source_name, source_sha256, source_size_bytes,
    created_at, updated_at, stage, progress_current, progress_total,
    manifest_sha256, cvat_job_id, seed_revision, seed_count, seed_sha256,
    cvat_task_id, component_versions, track_set_sha256, discovery_sha256,
    audit_sha256, review_queue_sha256, publish_sha256, cvat_annotation_revision,
    published_track_count, error_code, error_message"""


@dataclass(frozen=True)
class VideoJobClaim:
    job: VideoJob
    claim_token: str


class Store:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            if connection.execute("PRAGMA user_version").fetchone()[0] > SCHEMA_VERSION:
                raise ValueError("Database schema is newer than this backend")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS video_export_options (
                    job_id TEXT PRIMARY KEY REFERENCES video_jobs(job_id), config TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS video_track_review (
                    job_id TEXT NOT NULL REFERENCES video_jobs(job_id),
                    track_id TEXT NOT NULL, state TEXT NOT NULL CHECK(state IN ('unseen','seen','resolved')),
                    PRIMARY KEY(job_id, track_id)
                );
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, scene TEXT NOT NULL, metadata TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tracks (
                    run_id TEXT NOT NULL REFERENCES runs(id), id TEXT NOT NULL,
                    payload TEXT NOT NULL, PRIMARY KEY (run_id, id)
                );
                CREATE TABLE IF NOT EXISTS telemetry (
                    event_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, track_id TEXT NOT NULL,
                    received_at TEXT NOT NULL, payload TEXT NOT NULL,
                    FOREIGN KEY (run_id, track_id) REFERENCES tracks(run_id, id)
                );
                CREATE TABLE IF NOT EXISTS video_jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    source_sha256 TEXT NOT NULL,
                    source_size_bytes INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    progress_current INTEGER NOT NULL DEFAULT 0,
                    progress_total INTEGER NOT NULL DEFAULT 0,
                    manifest_sha256 TEXT,
                    cvat_job_id INTEGER,
                    seed_revision INTEGER,
                    seed_count INTEGER NOT NULL DEFAULT 0,
                    seed_sha256 TEXT,
                    cvat_task_id INTEGER,
                    component_versions TEXT NOT NULL DEFAULT '[]',
                    seed_payload TEXT,
                    claim_token TEXT,
                    claim_expires_at TEXT,
                    track_set_sha256 TEXT,
                    track_payload TEXT,
                    discovery_sha256 TEXT,
                    discovery_payload TEXT,
                    audit_sha256 TEXT,
                    audit_payload TEXT,
                    review_queue_sha256 TEXT,
                    review_queue_payload TEXT,
                    publish_sha256 TEXT,
                    publish_payload TEXT,
                    cvat_annotation_revision INTEGER,
                    published_track_count INTEGER NOT NULL DEFAULT 0,
                    error_code TEXT,
                    error_message TEXT
                );
                CREATE INDEX IF NOT EXISTS video_jobs_status_created
                    ON video_jobs(status, created_at);
                CREATE TABLE IF NOT EXISTS video_chunks (
                    job_id TEXT NOT NULL REFERENCES video_jobs(job_id),
                    chunk_index INTEGER NOT NULL,
                    start_frame INTEGER NOT NULL,
                    stop_frame INTEGER NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('pending','running','done','failed','cancelled')),
                    attempts INTEGER NOT NULL DEFAULT 0,
                    model_version TEXT NOT NULL,
                    input_sha256 TEXT NOT NULL,
                    output_sha256 TEXT,
                    claim_token TEXT,
                    claim_expires_at TEXT,
                    error TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (job_id, chunk_index)
                );
                CREATE INDEX IF NOT EXISTS video_chunks_status
                    ON video_chunks(job_id, status, chunk_index);
                CREATE TABLE IF NOT EXISTS video_user_intents (
                    intent_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    cvat_user_id INTEGER NOT NULL,
                    cvat_job_id INTEGER NOT NULL,
                    video_job_id TEXT,
                    method TEXT NOT NULL,
                    route TEXT NOT NULL,
                    completed_at TEXT,
                    http_status INTEGER
                );
                CREATE INDEX IF NOT EXISTS video_user_intents_job_created
                    ON video_user_intents(cvat_job_id, created_at);
                CREATE TABLE IF NOT EXISTS video_sequence_audits (
                    job_id TEXT PRIMARY KEY REFERENCES video_jobs(job_id),
                    status TEXT NOT NULL CHECK (status IN ('queued','running','done','failed')),
                    attempts INTEGER NOT NULL DEFAULT 0,
                    claim_token TEXT,
                    claim_expires_at TEXT,
                    partial_payload TEXT NOT NULL DEFAULT '[]',
                    error TEXT,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS video_sequence_audits_status
                    ON video_sequence_audits(status, updated_at);
                CREATE TABLE IF NOT EXISTS video_box_rescues (
                    job_id TEXT PRIMARY KEY REFERENCES video_jobs(job_id),
                    status TEXT NOT NULL CHECK (status IN ('queued','running','done','failed')),
                    attempts INTEGER NOT NULL DEFAULT 0,
                    claim_token TEXT,
                    claim_expires_at TEXT,
                    partial_payload TEXT NOT NULL DEFAULT '[]',
                    proposals_payload TEXT NOT NULL DEFAULT '[]',
                    error TEXT,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS video_box_rescues_status
                    ON video_box_rescues(status, updated_at);
            """)
            # Serialize column checks and ALTER TABLE across API/worker startup.
            connection.execute("BEGIN IMMEDIATE")
            _ensure_video_job_columns(connection)
            intent_columns = {row[1] for row in connection.execute("PRAGMA table_info(video_user_intents)")}
            if "completed_at" not in intent_columns:
                connection.execute("ALTER TABLE video_user_intents ADD COLUMN completed_at TEXT")
            if "http_status" not in intent_columns:
                connection.execute("ALTER TABLE video_user_intents ADD COLUMN http_status INTEGER")
            _recover_legacy_unleased_extractions(connection)
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def get_export_options(self, job_id: str) -> dict:
        self.get_video_job(job_id)
        with self.connect() as c:
            row = c.execute("SELECT config FROM video_export_options WHERE job_id=?", (job_id,)).fetchone()
        return json.loads(row[0]) if row else {}

    def set_export_options(self, job_id: str, config: dict) -> None:
        self.get_video_job(job_id)
        with self.connect() as c:
            c.execute(
                "INSERT INTO video_export_options VALUES(?,?) ON CONFLICT(job_id) DO UPDATE SET config=excluded.config",
                (job_id, json.dumps(config, sort_keys=True)),
            )

    def track_review_states(self, job_id: str) -> dict[str, str]:
        self.get_video_job(job_id)
        with self.connect() as c:
            return dict(c.execute("SELECT track_id,state FROM video_track_review WHERE job_id=?", (job_id,)).fetchall())

    def set_track_review_state(self, job_id: str, track_id: str, state: str) -> None:
        self.get_video_job(job_id)
        if state not in {"unseen", "seen", "resolved"}:
            raise ValueError("invalid track review state")
        with self.connect() as c:
            c.execute(
                "INSERT INTO video_track_review VALUES(?,?,?) ON CONFLICT(job_id,track_id) DO UPDATE SET state=excluded.state",
                (job_id, track_id, state),
            )

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Open one transaction with enforced foreign keys."""
        connection = sqlite3.connect(self.path, timeout=15)
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def save_run(self, scene: str, metadata: dict, tracks: list[Track]) -> str:
        """Atomically persist a run and all observed tracks."""
        run_id = str(uuid4())
        with self.connect() as connection:
            connection.execute("INSERT INTO runs VALUES (?, ?, ?)", (run_id, scene, json.dumps(metadata)))
            connection.executemany(
                "INSERT INTO tracks VALUES (?, ?, ?)", [(run_id, t.track_id, t.model_dump_json()) for t in tracks]
            )
        return run_id

    def queue(self, query: QueueQuery) -> dict:
        """Read a stable, paginated snapshot of a run."""
        with self.connect() as connection:
            run = connection.execute("SELECT metadata FROM runs WHERE id=?", (query.run_id,)).fetchone()
            if run is None:
                raise KeyError(query.run_id)
            order = (
                "id"
                if query.order == "track_id"
                else """
                CASE json_extract(payload, '$.discrepancy.status')
                WHEN 'invalid' THEN 0 WHEN 'discrepancy' THEN 1 WHEN 'insufficient_evidence' THEN 2
                WHEN 'ambiguous' THEN 3 ELSE 4 END,
                json_extract(payload, '$.discrepancy.discrepancy_score') DESC, id
            """
            )
            rows = connection.execute(
                f"""SELECT payload FROM tracks
                    WHERE run_id=? AND json_extract(payload, '$.review_status')='pending'
                    ORDER BY {order} LIMIT ? OFFSET ?""",
                (query.run_id, query.limit, query.offset),
            ).fetchall()
            total = connection.execute(
                """SELECT COUNT(*) FROM tracks
                   WHERE run_id=? AND json_extract(payload, '$.review_status')='pending'""",
                (query.run_id,),
            ).fetchone()[0]
        return {"metadata": json.loads(run[0]), "total": total, "tracks": [json.loads(row[0]) for row in rows]}

    def telemetry(self, event: TelemetryRequest) -> bool:
        """Serialize idempotency checks so concurrent retries cannot duplicate events."""
        payload = event.model_dump_json()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            old = connection.execute("SELECT payload FROM telemetry WHERE event_id=?", (event.event_id,)).fetchone()
            if old:
                if old[0] != payload:
                    raise ValueError("event_id already exists with a different payload")
                return False
            connection.execute(
                "INSERT INTO telemetry VALUES (?, ?, ?, ?, ?)",
                (
                    event.event_id,
                    event.run_id,
                    event.track_id,
                    datetime.now(UTC).isoformat(),
                    payload,
                ),
            )
        return True

    def create_video_job(self, job: VideoJob) -> None:
        """Persist a newly uploaded video job."""
        values = _job_values(job)
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO video_jobs (
                    job_id, status, source_name, source_sha256, source_size_bytes,
                    created_at, updated_at, stage, progress_current, progress_total,
                    manifest_sha256, cvat_job_id, seed_revision, seed_count,
                    seed_sha256, cvat_task_id, component_versions, track_set_sha256,
                    discovery_sha256, audit_sha256, review_queue_sha256, publish_sha256,
                    cvat_annotation_revision, published_track_count, error_code, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                values,
            )

    def record_video_user_intent(
        self,
        *,
        cvat_user_id: int,
        cvat_job_id: int,
        video_job_id: str | None,
        method: str,
        route: str,
    ) -> str:
        """Persist the CVAT actor before an authorized operation can use the service PAT."""
        if cvat_user_id < 1 or cvat_job_id < 1 or method not in {"POST", "PUT", "PATCH", "DELETE"}:
            raise ValueError("Invalid CVAT action actor or method")
        if not route.startswith("/api/v1/video/jobs/") or len(route) > 256:
            raise ValueError("Invalid CVAT action route")
        intent_id = str(uuid4())
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO video_user_intents
                    (intent_id, created_at, cvat_user_id, cvat_job_id, video_job_id, method, route)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (intent_id, datetime.now(UTC).isoformat(), cvat_user_id, cvat_job_id, video_job_id, method, route),
            )
        return intent_id

    def finish_video_user_intent(self, intent_id: str, http_status: int) -> None:
        """Record the HTTP outcome of a previously authorized user action."""
        if not 100 <= http_status <= 599:
            raise ValueError("Invalid HTTP status for CVAT action")
        with self.connect() as connection:
            updated = connection.execute(
                """UPDATE video_user_intents SET completed_at=?, http_status=?
                   WHERE intent_id=? AND completed_at IS NULL""",
                (datetime.now(UTC).isoformat(), http_status, intent_id),
            )
            if updated.rowcount != 1:
                raise ValueError("CVAT action intent was not found or already completed")

    def get_video_job(self, job_id: str) -> VideoJob:
        with self.connect() as connection:
            row = connection.execute(f"SELECT {VIDEO_JOB_COLUMNS} FROM video_jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return _job_from_row(row)

    def find_video_job_by_cvat_job(self, cvat_job_id: int) -> VideoJob | None:
        with self.connect() as connection:
            row = connection.execute(
                f"SELECT {VIDEO_JOB_COLUMNS} FROM video_jobs WHERE cvat_job_id=? ORDER BY created_at DESC LIMIT 1",
                (cvat_job_id,),
            ).fetchone()
        return _job_from_row(row) if row is not None else None

    def transition_video_job(
        self,
        job_id: str,
        *,
        expected: set[VideoJobStatus],
        status: VideoJobStatus,
        stage: str,
        progress_current: int | None = None,
        progress_total: int | None = None,
    ) -> VideoJob:
        """Perform a guarded state transition so concurrent workers cannot steal work."""
        now = datetime.now(UTC).isoformat()
        placeholders = ",".join("?" for _ in expected)
        assignments = ["status=?", "stage=?", "updated_at=?", "error_code=NULL", "error_message=NULL"]
        values: list[object] = [status.value, stage, now]
        if progress_current is not None:
            assignments.append("progress_current=?")
            values.append(progress_current)
        if progress_total is not None:
            assignments.append("progress_total=?")
            values.append(progress_total)
        values.extend([job_id, *(item.value for item in expected)])
        with self.connect() as connection:
            cursor = connection.execute(
                f"UPDATE video_jobs SET {', '.join(assignments)} WHERE job_id=? AND status IN ({placeholders})",
                values,
            )
            if cursor.rowcount != 1:
                exists = connection.execute("SELECT 1 FROM video_jobs WHERE job_id=?", (job_id,)).fetchone()
                if exists is None:
                    raise KeyError(job_id)
                raise ValueError(f"video job {job_id} is not in an allowed state")
        return self.get_video_job(job_id)

    def retry_video_cvat_import(self, job_id: str) -> VideoJob:
        """Retry only failed extraction, preserving the CVAT binding and sampling request."""
        self.get_video_job(job_id)
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE video_jobs SET status=?, stage='queued_for_cvat_import', updated_at=?,
                    progress_current=0, progress_total=0, error_code=NULL, error_message=NULL,
                    claim_token=NULL, claim_expires_at=NULL
                    WHERE job_id=? AND status=? AND cvat_job_id IS NOT NULL
                    AND manifest_sha256 IS NULL
                    AND error_code IN ('video_extraction_failed', 'video_extraction_internal_error')""",
                (VideoJobStatus.QUEUED.value, datetime.now(UTC).isoformat(), job_id, VideoJobStatus.FAILED.value),
            )
            if cursor.rowcount != 1:
                raise ValueError("CVAT frame import is not in a retryable extraction state")
        return self.get_video_job(job_id)

    def claim_next_video_job(self, *, lease_seconds: float = 60) -> VideoJobClaim | None:
        """Atomically claim the oldest extraction job across worker processes."""
        if lease_seconds <= 0:
            raise ValueError("video extraction lease must be positive")
        now_value = datetime.now(UTC)
        now = now_value.isoformat()
        claim_token = str(uuid4())
        expires_at = (now_value + timedelta(seconds=lease_seconds)).isoformat()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT job_id FROM video_jobs WHERE status=? ORDER BY created_at, job_id LIMIT 1",
                (VideoJobStatus.QUEUED.value,),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                """UPDATE video_jobs SET status=?, stage=?, updated_at=?, claim_token=?, claim_expires_at=?,
                    progress_current=0, progress_total=0, error_code=NULL, error_message=NULL
                    WHERE job_id=? AND status=?""",
                (
                    VideoJobStatus.EXTRACTING.value,
                    "extracting_frames",
                    now,
                    claim_token,
                    expires_at,
                    row[0],
                    VideoJobStatus.QUEUED.value,
                ),
            )
        return VideoJobClaim(job=self.get_video_job(row[0]), claim_token=claim_token)

    def requeue_expired_video_extractions(self) -> int:
        """Recover only claims whose renewable worker lease has expired."""
        now = datetime.now(UTC).isoformat()
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE video_jobs SET status=?, stage=?, updated_at=?,
                    progress_current=0, progress_total=0, claim_token=NULL, claim_expires_at=NULL,
                    error_code=NULL, error_message=NULL
                    WHERE status=? AND claim_expires_at IS NOT NULL AND claim_expires_at <= ?""",
                (
                    VideoJobStatus.QUEUED.value,
                    "requeued_after_expired_lease",
                    now,
                    VideoJobStatus.EXTRACTING.value,
                    now,
                ),
            )
        return cursor.rowcount

    def update_video_progress(
        self, job_id: str, *, claim_token: str, current: int, total: int, lease_seconds: float = 60
    ) -> None:
        if current < 0 or total < 1 or current > total:
            raise ValueError("video extraction progress must satisfy 0 <= current <= total")
        if lease_seconds <= 0:
            raise ValueError("video extraction lease must be positive")
        now_value = datetime.now(UTC)
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE video_jobs SET updated_at=?, progress_current=?, progress_total=?, claim_expires_at=?
                    WHERE job_id=? AND status=? AND claim_token=? AND progress_current <= ?""",
                (
                    now_value.isoformat(),
                    current,
                    total,
                    (now_value + timedelta(seconds=lease_seconds)).isoformat(),
                    job_id,
                    VideoJobStatus.EXTRACTING.value,
                    claim_token,
                    current,
                ),
            )
        if cursor.rowcount != 1:
            raise ValueError(f"video job {job_id} extraction claim is no longer active")

    def complete_video_extraction(
        self, job_id: str, *, claim_token: str, manifest_sha256: str, frame_count: int
    ) -> VideoJob:
        now = datetime.now(UTC).isoformat()
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE video_jobs SET status=?, stage=?, updated_at=?, manifest_sha256=?,
                    progress_current=?, progress_total=?, claim_token=NULL, claim_expires_at=NULL,
                    error_code=NULL, error_message=NULL
                    WHERE job_id=? AND status=? AND claim_token=?""",
                (
                    VideoJobStatus.AWAITING_SEED.value,
                    "awaiting_seed",
                    now,
                    manifest_sha256,
                    frame_count,
                    frame_count,
                    job_id,
                    VideoJobStatus.EXTRACTING.value,
                    claim_token,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"video job {job_id} is not being extracted")
        return self.get_video_job(job_id)

    def attach_video_seed(
        self,
        job_id: str,
        *,
        cvat_job_id: int,
        seed_revision: int,
        seed_count: int,
        seed_sha256: str,
        seed_payload: str,
        expected_updated_at: datetime,
    ) -> VideoJob:
        # Keep the optimistic-lock token distinct even within one Windows clock tick.
        now = max(datetime.now(UTC), expected_updated_at + timedelta(microseconds=1)).isoformat()
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE video_jobs SET status=?, stage=?, updated_at=?, cvat_job_id=?,
                    seed_revision=?, seed_count=?, seed_sha256=?, seed_payload=?,
                    error_code=NULL, error_message=NULL
                    WHERE job_id=? AND (status IN (?, ?) OR
                    (status='failed' AND error_code='auto_seed_failed')) AND updated_at=?
                    AND (cvat_job_id IS NULL OR cvat_job_id=?)""",
                (
                    VideoJobStatus.SEED_READY.value,
                    "seed_ready",
                    now,
                    cvat_job_id,
                    seed_revision,
                    seed_count,
                    seed_sha256,
                    seed_payload,
                    job_id,
                    VideoJobStatus.AWAITING_SEED.value,
                    VideoJobStatus.SEED_READY.value,
                    expected_updated_at.isoformat(),
                    cvat_job_id,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"video job {job_id} is not ready for seed capture")
        return self.get_video_job(job_id)

    def queue_video_auto_seed(self, job_id: str) -> VideoJob:
        now = datetime.now(UTC).isoformat()
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE video_jobs SET status=?, stage=?, updated_at=?, progress_current=0,
                    progress_total=1, claim_token=NULL, claim_expires_at=NULL,
                    error_code=NULL, error_message=NULL
                    WHERE job_id=? AND manifest_sha256 IS NOT NULL AND cvat_job_id IS NOT NULL
                    AND (status=? OR (status=? AND error_code LIKE 'auto_seed_%'))""",
                (
                    VideoJobStatus.SEEDING.value,
                    "queued_for_auto_seed",
                    now,
                    job_id,
                    VideoJobStatus.AWAITING_SEED.value,
                    VideoJobStatus.FAILED.value,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"video job {job_id} is not ready for SAM3 frame-0 labeling")
        return self.get_video_job(job_id)

    def claim_next_video_auto_seed(self, *, lease_seconds: float = 120) -> VideoJobClaim | None:
        if lease_seconds <= 0:
            raise ValueError("auto-seed lease must be positive")
        now = datetime.now(UTC)
        token = str(uuid4())
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """SELECT job_id FROM video_jobs WHERE status=? AND stage=?
                    ORDER BY updated_at, job_id LIMIT 1""",
                (VideoJobStatus.SEEDING.value, "queued_for_auto_seed"),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                """UPDATE video_jobs SET stage=?, updated_at=?, claim_token=?, claim_expires_at=?
                    WHERE job_id=? AND status=? AND stage=?""",
                (
                    "auto_seeding",
                    now.isoformat(),
                    token,
                    (now + timedelta(seconds=lease_seconds)).isoformat(),
                    row[0],
                    VideoJobStatus.SEEDING.value,
                    "queued_for_auto_seed",
                ),
            )
        return VideoJobClaim(job=self.get_video_job(row[0]), claim_token=token)

    def renew_video_auto_seed(self, job_id: str, *, claim_token: str, lease_seconds: float) -> None:
        now = datetime.now(UTC)
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE video_jobs SET updated_at=?, claim_expires_at=?
                    WHERE job_id=? AND status=? AND stage=? AND claim_token=?""",
                (
                    now.isoformat(),
                    (now + timedelta(seconds=lease_seconds)).isoformat(),
                    job_id,
                    VideoJobStatus.SEEDING.value,
                    "auto_seeding",
                    claim_token,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"video job {job_id} auto-seed claim is no longer active")

    def requeue_expired_video_auto_seeds(self) -> int:
        now = datetime.now(UTC).isoformat()
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE video_jobs SET stage=?, updated_at=?, claim_token=NULL,
                    claim_expires_at=NULL WHERE status=? AND stage=?
                    AND claim_expires_at IS NOT NULL AND claim_expires_at <= ?""",
                ("queued_for_auto_seed", now, VideoJobStatus.SEEDING.value, "auto_seeding", now),
            )
        return cursor.rowcount

    def complete_video_auto_seed(
        self,
        job_id: str,
        *,
        claim_token: str,
        seed_revision: int,
        seed_count: int,
        seed_sha256: str,
        seed_payload: str,
    ) -> VideoJob:
        now = datetime.now(UTC).isoformat()
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE video_jobs SET status=?, stage=?, updated_at=?, seed_revision=?,
                    seed_count=?, seed_sha256=?, seed_payload=?, progress_current=1,
                    claim_token=NULL, claim_expires_at=NULL, error_code=NULL, error_message=NULL
                    WHERE job_id=? AND status=? AND stage=? AND claim_token=?""",
                (
                    VideoJobStatus.SEED_READY.value,
                    "seed_ready",
                    now,
                    seed_revision,
                    seed_count,
                    seed_sha256,
                    seed_payload,
                    job_id,
                    VideoJobStatus.SEEDING.value,
                    "auto_seeding",
                    claim_token,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"video job {job_id} is not being auto-seeded")
        return self.get_video_job(job_id)

    def read_video_seed(self, job_id: str) -> str:
        with self.connect() as connection:
            row = connection.execute("SELECT seed_payload FROM video_jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        if row[0] is None:
            raise ValueError(f"video job {job_id} does not have a seed set")
        return str(row[0])

    def queue_video_propagation(self, job_id: str, *, frame_count: int) -> VideoJob:
        if frame_count < 1:
            raise ValueError("video propagation requires at least one frame")
        now = datetime.now(UTC).isoformat()
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE video_jobs SET status=?, stage=?, updated_at=?, progress_current=0,
                    progress_total=?, claim_token=NULL, claim_expires_at=NULL, track_set_sha256=NULL,
                    track_payload=NULL, discovery_sha256=NULL, discovery_payload=NULL,
                    audit_sha256=NULL, audit_payload=NULL, review_queue_sha256=NULL,
                    review_queue_payload=NULL, publish_sha256=NULL, publish_payload=NULL,
                    cvat_annotation_revision=NULL, published_track_count=0,
                    error_code=NULL, error_message=NULL
                    WHERE job_id=? AND (status=? OR (status=? AND
                    (error_code LIKE 'propagation_%' OR error_code LIKE 'video_propagation_%')))
                    AND seed_payload IS NOT NULL""",
                (
                    VideoJobStatus.PROPAGATING.value,
                    "queued_for_propagation",
                    now,
                    frame_count,
                    job_id,
                    VideoJobStatus.SEED_READY.value,
                    VideoJobStatus.FAILED.value,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"video job {job_id} is not ready for propagation")
        return self.get_video_job(job_id)

    def claim_next_video_propagation(self, *, lease_seconds: float = 60) -> VideoJobClaim | None:
        if lease_seconds <= 0:
            raise ValueError("video propagation lease must be positive")
        now_value = datetime.now(UTC)
        token = str(uuid4())
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """SELECT job_id FROM video_jobs WHERE status=? AND stage=?
                    ORDER BY updated_at, job_id LIMIT 1""",
                (VideoJobStatus.PROPAGATING.value, "queued_for_propagation"),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                """UPDATE video_jobs SET stage=?, updated_at=?, claim_token=?, claim_expires_at=?,
                    progress_current=0, error_code=NULL, error_message=NULL
                    WHERE job_id=? AND status=? AND stage=?""",
                (
                    "propagating",
                    now_value.isoformat(),
                    token,
                    (now_value + timedelta(seconds=lease_seconds)).isoformat(),
                    row[0],
                    VideoJobStatus.PROPAGATING.value,
                    "queued_for_propagation",
                ),
            )
        return VideoJobClaim(job=self.get_video_job(row[0]), claim_token=token)

    def requeue_expired_video_propagations(self) -> int:
        now = datetime.now(UTC).isoformat()
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE video_jobs SET stage=?, updated_at=?, progress_current=0,
                    claim_token=NULL, claim_expires_at=NULL, error_code=NULL, error_message=NULL
                    WHERE status=? AND stage=? AND claim_expires_at IS NOT NULL AND claim_expires_at <= ?""",
                (
                    "queued_for_propagation",
                    now,
                    VideoJobStatus.PROPAGATING.value,
                    "propagating",
                    now,
                ),
            )
        return cursor.rowcount

    def update_video_propagation_progress(
        self, job_id: str, *, claim_token: str, current: int, total: int, lease_seconds: float = 60
    ) -> None:
        if current < 0 or total < 1 or current > total:
            raise ValueError("video propagation progress must satisfy 0 <= current <= total")
        if lease_seconds <= 0:
            raise ValueError("video propagation lease must be positive")
        now_value = datetime.now(UTC)
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE video_jobs SET updated_at=?, progress_current=?, progress_total=?, claim_expires_at=?
                    WHERE job_id=? AND status=? AND stage=? AND claim_token=? AND progress_current <= ?""",
                (
                    now_value.isoformat(),
                    current,
                    total,
                    (now_value + timedelta(seconds=lease_seconds)).isoformat(),
                    job_id,
                    VideoJobStatus.PROPAGATING.value,
                    "propagating",
                    claim_token,
                    current,
                ),
            )
        if cursor.rowcount != 1:
            raise ValueError(f"video job {job_id} propagation claim is no longer active")

    def complete_video_propagation(
        self,
        job_id: str,
        *,
        claim_token: str,
        track_set_sha256: str,
        track_payload: str,
        component_versions: list[ComponentVersion],
        frame_count: int,
    ) -> VideoJob:
        now = datetime.now(UTC).isoformat()
        versions = json.dumps([item.model_dump(mode="json") for item in component_versions], sort_keys=True)
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE video_jobs SET status=?, stage=?, updated_at=?, track_set_sha256=?,
                    track_payload=?, component_versions=?, progress_current=?, progress_total=?,
                    claim_token=NULL, claim_expires_at=NULL, error_code=NULL, error_message=NULL
                    WHERE job_id=? AND status=? AND stage=? AND claim_token=?""",
                (
                    VideoJobStatus.DISCOVERING.value,
                    "queued_for_discovery",
                    now,
                    track_set_sha256,
                    track_payload,
                    versions,
                    frame_count,
                    frame_count,
                    job_id,
                    VideoJobStatus.PROPAGATING.value,
                    "propagating",
                    claim_token,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"video job {job_id} is not being propagated")
        return self.get_video_job(job_id)

    def read_video_track_set(self, job_id: str) -> str:
        with self.connect() as connection:
            row = connection.execute("SELECT track_payload FROM video_jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        if row[0] is None:
            raise ValueError(f"video job {job_id} does not have a propagated track set")
        return str(row[0])

    def queue_video_sequence(self, job_id: str, *, frame_count: int) -> VideoJob:
        """Queue seed-free SAM3 text tracking, separate from legacy propagation."""
        if frame_count < 2:
            raise ValueError("sequence tracking requires at least two selected frames")
        now = datetime.now(UTC).isoformat()
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE video_jobs SET status=?, stage=?, updated_at=?, progress_current=0,
                    progress_total=?, claim_token=NULL, claim_expires_at=NULL,
                    error_code=NULL, error_message=NULL
                    WHERE job_id=? AND manifest_sha256 IS NOT NULL
                    AND (status=? OR (status=? AND error_code LIKE 'sequence_%'))""",
                (
                    VideoJobStatus.PROPAGATING.value,
                    "queued_for_sequence",
                    now,
                    frame_count,
                    job_id,
                    VideoJobStatus.AWAITING_SEED.value,
                    VideoJobStatus.FAILED.value,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"video job {job_id} is not ready for seed-free sequence tracking")
        return self.get_video_job(job_id)

    def claim_next_video_sequence(self, *, lease_seconds: float) -> VideoJobClaim | None:
        if lease_seconds < 3:
            raise ValueError("sequence lease must be at least three seconds")
        now = datetime.now(UTC)
        token = str(uuid4())
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """SELECT job_id FROM video_jobs WHERE status=? AND stage=?
                    ORDER BY updated_at, job_id LIMIT 1""",
                (VideoJobStatus.PROPAGATING.value, "queued_for_sequence"),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                """UPDATE video_jobs SET stage=?, updated_at=?, claim_token=?, claim_expires_at=?
                    WHERE job_id=? AND status=? AND stage=?""",
                (
                    "sequence_running",
                    now.isoformat(),
                    token,
                    (now + timedelta(seconds=lease_seconds)).isoformat(),
                    row[0],
                    VideoJobStatus.PROPAGATING.value,
                    "queued_for_sequence",
                ),
            )
        return VideoJobClaim(self.get_video_job(row[0]), token)

    def renew_video_sequence(
        self, job_id: str, *, claim_token: str, lease_seconds: float, progress: int, total: int
    ) -> None:
        if not 0 <= progress <= total or total < 1:
            raise ValueError("invalid sequence progress")
        now = datetime.now(UTC)
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE video_jobs SET updated_at=?, claim_expires_at=?,
                    progress_current=?, progress_total=? WHERE job_id=? AND status=?
                    AND stage=? AND claim_token=?""",
                (
                    now.isoformat(),
                    (now + timedelta(seconds=lease_seconds)).isoformat(),
                    progress,
                    total,
                    job_id,
                    VideoJobStatus.PROPAGATING.value,
                    "sequence_running",
                    claim_token,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"sequence claim for {job_id} is no longer active")

    def requeue_expired_video_sequences(self) -> int:
        now = datetime.now(UTC).isoformat()
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE video_jobs SET stage=?, updated_at=?, claim_token=NULL,
                    claim_expires_at=NULL WHERE status=? AND stage=?
                    AND claim_expires_at <= ?""",
                ("queued_for_sequence", now, VideoJobStatus.PROPAGATING.value, "sequence_running", now),
            )
        return cursor.rowcount

    def complete_video_sequence(
        self,
        job_id: str,
        *,
        claim_token: str,
        tracks: str,
        track_sha256: str,
        discovery: str,
        discovery_sha256: str,
        review: str,
        review_sha256: str,
        component_versions: list[ComponentVersion],
        frame_count: int,
        queue_box_rescue: bool = False,
    ) -> VideoJob:
        now = datetime.now(UTC).isoformat()
        versions = json.dumps([item.model_dump(mode="json") for item in component_versions], sort_keys=True)
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE video_jobs SET status=?, stage=?, updated_at=?,
                    track_payload=?, track_set_sha256=?, discovery_payload=?, discovery_sha256=?,
                    review_queue_payload=?, review_queue_sha256=?, component_versions=?,
                    progress_current=?, progress_total=?, claim_token=NULL,
                    claim_expires_at=NULL, error_code=NULL, error_message=NULL
                    WHERE job_id=? AND status=? AND stage=? AND claim_token=?""",
                (
                    VideoJobStatus.AWAITING_REVIEW.value,
                    "sequence_draft_ready",
                    now,
                    tracks,
                    track_sha256,
                    discovery,
                    discovery_sha256,
                    review,
                    review_sha256,
                    versions,
                    frame_count,
                    frame_count,
                    job_id,
                    VideoJobStatus.PROPAGATING.value,
                    "sequence_running",
                    claim_token,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"sequence claim for {job_id} is no longer active")
            if queue_box_rescue:
                connection.execute(
                    """INSERT OR IGNORE INTO video_box_rescues (job_id, status, updated_at)
                    VALUES (?, 'queued', ?)""",
                    (job_id, now),
                )
        return self.get_video_job(job_id)

    def get_video_box_rescue_state(self, job_id: str) -> dict | None:
        with self.connect() as connection:
            row = connection.execute(
                """SELECT status, attempts, proposals_payload, partial_payload, error, updated_at
                FROM video_box_rescues WHERE job_id=?""",
                (job_id,),
            ).fetchone()
        if row is None:
            return None
        proposals = json.loads(row[2])
        partial = json.loads(row[3])
        if row[0] != "done":
            proposals_by_id = {
                proposal.get("proposal_id"): proposal
                for frame_result in partial
                for proposal in frame_result.get("proposals", [])
            }
            for proposal in proposals:
                proposals_by_id.setdefault(proposal.get("proposal_id"), proposal)
            proposals = list(proposals_by_id.values())
        return {
            "status": row[0],
            "attempts": row[1],
            "proposal_count": len(proposals),
            "completed_frame_count": len(partial),
            "proposals": proposals[:20],
            "truncated_proposal_count": max(0, sum(len(r.get("proposals", [])) for r in partial) - 20),
            "error": row[4],
            "updated_at": row[5],
        }

    def claim_next_video_box_rescue(self, *, lease_seconds: float) -> VideoJobClaim | None:
        if lease_seconds < 3:
            raise ValueError("box-rescue lease must be at least three seconds")
        now = datetime.now(UTC)
        token = str(uuid4())
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """SELECT r.job_id FROM video_box_rescues r JOIN video_jobs j USING(job_id)
                WHERE j.status=? AND (j.cvat_job_id IS NULL OR j.publish_sha256 IS NOT NULL)
                    AND (r.status='queued' OR
                    (r.status='running' AND r.claim_expires_at <= ?))
                ORDER BY r.updated_at, r.job_id LIMIT 1""",
                (VideoJobStatus.AWAITING_REVIEW.value, now.isoformat()),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                """UPDATE video_box_rescues SET status='running', attempts=attempts+1,
                claim_token=?, claim_expires_at=?, updated_at=? WHERE job_id=?""",
                (token, (now + timedelta(seconds=lease_seconds)).isoformat(), now.isoformat(), row[0]),
            )
        return VideoJobClaim(self.get_video_job(row[0]), token)

    def read_video_box_rescue_partial(self, job_id: str, *, claim_token: str) -> list[dict]:
        with self.connect() as connection:
            row = connection.execute(
                """SELECT partial_payload FROM video_box_rescues
                WHERE job_id=? AND status='running' AND claim_token=?""",
                (job_id, claim_token),
            ).fetchone()
        if row is None:
            raise ValueError("box-rescue claim is no longer active")
        return json.loads(row[0])

    def save_video_box_rescue_partial(
        self, job_id: str, *, claim_token: str, partial: list[dict], lease_seconds: float
    ) -> None:
        now = datetime.now(UTC)
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE video_box_rescues SET partial_payload=?, updated_at=?, claim_expires_at=?
                WHERE job_id=? AND status='running' AND claim_token=?""",
                (
                    json.dumps(partial, sort_keys=True),
                    now.isoformat(),
                    (now + timedelta(seconds=lease_seconds)).isoformat(),
                    job_id,
                    claim_token,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("box-rescue claim is no longer active")

    def extend_video_box_rescue_lease(self, job_id: str, *, claim_token: str, lease_seconds: float) -> None:
        now = datetime.now(UTC)
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE video_box_rescues SET claim_expires_at=? WHERE job_id=?
                AND status='running' AND claim_token=?""",
                ((now + timedelta(seconds=lease_seconds)).isoformat(), job_id, claim_token),
            )
            if cursor.rowcount != 1:
                raise ValueError("box-rescue claim is no longer active")

    def complete_video_box_rescue(self, job_id: str, *, claim_token: str, proposals: list[dict]) -> None:
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE video_box_rescues SET status='done', proposals_payload=?, error=NULL,
                claim_token=NULL, claim_expires_at=NULL, updated_at=?
                WHERE job_id=? AND status='running' AND claim_token=?""",
                (json.dumps(proposals[:20], sort_keys=True), datetime.now(UTC).isoformat(), job_id, claim_token),
            )
            if cursor.rowcount != 1:
                raise ValueError("box-rescue claim is no longer active")

    def fail_video_box_rescue(self, job_id: str, *, claim_token: str, error: str) -> None:
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE video_box_rescues SET status='failed', error=?, claim_token=NULL,
                claim_expires_at=NULL, updated_at=? WHERE job_id=? AND status='running' AND claim_token=?""",
                (error[:500], datetime.now(UTC).isoformat(), job_id, claim_token),
            )
            if cursor.rowcount != 1:
                raise ValueError("box-rescue claim is no longer active")

    def retry_video_box_rescue(self, job_id: str) -> None:
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE video_box_rescues SET status='queued', error=NULL, claim_token=NULL,
                claim_expires_at=NULL, updated_at=? WHERE job_id=? AND status='failed'""",
                (datetime.now(UTC).isoformat(), job_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("VLM box rescue is not failed")

    def set_video_box_rescue_proposal_status(self, job_id: str, *, proposal_id: str, status: str) -> dict:
        if status not in {"accepting", "pending", "accepted", "rejected"}:
            raise ValueError("invalid proposal status")
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT proposals_payload FROM video_box_rescues WHERE job_id=? AND status='done'",
                (job_id,),
            ).fetchone()
            if row is None:
                raise ValueError("VLM box proposals are not ready")
            proposals = json.loads(row[0])
            proposal = next((item for item in proposals if item.get("proposal_id") == proposal_id), None)
            if proposal is None:
                raise KeyError(proposal_id)
            current = proposal.get("status")
            if current == status or (current == "accepted" and status == "accepting"):
                return proposal
            allowed = {"pending": {"accepting", "accepted", "rejected"}, "accepting": {"accepted", "pending"}}
            if status not in allowed.get(current, set()):
                raise ValueError(
                    "VLM proposal decision is busy or already reviewed; retry Accept if its outcome is unknown"
                )
            proposal["status"] = status
            connection.execute(
                "UPDATE video_box_rescues SET proposals_payload=?, updated_at=? WHERE job_id=?",
                (json.dumps(proposals, sort_keys=True), datetime.now(UTC).isoformat(), job_id),
            )
        return proposal

    def claim_next_video_sequence_draft(self, *, lease_seconds: float) -> VideoJobClaim | None:
        """Claim one completed sequence's pending CVAT export, including expired claims."""
        if lease_seconds < 3:
            raise ValueError("draft export lease must be at least three seconds")
        now = datetime.now(UTC)
        token = str(uuid4())
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """SELECT job_id FROM video_jobs WHERE status=? AND publish_sha256 IS NULL
                    AND (stage='sequence_draft_ready' OR
                        (stage='sequence_draft_exporting' AND claim_expires_at <= ?))
                    ORDER BY updated_at, job_id LIMIT 1""",
                (VideoJobStatus.AWAITING_REVIEW.value, now.isoformat()),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                """UPDATE video_jobs SET stage='sequence_draft_exporting', updated_at=?,
                    claim_token=?, claim_expires_at=? WHERE job_id=?""",
                (now.isoformat(), token, (now + timedelta(seconds=lease_seconds)).isoformat(), row[0]),
            )
        return VideoJobClaim(self.get_video_job(row[0]), token)

    def fail_video_sequence_draft(self, job_id: str, *, claim_token: str, message: str) -> VideoJob:
        """Retain inference artifacts after a CVAT export error; retry never needs a GPU."""
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE video_jobs SET stage='draft_export_failed', updated_at=?,
                    claim_token=NULL, claim_expires_at=NULL,
                    error_code='sequence_draft_export_failed', error_message=?
                    WHERE job_id=? AND status=? AND stage='sequence_draft_exporting'
                    AND claim_token=?""",
                (
                    datetime.now(UTC).isoformat(),
                    message[:500],
                    job_id,
                    VideoJobStatus.AWAITING_REVIEW.value,
                    claim_token,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("draft export claim is no longer active")
        return self.get_video_job(job_id)

    def queue_video_discovery(self, job_id: str, *, frame_count: int) -> VideoJob:
        if frame_count < 1:
            raise ValueError("video discovery requires at least one frame")
        now = datetime.now(UTC).isoformat()
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE video_jobs SET status=?, stage=?, updated_at=?, progress_current=0,
                    progress_total=?, claim_token=NULL, claim_expires_at=NULL,
                    discovery_sha256=NULL, discovery_payload=NULL,
                    audit_sha256=NULL, audit_payload=NULL, review_queue_sha256=NULL,
                    review_queue_payload=NULL, publish_sha256=NULL, publish_payload=NULL,
                    cvat_annotation_revision=NULL, published_track_count=0,
                    error_code=NULL, error_message=NULL
                    WHERE job_id=? AND status=? AND
                    (error_code LIKE 'discovery_%' OR error_code LIKE 'video_discovery_%')
                    AND track_payload IS NOT NULL""",
                (
                    VideoJobStatus.DISCOVERING.value,
                    "queued_for_discovery",
                    now,
                    frame_count,
                    job_id,
                    VideoJobStatus.FAILED.value,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"video job {job_id} is not ready for discovery retry")
        return self.get_video_job(job_id)

    def claim_next_video_discovery(self, *, lease_seconds: float = 60) -> VideoJobClaim | None:
        if lease_seconds <= 0:
            raise ValueError("video discovery lease must be positive")
        now_value = datetime.now(UTC)
        token = str(uuid4())
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """SELECT job_id FROM video_jobs WHERE status=? AND stage=?
                    ORDER BY updated_at, job_id LIMIT 1""",
                (VideoJobStatus.DISCOVERING.value, "queued_for_discovery"),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                """UPDATE video_jobs SET stage=?, updated_at=?, claim_token=?, claim_expires_at=?,
                    progress_current=0, error_code=NULL, error_message=NULL
                    WHERE job_id=? AND status=? AND stage=?""",
                (
                    "discovering",
                    now_value.isoformat(),
                    token,
                    (now_value + timedelta(seconds=lease_seconds)).isoformat(),
                    row[0],
                    VideoJobStatus.DISCOVERING.value,
                    "queued_for_discovery",
                ),
            )
        return VideoJobClaim(job=self.get_video_job(row[0]), claim_token=token)

    def requeue_expired_video_discovery(self) -> int:
        now = datetime.now(UTC).isoformat()
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE video_jobs SET stage=?, updated_at=?, progress_current=0,
                    claim_token=NULL, claim_expires_at=NULL, error_code=NULL, error_message=NULL
                    WHERE status=? AND stage=? AND claim_expires_at IS NOT NULL AND claim_expires_at <= ?""",
                (
                    "queued_for_discovery",
                    now,
                    VideoJobStatus.DISCOVERING.value,
                    "discovering",
                    now,
                ),
            )
        return cursor.rowcount

    def update_video_discovery_progress(
        self, job_id: str, *, claim_token: str, current: int, total: int, lease_seconds: float = 60
    ) -> None:
        if current < 0 or total < 1 or current > total:
            raise ValueError("video discovery progress must satisfy 0 <= current <= total")
        if lease_seconds <= 0:
            raise ValueError("video discovery lease must be positive")
        now_value = datetime.now(UTC)
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE video_jobs SET updated_at=?, progress_current=?, progress_total=?, claim_expires_at=?
                    WHERE job_id=? AND status=? AND stage=? AND claim_token=? AND progress_current <= ?""",
                (
                    now_value.isoformat(),
                    current,
                    total,
                    (now_value + timedelta(seconds=lease_seconds)).isoformat(),
                    job_id,
                    VideoJobStatus.DISCOVERING.value,
                    "discovering",
                    claim_token,
                    current,
                ),
            )
        if cursor.rowcount != 1:
            raise ValueError(f"video job {job_id} discovery claim is no longer active")

    def complete_video_discovery(
        self,
        job_id: str,
        *,
        claim_token: str,
        track_set_sha256: str,
        track_payload: str,
        discovery_sha256: str,
        discovery_payload: str,
        component_versions: list[ComponentVersion],
        frame_count: int,
        review_queue_sha256: str | None = None,
        review_queue_payload: str | None = None,
    ) -> VideoJob:
        if (review_queue_sha256 is None) != (review_queue_payload is None):
            raise ValueError("review queue hash and payload must be supplied together")
        bypass_audit = review_queue_payload is not None
        now = datetime.now(UTC).isoformat()
        versions = json.dumps([item.model_dump(mode="json") for item in component_versions], sort_keys=True)
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE video_jobs SET status=?, stage=?, updated_at=?, track_set_sha256=?, track_payload=?,
                    discovery_sha256=?, discovery_payload=?, component_versions=?, progress_current=?, progress_total=?,
                    audit_sha256=NULL, audit_payload=NULL, review_queue_sha256=?,
                    review_queue_payload=?, publish_sha256=NULL, publish_payload=NULL,
                    cvat_annotation_revision=NULL, published_track_count=0,
                    claim_token=NULL, claim_expires_at=NULL,
                    error_code=NULL, error_message=NULL
                    WHERE job_id=? AND status=? AND stage=? AND claim_token=?""",
                (
                    VideoJobStatus.AWAITING_REVIEW.value if bypass_audit else VideoJobStatus.AUDITING.value,
                    "awaiting_review" if bypass_audit else "queued_for_audit",
                    now,
                    track_set_sha256,
                    track_payload,
                    discovery_sha256,
                    discovery_payload,
                    versions,
                    frame_count,
                    frame_count,
                    review_queue_sha256,
                    review_queue_payload,
                    job_id,
                    VideoJobStatus.DISCOVERING.value,
                    "discovering",
                    claim_token,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"video job {job_id} is not running discovery")
        return self.get_video_job(job_id)

    def read_video_discovery(self, job_id: str) -> str:
        with self.connect() as connection:
            row = connection.execute("SELECT discovery_payload FROM video_jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        if row[0] is None:
            raise ValueError(f"video job {job_id} does not have discovery evidence")
        return str(row[0])

    def queue_video_audit(self, job_id: str, *, frame_count: int) -> VideoJob:
        if frame_count < 1:
            raise ValueError("video audit requires at least one frame")
        now = datetime.now(UTC).isoformat()
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE video_jobs SET status=?, stage=?, updated_at=?, progress_current=0,
                    progress_total=?, claim_token=NULL, claim_expires_at=NULL,
                    audit_sha256=NULL, audit_payload=NULL, review_queue_sha256=NULL,
                    review_queue_payload=NULL, publish_sha256=NULL, publish_payload=NULL,
                    cvat_annotation_revision=NULL, published_track_count=0,
                    error_code=NULL, error_message=NULL
                    WHERE job_id=? AND status=? AND
                    (error_code LIKE 'audit_%' OR error_code LIKE 'video_audit_%')
                    AND discovery_payload IS NOT NULL AND track_payload IS NOT NULL""",
                (
                    VideoJobStatus.AUDITING.value,
                    "queued_for_audit",
                    now,
                    frame_count,
                    job_id,
                    VideoJobStatus.FAILED.value,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"video job {job_id} is not ready for audit retry")
        return self.get_video_job(job_id)

    def claim_next_video_audit(self, *, lease_seconds: float = 60) -> VideoJobClaim | None:
        if lease_seconds <= 0:
            raise ValueError("video audit lease must be positive")
        now_value = datetime.now(UTC)
        token = str(uuid4())
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """SELECT job_id FROM video_jobs WHERE status=? AND stage=?
                    ORDER BY updated_at, job_id LIMIT 1""",
                (VideoJobStatus.AUDITING.value, "queued_for_audit"),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                """UPDATE video_jobs SET stage=?, updated_at=?, claim_token=?, claim_expires_at=?,
                    progress_current=0, error_code=NULL, error_message=NULL
                    WHERE job_id=? AND status=? AND stage=?""",
                (
                    "auditing",
                    now_value.isoformat(),
                    token,
                    (now_value + timedelta(seconds=lease_seconds)).isoformat(),
                    row[0],
                    VideoJobStatus.AUDITING.value,
                    "queued_for_audit",
                ),
            )
        return VideoJobClaim(job=self.get_video_job(row[0]), claim_token=token)

    def requeue_expired_video_audits(self) -> int:
        now = datetime.now(UTC).isoformat()
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE video_jobs SET stage=?, updated_at=?, progress_current=0,
                    claim_token=NULL, claim_expires_at=NULL, error_code=NULL, error_message=NULL
                    WHERE status=? AND stage=? AND claim_expires_at IS NOT NULL AND claim_expires_at <= ?""",
                (
                    "queued_for_audit",
                    now,
                    VideoJobStatus.AUDITING.value,
                    "auditing",
                    now,
                ),
            )
        return cursor.rowcount

    def update_video_audit_progress(
        self, job_id: str, *, claim_token: str, current: int, total: int, lease_seconds: float = 60
    ) -> None:
        if current < 0 or total < 1 or current > total:
            raise ValueError("video audit progress must satisfy 0 <= current <= total")
        if lease_seconds <= 0:
            raise ValueError("video audit lease must be positive")
        now_value = datetime.now(UTC)
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE video_jobs SET updated_at=?, progress_current=?, progress_total=?, claim_expires_at=?
                    WHERE job_id=? AND status=? AND stage=? AND claim_token=? AND progress_current <= ?""",
                (
                    now_value.isoformat(),
                    current,
                    total,
                    (now_value + timedelta(seconds=lease_seconds)).isoformat(),
                    job_id,
                    VideoJobStatus.AUDITING.value,
                    "auditing",
                    claim_token,
                    current,
                ),
            )
        if cursor.rowcount != 1:
            raise ValueError(f"video job {job_id} audit claim is no longer active")

    def complete_video_audit(
        self,
        job_id: str,
        *,
        claim_token: str,
        audit_sha256: str,
        audit_payload: str,
        review_queue_sha256: str,
        review_queue_payload: str,
        component_versions: list[ComponentVersion],
        reviewed_frame_count: int,
    ) -> VideoJob:
        now = datetime.now(UTC).isoformat()
        versions = json.dumps([item.model_dump(mode="json") for item in component_versions], sort_keys=True)
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE video_jobs SET status=?, stage=?, updated_at=?, audit_sha256=?, audit_payload=?,
                    review_queue_sha256=?, review_queue_payload=?, component_versions=?, progress_current=?,
                    progress_total=?, claim_token=NULL, claim_expires_at=NULL, error_code=NULL, error_message=NULL
                    WHERE job_id=? AND status=? AND stage=? AND claim_token=?""",
                (
                    VideoJobStatus.AWAITING_REVIEW.value,
                    "awaiting_review",
                    now,
                    audit_sha256,
                    audit_payload,
                    review_queue_sha256,
                    review_queue_payload,
                    versions,
                    reviewed_frame_count,
                    reviewed_frame_count,
                    job_id,
                    VideoJobStatus.AUDITING.value,
                    "auditing",
                    claim_token,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"video job {job_id} is not running audit")
        return self.get_video_job(job_id)

    def complete_video_audit_without_vlm(
        self,
        job_id: str,
        *,
        claim_token: str,
        review_queue_sha256: str,
        review_queue_payload: str,
        frame_count: int,
    ) -> VideoJob:
        """Drain an audit claim created before the legacy VLM audit was retired."""
        now = datetime.now(UTC).isoformat()
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE video_jobs SET status=?, stage=?, updated_at=?, audit_sha256=NULL,
                    audit_payload=NULL, review_queue_sha256=?, review_queue_payload=?,
                    progress_current=?, progress_total=?, claim_token=NULL, claim_expires_at=NULL,
                    error_code=NULL, error_message=NULL
                    WHERE job_id=? AND status=? AND stage=? AND claim_token=?""",
                (
                    VideoJobStatus.AWAITING_REVIEW.value,
                    "awaiting_review",
                    now,
                    review_queue_sha256,
                    review_queue_payload,
                    frame_count,
                    frame_count,
                    job_id,
                    VideoJobStatus.AUDITING.value,
                    "auditing",
                    claim_token,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"video job {job_id} is not running audit")
        return self.get_video_job(job_id)

    def read_video_audit(self, job_id: str) -> str:
        return self._read_video_payload(job_id, "audit_payload", "audit result")

    def read_video_review_queue(self, job_id: str) -> str:
        return self._read_video_payload(job_id, "review_queue_payload", "review queue")

    def record_video_publish(
        self,
        job_id: str,
        *,
        cvat_job_id: int,
        publish_sha256: str,
        publish_payload: str,
        cvat_annotation_revision: int,
        published_track_count: int,
    ) -> VideoJob:
        now = datetime.now(UTC).isoformat()
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE video_jobs SET stage=?, updated_at=?, publish_sha256=?, publish_payload=?,
                    cvat_annotation_revision=?, published_track_count=?, error_code=NULL, error_message=NULL,
                    claim_token=NULL, claim_expires_at=NULL
                    WHERE job_id=? AND status=? AND cvat_job_id=?
                    AND track_payload IS NOT NULL AND review_queue_payload IS NOT NULL""",
                (
                    "published_to_cvat",
                    now,
                    publish_sha256,
                    publish_payload,
                    cvat_annotation_revision,
                    published_track_count,
                    job_id,
                    VideoJobStatus.AWAITING_REVIEW.value,
                    cvat_job_id,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"video job {job_id} is not ready for CVAT publishing")
        return self.get_video_job(job_id)

    def read_video_publish_result(self, job_id: str) -> str:
        return self._read_video_payload(job_id, "publish_payload", "CVAT publish result")

    def get_sequence_audit_state(self, job_id: str) -> dict[str, object] | None:
        with self.connect() as connection:
            row = connection.execute(
                """SELECT status, attempts, partial_payload, error, updated_at
                FROM video_sequence_audits WHERE job_id=?""",
                (job_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "status": row[0],
            "attempts": row[1],
            "completed_items": len(json.loads(row[2])),
            "error": row[3],
            "updated_at": row[4],
        }

    def retry_sequence_audit(self, job_id: str) -> None:
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE video_sequence_audits SET status='queued', error=NULL,
                claim_token=NULL, claim_expires_at=NULL, updated_at=?
                WHERE job_id=? AND status='failed'""",
                (datetime.now(UTC).isoformat(), job_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("sequence audit is not failed")

    def claim_next_sequence_audit(self, *, lease_seconds: float) -> VideoJobClaim | None:
        if lease_seconds < 3:
            raise ValueError("sequence audit lease must be at least three seconds")
        now = datetime.now(UTC)
        token = str(uuid4())
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """SELECT a.job_id FROM video_sequence_audits a
                JOIN video_jobs j ON j.job_id=a.job_id
                WHERE j.publish_sha256 IS NOT NULL AND j.status=?
                AND (a.status='queued' OR (a.status='running' AND a.claim_expires_at <= ?))
                ORDER BY a.updated_at, a.job_id LIMIT 1""",
                (VideoJobStatus.AWAITING_REVIEW.value, now.isoformat()),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                """UPDATE video_sequence_audits SET status='running', attempts=attempts+1,
                claim_token=?, claim_expires_at=?, updated_at=? WHERE job_id=?""",
                (token, (now + timedelta(seconds=lease_seconds)).isoformat(), now.isoformat(), row[0]),
            )
        return VideoJobClaim(self.get_video_job(row[0]), token)

    def read_sequence_audit_partial(self, job_id: str, *, claim_token: str) -> list[dict]:
        with self.connect() as connection:
            row = connection.execute(
                """SELECT partial_payload FROM video_sequence_audits
                WHERE job_id=? AND status='running' AND claim_token=?""",
                (job_id, claim_token),
            ).fetchone()
        if row is None:
            raise ValueError("sequence audit claim is no longer active")
        return json.loads(row[0])

    def extend_sequence_audit_lease(self, job_id: str, *, claim_token: str, lease_seconds: float) -> None:
        now = datetime.now(UTC)
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE video_sequence_audits SET claim_expires_at=?
                WHERE job_id=? AND status='running' AND claim_token=?""",
                ((now + timedelta(seconds=lease_seconds)).isoformat(), job_id, claim_token),
            )
            if cursor.rowcount != 1:
                raise ValueError("sequence audit claim is no longer active")

    def save_sequence_audit_partial(
        self, job_id: str, *, claim_token: str, evidence: list[dict], lease_seconds: float
    ) -> None:
        now = datetime.now(UTC)
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE video_sequence_audits SET partial_payload=?, updated_at=?,
                claim_expires_at=? WHERE job_id=? AND status='running' AND claim_token=?""",
                (
                    json.dumps(evidence, sort_keys=True),
                    now.isoformat(),
                    (now + timedelta(seconds=lease_seconds)).isoformat(),
                    job_id,
                    claim_token,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("sequence audit claim is no longer active")

    def complete_sequence_audit(
        self,
        job_id: str,
        *,
        claim_token: str,
        previous_review_sha256: str,
        audit_payload: str,
        audit_sha256: str,
        review_payload: str,
        review_sha256: str,
    ) -> VideoJob:
        now = datetime.now(UTC).isoformat()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """UPDATE video_jobs SET audit_payload=?, audit_sha256=?,
                review_queue_payload=?, review_queue_sha256=?, updated_at=?
                WHERE job_id=? AND status=? AND publish_sha256 IS NOT NULL
                AND review_queue_sha256=? AND EXISTS (
                    SELECT 1 FROM video_sequence_audits WHERE job_id=?
                    AND status='running' AND claim_token=?)""",
                (
                    audit_payload,
                    audit_sha256,
                    review_payload,
                    review_sha256,
                    now,
                    job_id,
                    VideoJobStatus.AWAITING_REVIEW.value,
                    previous_review_sha256,
                    job_id,
                    claim_token,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("sequence audit cannot replace a changed review queue")
            connection.execute(
                """UPDATE video_sequence_audits SET status='done', error=NULL,
                claim_token=NULL, claim_expires_at=NULL, updated_at=? WHERE job_id=?""",
                (now, job_id),
            )
        return self.get_video_job(job_id)

    def fail_sequence_audit(self, job_id: str, *, claim_token: str, error: str) -> None:
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE video_sequence_audits SET status='failed', error=?,
                claim_token=NULL, claim_expires_at=NULL, updated_at=?
                WHERE job_id=? AND status='running' AND claim_token=?""",
                (error[:500], datetime.now(UTC).isoformat(), job_id, claim_token),
            )
            if cursor.rowcount != 1:
                raise ValueError("sequence audit claim is no longer active")

    def _read_video_payload(self, job_id: str, column: str, description: str) -> str:
        if column not in {"audit_payload", "review_queue_payload", "publish_payload"}:
            raise ValueError("unsupported video payload column")
        with self.connect() as connection:
            row = connection.execute(f"SELECT {column} FROM video_jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        if row[0] is None:
            raise ValueError(f"video job {job_id} does not have a {description}")
        return str(row[0])

    def bind_cvat_task(self, job_id: str, *, task_id: int, cvat_job_id: int, manifest_sha256: str) -> VideoJob:
        now = datetime.now(UTC).isoformat()
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE video_jobs SET updated_at=?, cvat_task_id=?, cvat_job_id=?
                    WHERE job_id=? AND status IN (?, ?) AND manifest_sha256=? AND cvat_task_id IS NULL""",
                (
                    now,
                    task_id,
                    cvat_job_id,
                    job_id,
                    VideoJobStatus.AWAITING_SEED.value,
                    VideoJobStatus.SEED_READY.value,
                    manifest_sha256,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"video job {job_id} cannot bind the CVAT task")
        return self.get_video_job(job_id)

    def cancel_video_job(self, job_id: str) -> VideoJob:
        now = datetime.now(UTC).isoformat()
        terminal = (
            VideoJobStatus.COMPLETED.value,
            VideoJobStatus.FAILED.value,
            VideoJobStatus.CANCELLED.value,
        )
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE video_jobs SET status=?, stage=?, updated_at=?, claim_token=NULL, claim_expires_at=NULL
                    WHERE job_id=? AND status NOT IN (?, ?, ?)""",
                (VideoJobStatus.CANCELLED.value, "cancelled", now, job_id, *terminal),
            )
            if cursor.rowcount != 1:
                exists = connection.execute("SELECT 1 FROM video_jobs WHERE job_id=?", (job_id,)).fetchone()
                if exists is None:
                    raise KeyError(job_id)
                raise ValueError(f"video job {job_id} is already terminal")
            connection.execute(
                """UPDATE video_chunks SET status='cancelled', claim_token=NULL,
                   claim_expires_at=NULL, updated_at=?
                   WHERE job_id=? AND status IN ('pending', 'running', 'failed')""",
                (now, job_id),
            )
        return self.get_video_job(job_id)

    def fail_video_job(self, job_id: str, *, code: str, message: str, claim_token: str | None = None) -> VideoJob:
        now = datetime.now(UTC).isoformat()
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE video_jobs SET status=?, stage=?, updated_at=?, error_code=?, error_message=?,
                    claim_token=NULL, claim_expires_at=NULL
                    WHERE job_id=? AND status NOT IN (?, ?)
                    AND (? IS NULL OR claim_token=?)""",
                (
                    VideoJobStatus.FAILED.value,
                    "failed",
                    now,
                    code[:64],
                    message[:500],
                    job_id,
                    VideoJobStatus.COMPLETED.value,
                    VideoJobStatus.CANCELLED.value,
                    claim_token,
                    claim_token,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"video job {job_id} cannot be failed")
        return self.get_video_job(job_id)


def _job_values(job: VideoJob) -> tuple[object, ...]:
    return (
        job.job_id,
        job.status.value,
        job.source_name,
        job.source_sha256,
        job.source_size_bytes,
        job.created_at.isoformat(),
        job.updated_at.isoformat(),
        job.stage,
        job.progress_current,
        job.progress_total,
        job.manifest_sha256,
        job.cvat_job_id,
        job.seed_revision,
        job.seed_count,
        job.seed_sha256,
        job.cvat_task_id,
        json.dumps([item.model_dump(mode="json") for item in job.component_versions], sort_keys=True),
        job.track_set_sha256,
        job.discovery_sha256,
        job.audit_sha256,
        job.review_queue_sha256,
        job.publish_sha256,
        job.cvat_annotation_revision,
        job.published_track_count,
        job.error_code,
        job.error_message,
    )


def _job_from_row(row: tuple[object, ...]) -> VideoJob:
    return VideoJob(
        job_id=row[0],
        status=row[1],
        source_name=row[2],
        source_sha256=row[3],
        source_size_bytes=row[4],
        created_at=row[5],
        updated_at=row[6],
        stage=row[7],
        progress_current=row[8],
        progress_total=row[9],
        manifest_sha256=row[10],
        cvat_job_id=row[11],
        seed_revision=row[12],
        seed_count=row[13],
        seed_sha256=row[14],
        cvat_task_id=row[15],
        component_versions=json.loads(str(row[16])),
        track_set_sha256=row[17],
        discovery_sha256=row[18],
        audit_sha256=row[19],
        review_queue_sha256=row[20],
        publish_sha256=row[21],
        cvat_annotation_revision=row[22],
        published_track_count=row[23],
        error_code=row[24],
        error_message=row[25],
    )


def _ensure_video_job_columns(connection: sqlite3.Connection) -> None:
    columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(video_jobs)").fetchall()}
    migrations = {
        "cvat_task_id": "ALTER TABLE video_jobs ADD COLUMN cvat_task_id INTEGER",
        "component_versions": "ALTER TABLE video_jobs ADD COLUMN component_versions TEXT NOT NULL DEFAULT '[]'",
        "seed_payload": "ALTER TABLE video_jobs ADD COLUMN seed_payload TEXT",
        "claim_token": "ALTER TABLE video_jobs ADD COLUMN claim_token TEXT",
        "claim_expires_at": "ALTER TABLE video_jobs ADD COLUMN claim_expires_at TEXT",
        "track_set_sha256": "ALTER TABLE video_jobs ADD COLUMN track_set_sha256 TEXT",
        "track_payload": "ALTER TABLE video_jobs ADD COLUMN track_payload TEXT",
        "discovery_sha256": "ALTER TABLE video_jobs ADD COLUMN discovery_sha256 TEXT",
        "discovery_payload": "ALTER TABLE video_jobs ADD COLUMN discovery_payload TEXT",
        "audit_sha256": "ALTER TABLE video_jobs ADD COLUMN audit_sha256 TEXT",
        "audit_payload": "ALTER TABLE video_jobs ADD COLUMN audit_payload TEXT",
        "review_queue_sha256": "ALTER TABLE video_jobs ADD COLUMN review_queue_sha256 TEXT",
        "review_queue_payload": "ALTER TABLE video_jobs ADD COLUMN review_queue_payload TEXT",
        "publish_sha256": "ALTER TABLE video_jobs ADD COLUMN publish_sha256 TEXT",
        "publish_payload": "ALTER TABLE video_jobs ADD COLUMN publish_payload TEXT",
        "cvat_annotation_revision": "ALTER TABLE video_jobs ADD COLUMN cvat_annotation_revision INTEGER",
        "published_track_count": "ALTER TABLE video_jobs ADD COLUMN published_track_count INTEGER NOT NULL DEFAULT 0",
    }
    for column, statement in migrations.items():
        if column not in columns:
            try:
                connection.execute(statement)
            except sqlite3.OperationalError as error:
                if "duplicate column name" not in str(error).lower():
                    raise


def _recover_legacy_unleased_extractions(connection: sqlite3.Connection) -> None:
    """A pre-lease worker cannot still own work while this migration transaction runs."""
    now = datetime.now(UTC).isoformat()
    connection.execute(
        """UPDATE video_jobs SET status=?, stage=?, updated_at=?, progress_current=0, progress_total=0
            WHERE status=? AND claim_token IS NULL""",
        (
            VideoJobStatus.QUEUED.value,
            "requeued_after_lease_migration",
            now,
            VideoJobStatus.EXTRACTING.value,
        ),
    )
