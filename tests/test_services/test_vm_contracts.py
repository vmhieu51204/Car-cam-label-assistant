from pathlib import Path

import pytest

from src.config import Settings
from src.services.vm.contracts import VmProfile, file_sha256, preflight


def test_preflight_blocks_unconfigured_without_network():
    with pytest.raises(ValueError, match="enabled"):
        preflight(Settings(_env_file=None))


def test_profile_immutable_pins_and_redacted_input(tmp_path: Path):
    weights = tmp_path / "sam3.1_multiplex.pt"
    weights.write_bytes(b"fixture checkpoint")
    settings = Settings(
        _env_file=None,
        vm_workflow_enabled=True,
        local_sam_enabled=True,
        local_sam_python=weights,
        privacy_python=weights,
        privacy_face_checkpoint=weights,
        privacy_plate_checkpoint=weights,
        privacy_face_sha256=file_sha256(weights),
        privacy_plate_sha256=file_sha256(weights),
        privacy_source_commit="a" * 40,
        local_sam_commit="b" * 40,
        local_sam31_checkpoint=weights,
        local_sam31_sha256=file_sha256(weights),
    )
    profile = preflight(settings)
    assert (profile.window, profile.overlap, profile.input_kind) == (48, 4, "redacted")
    assert VmProfile.model_validate_json(profile.model_dump_json()) == profile
    weights.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        preflight(settings)


def test_additive_vm_tables_preserve_legacy_job_and_schema_version(tmp_path):
    from datetime import UTC, datetime

    from src.models.video import VideoJob, VideoJobStatus
    from src.services.storage.repository import Store
    from src.services.vm.storage import VmStore

    db = tmp_path / "legacy.sqlite3"
    store = Store(db)
    now = datetime.now(UTC)
    original = VideoJob(
        job_id="00000000-0000-4000-8000-000000000001",
        status=VideoJobStatus.AWAITING_REVIEW,
        source_name="legacy.mp4",
        source_sha256="a" * 64,
        source_size_bytes=1,
        created_at=now,
        updated_at=now,
        stage="awaiting_review",
        cvat_job_id=1,
    )
    store.create_video_job(original)
    with store.connect() as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
    VmStore(db, tmp_path / "vm")
    assert store.get_video_job(original.job_id) == original
    with store.connect() as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == version
