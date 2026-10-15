from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image

from src.models.video import (
    AuditEvidence,
    AuditResult,
    BoundingBox2D,
    ComponentVersion,
    DiscoveryEvent,
    DiscoveryResult,
    FrameManifest,
    FrameRecord,
    TrackObservation,
    TrackSet,
    VideoTrack,
)
from src.services.video.audit import (
    PROMPT_VERSION,
    prepare_audit_items,
)
from src.services.video.review import build_review_queue
from src.services.video.workspace import VideoWorkspace

JOB_ID = "66666666-6666-4666-8666-666666666666"


def _workspace_manifest(workspace: VideoWorkspace) -> tuple[FrameManifest, bytes]:
    frame_dir = workspace.frames_dir(JOB_ID)
    frame_dir.mkdir(parents=True)
    hashes = []
    for index in range(3):
        path = frame_dir / f"frame_{index:06d}.jpg"
        Image.new("RGB", (100, 80), color=(20 + index * 20, 30, 40)).save(path, format="JPEG")
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    manifest = FrameManifest(
        job_id=JOB_ID,
        source_name="clip.mp4",
        source_sha256="a" * 64,
        codec="h264",
        width=100,
        height=80,
        duration_s=1.5,
        average_fps=2,
        variable_frame_rate=False,
        ffmpeg_version="ffmpeg test",
        frames=[
            FrameRecord(
                frame_index=index,
                timestamp_s=index * 0.5,
                source_timestamp_s=index * 0.5,
                relative_path=f"frames/frame_{index:06d}.jpg",
                sha256=digest,
                width=100,
                height=80,
            )
            for index, digest in enumerate(hashes)
        ],
    )
    return manifest, workspace.write_manifest(manifest)


def _tracks(manifest_sha: str, seed_sha: str) -> TrackSet:
    return TrackSet(
        job_id=JOB_ID,
        manifest_sha256=manifest_sha,
        seed_sha256=seed_sha,
        component_versions=[ComponentVersion(component="sam2-video", version="sam2-test")],
        tracks=[
            VideoTrack(
                track_id="shape:5",
                label_id=1,
                label_name="car",
                source="human_seed",
                observations=[
                    TrackObservation(
                        frame_index=index,
                        bbox=BoundingBox2D(xtl=10 + index, ytl=10, xbr=30 + index, ybr=30),
                        score=1 if index == 0 else 0.9,
                    )
                    for index in range(3)
                ],
            ),
            VideoTrack(
                track_id="auto:missing:000001",
                label_id=1,
                label_name="car",
                source="auto",
                observations=[
                    TrackObservation(
                        frame_index=index,
                        bbox=BoundingBox2D(xtl=60 + index, ytl=20, xbr=80 + index, ybr=40),
                        score=0.88,
                    )
                    for index in (1, 2)
                ],
            ),
        ],
    )


def _discovery(manifest_sha: str, propagated_sha: str) -> DiscoveryResult:
    proposal = _tracks(manifest_sha, "c" * 64).tracks[1]
    return DiscoveryResult(
        job_id=JOB_ID,
        manifest_sha256=manifest_sha,
        propagated_track_sha256=propagated_sha,
        component_versions=[ComponentVersion(component="rtdetr-object-detector", version="rtdetr-test")],
        proposal_tracks=[proposal],
        events=[
            DiscoveryEvent(
                event_id="missing:auto:missing:000001",
                event_type="possible_missing_object",
                frame_index=1,
                track_ids=["auto:missing:000001"],
                label_name="car",
                bbox=proposal.observations[0].bbox,
                score=0.88,
                reason="independent detector track does not match a propagated seed",
            )
        ],
    )


def test_audit_selection_flags_gap_and_scene_cut_on_both_sides(tmp_path: Path) -> None:
    workspace = VideoWorkspace(tmp_path / "video-jobs")
    manifest, encoded = _workspace_manifest(workspace)
    manifest.frames[2].scene_cut_before = True
    tracks = _tracks(hashlib.sha256(encoded).hexdigest(), "c" * 64)
    tracks.tracks[0].observations.pop(1)
    discovery = _discovery(tracks.manifest_sha256, "d" * 64)
    items = prepare_audit_items(
        manifest=manifest, tracks=tracks, discovery=discovery, workspace=workspace, max_frames=3
    )
    by_frame = {item.frame_index: set(item.selection_reasons) for item in items}
    assert "track_gap" in by_frame[1]
    assert "track_reappears" in by_frame[2]
    assert "scene_cut_before" in by_frame[1]
    assert "scene_cut_after" in by_frame[2]


def test_audit_selection_renders_deterministic_track_overlays(tmp_path: Path):
    workspace = VideoWorkspace(tmp_path / "video-jobs")
    manifest, manifest_bytes = _workspace_manifest(workspace)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    tracks = _tracks(manifest_sha, "c" * 64)
    discovery = _discovery(manifest_sha, "d" * 64)

    first = prepare_audit_items(
        manifest=manifest,
        tracks=tracks,
        discovery=discovery,
        workspace=workspace,
        periodic_stride=30,
    )
    first_hashes = [item.overlay_sha256 for item in first]
    second = prepare_audit_items(
        manifest=manifest,
        tracks=tracks,
        discovery=discovery,
        workspace=workspace,
        periodic_stride=30,
    )

    assert [item.frame_index for item in first] == [0, 1, 2]
    assert "possible_missing_object" in first[1].selection_reasons
    assert first[1].track_ids == ("auto:missing:000001", "shape:5")
    assert first_hashes == [item.overlay_sha256 for item in second]


def test_audit_budget_covers_whole_video_even_when_events_cluster_at_start(tmp_path: Path):
    workspace = VideoWorkspace(tmp_path / "video-jobs")
    frame_dir = workspace.frames_dir(JOB_ID)
    frame_dir.mkdir(parents=True)
    frames = []
    for index in range(120):
        path = frame_dir / f"frame_{index:06d}.jpg"
        Image.new("RGB", (100, 80), color=(20, 30, 40)).save(path, format="JPEG")
        frames.append(
            FrameRecord(
                frame_index=index,
                timestamp_s=index * 0.5,
                source_timestamp_s=index * 0.5,
                relative_path=f"frames/frame_{index:06d}.jpg",
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                width=100,
                height=80,
                scene_change_score=0.8 if index == 73 else 0.01,
            )
        )
    manifest = FrameManifest(
        job_id=JOB_ID,
        source_name="clip.mp4",
        source_sha256="a" * 64,
        codec="h264",
        width=100,
        height=80,
        duration_s=60,
        average_fps=2,
        variable_frame_rate=False,
        ffmpeg_version="test",
        frames=frames,
    )
    items = prepare_audit_items(
        manifest=manifest,
        tracks=_tracks("b" * 64, "c" * 64),
        discovery=_discovery("b" * 64, "d" * 64),
        workspace=workspace,
        max_frames=12,
    )
    indexes = [item.frame_index for item in items]
    assert len(indexes) == 12
    assert indexes[0] == 0 and indexes[-1] == 119
    assert 73 in indexes
    assert sum(index >= 60 for index in indexes) >= 5
    capped = prepare_audit_items(
        manifest=manifest,
        tracks=_tracks("b" * 64, "c" * 64),
        discovery=_discovery("b" * 64, "d" * 64),
        workspace=workspace,
        max_frames=1000,
    )
    assert len(capped) == 50


def test_vlm_agreement_does_not_lower_detector_conflict():
    tracks = _tracks("b" * 64, "c" * 64)
    discovery = _discovery("b" * 64, "d" * 64)
    evidence = AuditEvidence(
        job_id=JOB_ID,
        frame_index=1,
        raw_frame_sha256="e" * 64,
        overlay_sha256="f" * 64,
        track_ids=["auto:missing:000001"],
        verdict="agree",
        reason="The proposal covers the later car.",
        component_version=ComponentVersion(component="video-vlm-auditor", version="vlm-test"),
    )
    audit = AuditResult(
        job_id=JOB_ID,
        manifest_sha256="b" * 64,
        track_set_sha256="1" * 64,
        discovery_sha256="2" * 64,
        prompt_version=PROMPT_VERSION,
        component_versions=[evidence.component_version],
        evidence=[evidence],
    )

    queue = build_review_queue(
        tracks=tracks,
        discovery=discovery,
        audit=audit,
        audit_sha256="3" * 64,
    )

    assert len(queue.units) == 1
    assert queue.units[0].priority == "critical"
    assert queue.units[0].review_status == "pending"
    assert queue.units[0].reason_codes == ["possible_missing_object", "vlm_agree"]
