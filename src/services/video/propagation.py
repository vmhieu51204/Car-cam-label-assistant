"""SAM2 propagation boundary and strict response validation."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from src.models.video import (
    FrameManifest,
    PropagationFrameInput,
    PropagationModelRequest,
    SeedSet,
    TrackObservation,
    TrackSet,
    VideoTrack,
)
from src.services.video.batch_runner import VideoBatchError, VideoBatchRunner
from src.services.video.workspace import VideoWorkspace


class VideoPropagationError(RuntimeError):
    """An actionable model-service failure safe to expose on a video job."""


class VideoPropagator(Protocol):
    def propagate(
        self,
        *,
        manifest: FrameManifest,
        manifest_sha256: str,
        seed_set: SeedSet,
        seed_sha256: str,
    ) -> TrackSet: ...


class KaggleSam3Propagator:
    def __init__(
        self,
        *,
        runner: VideoBatchRunner,
        workspace: VideoWorkspace,
        model_id: str = "facebook/sam3",
        code_file: Path,
    ) -> None:
        self.runner = runner
        self.workspace = workspace
        self.model_id = model_id
        self.code_file = code_file

    def propagate(
        self,
        *,
        manifest: FrameManifest,
        manifest_sha256: str,
        seed_set: SeedSet,
        seed_sha256: str,
    ) -> TrackSet:
        request = _build_request(manifest, manifest_sha256, seed_set, seed_sha256)
        job_dir = self.workspace.job_dir(manifest.job_id)
        files: dict[str, Path | bytes] = {
            "request.json": request.model_dump_json(by_alias=True).encode("utf-8"),
        }
        for frame in request.frames:
            path = (job_dir / frame.relative_path).resolve()
            if not path.is_relative_to(job_dir):
                raise VideoPropagationError("frame path escaped the video workspace")
            files[frame.relative_path] = path
        try:
            result = self.runner.run(stage="sam3", job_id=manifest.job_id, files=files, code_file=self.code_file)
            track_set = TrackSet.model_validate(result["track_set"])
        except (VideoBatchError, KeyError, ValueError) as error:
            raise VideoPropagationError(f"SAM3 propagation failed: {error}") from error
        return _validate_and_preserve_seeds(
            track_set,
            request,
            expected_model_id=self.model_id,
            expected_component="sam3-video",
        )


def _build_request(
    manifest: FrameManifest,
    manifest_sha256: str,
    seed_set: SeedSet,
    seed_sha256: str,
) -> PropagationModelRequest:
    return PropagationModelRequest(
        job_id=manifest.job_id,
        manifest_sha256=manifest_sha256,
        seed_sha256=seed_sha256,
        end_frame_index=_propagation_end(manifest),
        frames=[
            PropagationFrameInput(
                frame_index=frame.frame_index,
                relative_path=frame.relative_path,
                sha256=frame.sha256,
                width=frame.width,
                height=frame.height,
            )
            for frame in manifest.frames
        ],
        seed_set=seed_set,
    )


def _propagation_end(manifest: FrameManifest) -> int:
    for frame in manifest.frames[1:]:
        if frame.scene_cut_before:
            return frame.frame_index - 1
    return manifest.frame_count - 1


def _validate_and_preserve_seeds(
    track_set: TrackSet,
    request: PropagationModelRequest,
    *,
    expected_model_id: str,
    expected_component: str = "sam2-video",
) -> TrackSet:
    if (
        track_set.job_id != request.job_id
        or track_set.manifest_sha256 != request.manifest_sha256
        or track_set.seed_sha256 != request.seed_sha256
    ):
        raise VideoPropagationError("SAM2 response provenance does not match the request")
    if (
        len(track_set.component_versions) != 1
        or track_set.component_versions[0].component != expected_component
        or track_set.component_versions[0].version != expected_model_id
    ):
        raise VideoPropagationError("SAM2 response has an invalid component version")
    expected = {seed.object_id: seed for seed in request.seed_set.seeds}
    received = {track.track_id: track for track in track_set.tracks}
    if received.keys() != expected.keys():
        raise VideoPropagationError("SAM2 response does not contain exactly the seeded objects")
    tracks: list[VideoTrack] = []
    for track_id, seed in expected.items():
        track = received[track_id]
        if track.label_id != seed.label_id or track.label_name != seed.label_name:
            raise VideoPropagationError(f"SAM2 response changed the label for seed {track_id}")
        observations = {item.frame_index: item for item in track.observations}
        if any(index > request.end_frame_index or index >= len(request.frames) for index in observations):
            raise VideoPropagationError("SAM2 response crossed an unseeded scene cut")
        for index, observation in observations.items():
            frame = request.frames[index]
            if observation.bbox.xbr > frame.width or observation.bbox.ybr > frame.height:
                raise VideoPropagationError(f"SAM2 response box exceeds frame {index} bounds")
        observations[0] = TrackObservation(
            frame_index=0,
            bbox=seed.bbox,
            score=1,
            occluded=seed.occluded,
        )
        tracks.append(
            track.model_copy(
                update={
                    "source": "auto" if seed.source == "auto_seed" else "human_seed",
                    "review_status": "pending",
                    "observations": [observations[index] for index in sorted(observations)],
                }
            )
        )
    return track_set.model_copy(update={"tracks": tracks})
