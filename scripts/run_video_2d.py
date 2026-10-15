"""Run one entire image sequence through SAM3 text prompts and export review drafts.

This command is offline: it neither calls CVAT nor publishes training labels.
Give the true cadence of the source image sequence with --source-fps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from PIL import Image

from inference.sam3_sequence import Sam3SequenceBackend
from src.models.video import CvatSamplingRequest, VideoJob, VideoJobStatus
from src.services.storage.repository import Store
from src.services.video.chunk_processor import ChunkBackend, ChunkProcessor
from src.services.video.chunks import ChunkStore
from src.services.video.cvat_draft import build_sequence_cvat_draft
from src.services.video.export_2d import export_2d
from src.services.video.extractor import _histogram_distance
from src.services.video.sampling import choose_keyframes
from src.services.video.sequence_review import score_tracks
from src.services.video.stitching import StitchConfig

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Sequence2DResult:
    job_id: str
    source_frames: list[int]
    coco: Path
    cvat_xml: Path
    stitched: Path


def _images(directory: Path) -> list[Path]:
    files = sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not files or [path.stem for path in files] != [f"{index:06d}" for index in range(len(files))]:
        raise ValueError("sequence images must be contiguous 000000.png/jpg, 000001..., from frame zero")
    return files


def _source_hash(files: list[Path], *, source_fps: float, target_fps: float) -> str:
    digest = hashlib.sha256(json.dumps({"source_fps": source_fps, "target_fps": target_fps}, sort_keys=True).encode())
    for path in files:
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
    return digest.hexdigest()


def run_sequence(
    image_dir: Path,
    output_dir: Path,
    *,
    source_fps: float,
    target_fps: float | None = None,
    labels: tuple[str, ...] = ("car", "person"),
    checkpoint: Path | None = None,
    backend: ChunkBackend | None = None,
    window_size: int = 50,
    overlap: int = 5,
    stitch_decision_log_enabled: bool = True,
    cvat_labels: Path | None = None,
    cvat_annotations: Path | None = None,
) -> Sequence2DResult:
    if (cvat_labels is None) != (cvat_annotations is None):
        raise ValueError("CVAT labels and existing annotations must be supplied together")
    files = _images(image_dir)
    strategy = "all_frames" if target_fps is None else "uniform"
    target_fps = source_fps if target_fps is None else target_fps
    selection = CvatSamplingRequest(strategy=strategy, source_fps=source_fps, target_fps=target_fps)
    indexes, _ = choose_keyframes([0.0] * len(files), selection)
    digest = _source_hash(files, source_fps=source_fps, target_fps=target_fps)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "sequence-state.json"
    store = Store(output_dir / "sequence.sqlite3")
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("source_sha256") != digest:
            raise ValueError("sequence inputs changed; choose a new output directory")
        job_id = state["job_id"]
        store.get_video_job(job_id)
    else:
        job_id = str(uuid4())
        now = datetime.now(UTC)
        store.create_video_job(
            VideoJob(
                job_id=job_id,
                status=VideoJobStatus.QUEUED,
                source_name=image_dir.name,
                source_sha256=digest,
                source_size_bytes=sum(path.stat().st_size for path in files),
                created_at=now,
                updated_at=now,
                stage="queued_for_sequence",
            )
        )
        state_path.write_text(json.dumps({"job_id": job_id, "source_sha256": digest}), encoding="utf-8")
    staged = output_dir / "frames"
    staged.mkdir(exist_ok=True)
    frame_paths = {}
    width = height = None
    previous_histogram = None
    scene_cuts = []
    for local, source in enumerate(indexes):
        target = staged / f"frame_{local:06d}.jpg"
        with Image.open(files[source]) as opened:
            histogram = opened.convert("L").resize((64, 36)).histogram()
            distance = _histogram_distance(previous_histogram, histogram)
            if strategy == "all_frames" and distance is not None and distance >= selection.scene_cut_threshold:
                scene_cuts.append(local)
            previous_histogram = histogram
            image = opened.convert("RGB")
            if width is None:
                width, height = image.size
            elif image.size != (width, height):
                raise ValueError("source images have inconsistent dimensions")
            if not target.is_file():
                image.save(target, format="JPEG", quality=95)
        frame_paths[local] = target
    if backend is None:
        if checkpoint is None:
            raise ValueError("supply a private SAM3 checkpoint for actual inference")
        pins = tomllib.loads((ROOT / "kaggle/runtime-versions.toml").read_text(encoding="utf-8"))
        backend = Sam3SequenceBackend(
            frame_paths=frame_paths,
            checkpoint=checkpoint,
            checkpoint_sha256=pins["sam3_checkpoint_sha256"],
            sam3_commit=pins["sam3_git_commit"],
        )
    processor = ChunkProcessor(
        store=ChunkStore(store),
        artifact_root=output_dir / "chunks",
        backend=backend,
        window_size=window_size,
        overlap=overlap,
        stitch=StitchConfig(decision_log_enabled=stitch_decision_log_enabled),
    )
    try:
        stitched = processor.run(job_id, frame_count=len(indexes), input_sha256=digest, labels=labels,
                                 scene_cut_frames=tuple(scene_cuts))
    finally:
        coverage_path = output_dir / "chunks" / job_id / "frame-processing.json"
        if coverage_path.is_file():
            coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
            for row in coverage["frames"]:
                row["source_frame_index"] = indexes[row["frame_index"]]
            (output_dir / "frame-processing.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    review = score_tracks(stitched.tracks)
    (output_dir / "review-queue.json").write_text(
        json.dumps(
            {
                "schema": "autolabel-sequence-review/v1",
                "job_id": job_id,
                "calibrated": False,
                "tracks": [asdict(item) for item in review],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    artifacts = export_2d(
        stitched.tracks,
        source_frames=indexes,
        image_names=[path.name for path in files],
        width=width or 0,
        height=height or 0,
        output_dir=output_dir,
        review={item.track_id: item for item in review},
    )
    if cvat_labels is not None and cvat_annotations is not None:
        label_payload = json.loads(cvat_labels.read_text(encoding="utf-8"))
        task_labels = label_payload.get("labels") if isinstance(label_payload, dict) else label_payload
        if not isinstance(task_labels, list):
            raise ValueError("CVAT label snapshot must contain a labels list")
        frame_hashes = [hashlib.sha256(frame_paths[index].read_bytes()).hexdigest() for index in range(len(indexes))]
        draft = build_sequence_cvat_draft(
            job_id=job_id,
            source_sha256=digest,
            source_name=image_dir.name,
            source_frames=indexes,
            frame_sha256=frame_hashes,
            width=width or 0,
            height=height or 0,
            source_fps=source_fps,
            labels=task_labels,
            text_prompts=labels,
            tracks=stitched.tracks,
            review=review,
            existing_annotations=json.loads(cvat_annotations.read_text(encoding="utf-8")),
            model_version=backend.version,
        )
        (output_dir / "cvat-draft.json").write_text(
            json.dumps(draft, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    manifest = {
        "schema": "autolabel-sequence-2d/v1",
        "job_id": job_id,
        "source_sha256": digest,
        "source_fps": source_fps,
        "fps_source": "operator",
        "target_fps": target_fps,
        "sampling_strategy": strategy,
        "source_frame_count": len(files),
        "selected_source_frames": indexes,
        "scene_cut_frames": scene_cuts,
        "model_version": backend.version,
        "annotation_provenance": ("SAM3 observes every source frame; no temporal downsampling" if strategy == "all_frames"
                                  else "Legacy SAM3 sampled frames; geometric interpolation between observations"),
        "review_status": "pending",
    }
    (output_dir / "sequence-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if store.get_video_job(job_id).status is VideoJobStatus.QUEUED:
        store.transition_video_job(
            job_id, expected={VideoJobStatus.QUEUED}, status=VideoJobStatus.AWAITING_REVIEW, stage="draft_export_ready"
        )
    return Sequence2DResult(
        job_id, indexes, artifacts.coco, artifacts.cvat_xml, output_dir / "chunks" / job_id / "stitched.json"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images", type=Path, required=True, help="One KITTI-style numbered image_02/<sequence> directory"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-fps", type=float, required=True, help="Actual frame cadence after any CVAT/import filtering"
    )
    parser.add_argument("--target-fps", type=float, help="Legacy sampled runs only; omit to process every source frame")
    parser.add_argument("--classes", default="car,person")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--overlap", type=int, default=5)
    parser.add_argument("--stitch-decision-log", action=argparse.BooleanOptionalAction, default=True,
                        help="Log stitching evidence only; use --no-stitch-decision-log to disable")
    parser.add_argument("--cvat-labels", type=Path, help="Read-only CVAT task label snapshot; no network request")
    parser.add_argument(
        "--cvat-annotations", type=Path, help="Read-only existing annotation snapshot; no network request"
    )
    args = parser.parse_args()
    result = run_sequence(
        args.images,
        args.output,
        source_fps=args.source_fps,
        target_fps=args.target_fps,
        labels=tuple(item.strip() for item in args.classes.split(",")),
        checkpoint=args.checkpoint,
        window_size=args.window_size,
        overlap=args.overlap,
        stitch_decision_log_enabled=args.stitch_decision_log,
        cvat_labels=args.cvat_labels,
        cvat_annotations=args.cvat_annotations,
    )
    print(
        json.dumps(
            {
                "job_id": result.job_id,
                "sampled_frames": len(result.source_frames),
                "coco": str(result.coco),
                "cvat_xml": str(result.cvat_xml),
                "stitched": str(result.stitched),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
