"""Offline CVAT panel harness using the retained SAM3 result and review contracts.

Processed SAM3 videos are replayed from the local demo library when available.
The synthetic fixture remains available for tests. Playback loads no weights or external services. Use run_backend.py for the original CVAT/Kaggle/local GPU integration.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from threading import RLock
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, ConfigDict, Field

from src.models.video import BoundingBox2D, CvatSamplingRequest, FrameManifest, TrackSet
from demo_library import DemoLibrary
from src.services.evaluation.matching import MatchInput, greedy_matches
from src.services.video.chunks import plan_chunks
from src.services.video.sequence_kaggle import make_track_set, parse_sequence_result
from src.services.video.sequence_review import build_sequence_review_queue, SequenceReviewConfig
from src.services.video.track_triage import track_triage

ROOT = Path(__file__).resolve().parent
COUNT = 24
WIDTH, HEIGHT = 960, 540
LABELS = [{"id": 1, "name": "car"}, {"id": 2, "name": "person"}]


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def truth(frame):
    boxes = [("car", [100 + frame * 12, 270, 255 + frame * 12, 370])]
    if frame >= 8:
        boxes.append(("person", [720 - frame * 4, 230, 748 - frame * 4, 330]))
    return boxes


class Reevaluate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    iou_threshold: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)


class BoxEdit(BaseModel):
    frame: int = Field(ge=0)
    bbox: BoundingBox2D


class ReviewState(BaseModel):
    track_id: str
    state: str = Field(pattern=r"^(unseen|seen|resolved)$")


class Session:
    def __init__(self, library=None):
        self.library = library
        self.clip = None
        self.lock = RLock()
        self.job = None
        self.manifest = None
        self.tracks = None
        self.states = {}
        self.threshold = 0.5
        self.history = []

    def load_clip(self, video_id):
        entry, manifest, tracks = self.library.load(video_id)
        self.clip, self.manifest, self.tracks = entry, manifest, tracks
        self.states, self.history = {}, []
        self.job = {"job_id": manifest.job_id, "status": "awaiting_review", "stage": "saved-sam3-review",
            "cvat_task_id": 1, "cvat_job_id": 1, "manifest_sha256": tracks.manifest_sha256,
            "track_set_sha256": sha(tracks.model_dump(mode="json")), "publish_sha256": None,
            "published_track_count": 0, "progress_current": manifest.frame_count, "progress_total": manifest.frame_count,
            "component_versions": [c.model_dump() for c in tracks.component_versions], "data_source": "sam3-kaggle"}
        if hasattr(self, "workspace"):
            self.workspace.restore()
        self.evaluate(self.threshold)
        return self.job

    def require(self, job_id=None):
        if self.job is None or (job_id is not None and job_id != self.job["job_id"]):
            raise HTTPException(404, "No local run for this job")

    def run(self, request):
        self.clip = None
        if request.sam3_model_variant != "sam3":
            raise HTTPException(422, "The synthetic harness uses the SAM3 contract. GPU SAM3.1 remains in the original backend.")
        classes = request.sequence_classes
        if any(c not in {row["name"] for row in LABELS} for c in classes):
            raise HTTPException(422, "Synthetic labels are car and person")
        job_id = str(uuid4())
        frames = [{"frame_index": f, "source_frame_index": f, "timestamp_s": f / request.source_fps,
                   "source_timestamp_s": f / request.source_fps, "relative_path": f"frames/frame_{f:06d}.jpg",
                   "sha256": sha(["synthetic-frame", f]), "width": WIDTH, "height": HEIGHT} for f in range(COUNT)]
        self.manifest = FrameManifest(job_id=job_id, source_name="synthetic-street",
            source_sha256=sha(frames), source_fps=request.source_fps, target_fps=request.source_fps,
            source_frame_count=COUNT, codec="synthetic", width=WIDTH, height=HEIGHT,
            duration_s=COUNT / request.source_fps, average_fps=request.source_fps, variable_frame_rate=False,
            sampling_strategy="all_frames", ffmpeg_version="synthetic/no-decoder", frames=frames)
        config = {"window_size": 12, "overlap": 3, "model_variant": "sam3", "scene_cut_frames": []}
        predictions = []
        for window in plan_chunks(COUNT, window_size=12, overlap=3):
            for f in range(window.start, window.stop):
                for label, box in truth(f):
                    if label not in classes or (label == "person" and f in {12, 13}):
                        continue
                    if label == "car" and f == 9:
                        box = [box[0] + 180, box[1], box[2] + 180, box[3]]
                    predictions.append({"window_id": window.index, "track_id": label, "label": label,
                                        "frame_index": f, "bbox": box, "score": 0.85})
        result = {"schema": "autolabel-sam3-benchmark/v1", "model_runtime": {
            "environment": {"versions": {"sam3_commit": "synthetic-mock"}}}, "configurations": [{
                "id": "sequence-text", "config": config, "status": "completed",
                "selected_source_frames": list(range(COUNT)), "evaluated_source_frames": list(range(COUNT)),
                "window_predictions": predictions}]}
        stitched, component = parse_sequence_result(result, expected_config=config, frame_count=COUNT)
        self.tracks = make_track_set(manifest=self.manifest, manifest_sha256=sha(frames),
                                    stitched=stitched, labels=LABELS, classes=tuple(classes), component=component)
        self.states = {}
        self.history = []
        self.job = {"job_id": job_id, "status": "awaiting_review", "stage": "synthetic-review",
                    "cvat_task_id": 1, "cvat_job_id": 1, "manifest_sha256": sha(frames),
                    "track_set_sha256": sha(self.tracks.model_dump(mode="json")), "publish_sha256": None,
                    "published_track_count": 0, "progress_current": COUNT, "progress_total": COUNT,
                    "component_versions": [component.model_dump()], "data_source": "synthetic-mock"}
        if hasattr(self, "workspace"):
            self.workspace.restore()
        self.evaluate(self.threshold)
        return self.job

    def triage(self):
        self.require()
        rows = track_triage(self.tracks, self.manifest, SequenceReviewConfig())
        for row in rows:
            row["state"] = self.states.get(row["track_id"], "unseen")
        return {"tracks": rows, "total": len(rows), "seen": sum(r["state"] != "unseen" for r in rows)}

    def evaluate(self, threshold):
        self.require()
        self.threshold = threshold
        if self.clip:
            rows = self.triage()["tracks"]
            report = {"iteration": len(self.history)+1, "data_source": "sam3-kaggle", "reference_available": False,
                "iou_threshold": threshold, "tp": None, "fp": None, "fn": None,
                "precision": None, "recall": None, "f1": None,
                "track_count": len(rows), "flagged_tracks": sum(bool(r["reasons"]) for r in rows),
                "observation_count": sum(not o.outside for t in self.tracks.tracks for o in t.observations),
                "resolved_tracks": sum(r["state"] == "resolved" for r in rows)}
            self.history.append(report)
            self.job["track_set_sha256"] = sha(self.tracks.model_dump(mode="json"))
            return report
        gt = tuple(MatchInput(f, label, tuple(box), i) for f in range(COUNT)
                   for i, (label, box) in enumerate(truth(f)))
        pred = tuple(MatchInput(o.frame_index, t.label_name,
                     (o.bbox.xtl, o.bbox.ytl, o.bbox.xbr, o.bbox.ybr), i)
                     for i, t in enumerate(self.tracks.tracks) if t.review_status != "rejected"
                     for o in t.observations if not o.outside)
        pairs = greedy_matches(gt, pred, threshold=threshold, class_aware=True)
        tp = len(pairs); fp = len(pred) - tp; fn = len(gt) - tp
        report = {"iteration": len(self.history) + 1, "data_source": "synthetic-mock",
                  "iou_threshold": threshold, "tp": tp, "fp": fp, "fn": fn,
                  "precision": tp / len(pred) if pred else None, "recall": tp / len(gt) if gt else None,
                  "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}
        self.history.append(report)
        self.job["track_set_sha256"] = sha(self.tracks.model_dump(mode="json"))
        return report

    def track(self, track_id):
        self.require()
        for track in self.tracks.tracks:
            if track.track_id == track_id:
                return track
        raise HTTPException(404, "Unknown track")


def create_mock_app(demo_root=None, *, autoload=True, state_root=None, repair_runner=None):
    app = FastAPI(title="Car cam label assistant â€” local mock")
    library = DemoLibrary(demo_root or ROOT / "demo_results")
    session = Session(library)
    if autoload and library.catalog().get("default_video_id"):
        session.load_clip(library.catalog()["default_video_id"])
    app.state.session = session
    base = "/autolabel/api/v1/video/jobs"

    @app.get("/", include_in_schema=False)
    def index():
        return RedirectResponse("/tasks/1/jobs/1")

    @app.get("/mock/tasks/{task_id}/jobs/{job_id}", include_in_schema=False)
    def fallback_page(task_id: int, job_id: int):
        return FileResponse(ROOT / "src/web/mock.html")

    @app.get("/tasks/{task_id}/jobs/{job_id}", include_in_schema=False)
    def page(task_id: int, job_id: int):
        if task_id != 1 or job_id != 1:
            raise HTTPException(404, "The local harness has task 1 / job 1")
        if (ROOT / "ui/dist/index.html").is_file():
            return FileResponse(ROOT / "ui/dist/index.html")
        return FileResponse(ROOT / "src/web/mock.html")

    @app.get("/mock.js", include_in_schema=False)
    def script():
        return FileResponse(ROOT / "src/web/mock.js", media_type="text/javascript")

    @app.get("/autolabel-panel.js", include_in_schema=False)
    def panel():
        return FileResponse(ROOT / "integrations/cvat-autolabel/files/cvat-ui/autolabel-panel.js", media_type="text/javascript")

    @app.get("/mock/videos")
    def videos():
        return library.catalog()

    @app.post("/mock/videos/{video_id}/select")
    def select_video(video_id: str):
        with session.lock:
            return session.load_clip(video_id)

    @app.post("/mock/synthetic")
    def synthetic():
        with session.lock:
            return session.run(CvatSamplingRequest(source_fps=12, auto_start_sequence=True))

    @app.get("/mock/media/{video_id}/video")
    def result_video(video_id: str):
        return FileResponse(library.file(video_id, "annotated.mp4"), media_type="video/mp4")

    @app.get("/mock/media/{video_id}/poster")
    def poster(video_id: str):
        return FileResponse(library.file(video_id, "poster.jpg"), media_type="image/jpeg")

    @app.get("/mock/media/{video_id}/frames/{frame}")
    def frame_image(video_id: str, frame: int):
        entry = library.entry(video_id)
        if not 0 <= frame < entry["frame_count"]:
            raise HTTPException(404, "Frame outside demo video")
        return FileResponse(library.file(video_id, f"frames/frame_{frame:06d}.jpg"), media_type="image/jpeg")

    @app.get("/healthz")
    def health():
        return {"status": "ok", "mode": "sam3-replay" if session.clip else "synthetic-mock", "cloud_required": False, "weights_loaded": False}

    @app.get("/autolabel/api/v1/vm/capabilities")
    def vm_capabilities():
        return {"enabled": False}

    @app.get(base + "/cvat-jobs/{cvat_id}/labels")
    def labels(cvat_id: int):
        return sorted({t.label_name for t in session.tracks.tracks}) if session.clip else [row["name"] for row in LABELS]

    @app.get(base + "/cvat-jobs/{cvat_id}/capabilities")
    def capabilities(cvat_id: int):
        return {"track_review": True, "docked_panel": True, "track_filter": False,
                "vlm_box_rescue": False, "local_sam3": False, "local_sam31": False, "mock": True}

    @app.get(base + "/cvat-jobs/{cvat_id}")
    def current_job(cvat_id: int):
        with session.lock:
            session.require()
            return session.job

    @app.post(base + "/cvat-jobs/{cvat_id}/import")
    def start(cvat_id: int, payload: CvatSamplingRequest):
        with session.lock:
            return session.load_clip(session.clip["id"]) if session.clip else session.run(payload)

    @app.get(base + "/{job_id}/manifest")
    def manifest(job_id: str):
        with session.lock:
            session.require(job_id)
            return session.manifest

    @app.get(base + "/{job_id}/review-queue")
    def queue(job_id: str):
        with session.lock:
            session.require(job_id)
            return build_sequence_review_queue(tracks=session.tracks, manifest=session.manifest,
                track_set_sha256=session.job["track_set_sha256"], discovery_sha256=sha([]))

    @app.get(base + "/{job_id}/review")
    def review(job_id: str):
        with session.lock:
            session.require(job_id)
            return session.triage()

    @app.post(base + "/{job_id}/review/state")
    def set_state(job_id: str, payload: ReviewState):
        with session.lock:
            session.require(job_id)
            session.track(payload.track_id)
            session.states[payload.track_id] = payload.state
            return session.triage()

    @app.get(base + "/{job_id}/box-rescue")
    def rescue(job_id: str):
        session.require(job_id)
        return {"status": "disabled", "proposals": []}

    @app.get(base + "/{job_id}/manual-claude/latest")
    def latest(job_id: str):
        session.require(job_id)
        return {"result": None}

    @app.post(base + "/{job_id}/manual-claude")
    def claude(job_id: str):
        session.require(job_id)
        raise HTTPException(503, "Claude is disabled in the offline harness. Use the original backend for VLM review.")

    @app.get("/mock/state")
    def state():
        with session.lock:
            return {"job": session.job, "tracks": session.tracks, "history": session.history,
                    "width": session.manifest.width if session.clip else WIDTH,
                    "height": session.manifest.height if session.clip else HEIGHT,
                    "frame_count": session.manifest.frame_count if session.clip else COUNT,
                    "fps": session.manifest.source_fps if session.clip else 12,
                    "data_source": "sam3-kaggle" if session.clip else "synthetic-mock",
                    "videos": library.catalog()["videos"], "active_video": session.clip,
                    "ground_truth": None if session.clip else [truth(f) for f in range(COUNT)], "review": session.triage() if session.job else None,
                    "revision": session.workspace.revision, "can_undo": bool(session.workspace.undo_stack),
                    "labels": session.workspace.labels()}

    @app.post("/mock/run")
    def run():
        with session.lock:
            return session.load_clip(session.clip["id"]) if session.clip else session.run(CvatSamplingRequest(source_fps=12, auto_start_sequence=True))

    @app.post("/mock/reevaluate")
    def reevaluate(payload: Reevaluate):
        with session.lock:
            return session.evaluate(payload.iou_threshold)

    @app.post("/mock/tracks/{track_id}/correct")
    def correct(track_id: str):
        with session.lock:
            track = session.track(track_id)
            if session.clip:
                raise HTTPException(409, "These videos have no reference labels. Edit boxes manually instead.")
            # Intentionally use fixture truth only for this explicitly labeled mock correction.
            from src.models.video import TrackObservation
            track.observations = [TrackObservation(frame_index=f, bbox=BoundingBox2D(
                xtl=box[0], ytl=box[1], xbr=box[2], ybr=box[3]), score=1.0)
                for f in range(COUNT) for label, box in truth(f) if label == track.label_name]
            track.needs_review = False
            track.review_reasons = []
            session.states[track_id] = "resolved"
            return session.evaluate(session.threshold)

    @app.post("/mock/tracks/{track_id}/box")
    def edit(track_id: str, payload: BoxEdit):
        with session.lock:
            track = session.track(track_id)
            if payload.frame >= session.manifest.frame_count:
                raise HTTPException(422, "Frame outside current video")
            if payload.bbox.xbr > session.manifest.width or payload.bbox.ybr > session.manifest.height:
                raise HTTPException(422, "Box must fit within the current frame")
            from src.models.video import TrackObservation
            obs = TrackObservation(frame_index=payload.frame, bbox=payload.bbox)
            track.observations = sorted([o for o in track.observations if o.frame_index != payload.frame] + [obs], key=lambda o: o.frame_index)
            return session.workspace.changed()["report"]

    @app.get("/mock/export")
    def export():
        with session.lock:
            session.require()
            from fastapi.responses import JSONResponse
            body = {"version": 0, "tags": [], "shapes": [], "tracks": [{
                "label_id": t.label_id, "frame": t.observations[0].frame_index, "group": 0, "source": "auto",
                "attributes": [], "shapes": [{"type": "rectangle", "frame": o.frame_index,
                    "points": [o.bbox.xtl, o.bbox.ytl, o.bbox.xbr, o.bbox.ybr], "outside": o.outside,
                    "occluded": o.occluded, "rotation": 0, "z_order": 0, "attributes": []}
                    for o in t.observations]} for t in session.tracks.tracks if t.review_status != "rejected"]}
            return JSONResponse(body, headers={"Content-Disposition": 'attachment; filename="cvat-annotations.json"'})

    from guided_workspace import attach_workspace
    attach_workspace(app, session, ROOT, state_root=state_root or (Path(demo_root) / "review-state" if demo_root else None), runner=repair_runner)
    return app


app = create_mock_app()
