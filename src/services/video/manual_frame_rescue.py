"""User-triggered Claude suggestions on any source frame, separate from batch audits."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from src.models.video import BoundingBox2D
from src.services.cvat import CvatWriteError
from src.services.video.claude_box_rescue import ClaudeBoxRescueAuditor
from src.services.vm.storage import atomic_json, worker_lock

LOCKS: dict[tuple[str, int], asyncio.Lock] = {}


def rectangles_at_frame(annotations: dict, frame: int) -> list[dict]:
    boxes = []
    for shape in annotations.get("shapes", []):
        if shape.get("type") == "rectangle" and shape.get("frame") == frame and not shape.get("outside"):
            boxes.append({"label_id": shape["label_id"], "bbox": shape["points"]})
    for track in annotations.get("tracks", []):
        shapes = sorted((s for s in track.get("shapes", []) if s.get("type") == "rectangle"), key=lambda s: s["frame"])
        previous = next((s for s in reversed(shapes) if s["frame"] <= frame), None)
        if previous is None or previous.get("outside"):
            continue
        points = previous["points"]
        following = next((s for s in shapes if s["frame"] > frame), None)
        if following is not None:
            weight = (frame - previous["frame"]) / (following["frame"] - previous["frame"])
            points = [a + weight * (b - a) for a, b in zip(points, following["points"], strict=True)]
        boxes.append({"label_id": track["label_id"], "bbox": points})
    return boxes


def _save(path: Path, result: dict) -> None:
    atomic_json(path, result)


async def propose_frame(service: Any, settings: Any, job_id: str, frame: int, *, auditor: Any = None) -> dict:
    if not service.box_rescue_enabled:
        raise ValueError("Claude is not configured")
    job = service.store.get_video_job(job_id)
    if job.cvat_job_id is None:
        raise ValueError("This job is not linked to CVAT")
    async with LOCKS.setdefault((job_id, frame), asyncio.Lock()):
        client = service.seed_reader.client
        cvat_job = await client.get_job(job.cvat_job_id)
        if cvat_job.get("dimension", "2d") != "2d":
            raise ValueError("Frame suggestions require a 2D CVAT job")
        if not cvat_job.get("start_frame", 0) <= frame <= cvat_job["stop_frame"]:
            raise ValueError("Source frame is outside the CVAT job")
        annotations = await client.get_guarded_job_annotations(job.cvat_job_id)
        labels = await client.get_job_labels(job.cvat_job_id)
        names = {r["id"]: r["name"] for r in labels}
        if not names:
            raise ValueError("Create CVAT class labels before requesting suggestions")
        existing = [
            {"label": names[b["label_id"]], "bbox": b["bbox"]}
            for b in rectangles_at_frame(annotations, frame)
            if b["label_id"] in names
        ]
        encoded = [data async for data in client.iter_selected_job_frames(job.cvat_job_id, [frame])]
        image = Image.open(io.BytesIO(encoded[0])).convert("RGB")
        width, height = image.size
        draw = ImageDraw.Draw(image)
        for row in existing:
            draw.rectangle(row["bbox"], outline="blue", width=3)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        overlay = buffer.getvalue()
        digest = hashlib.sha256(overlay).hexdigest()
        auditor = auditor or ClaudeBoxRescueAuditor(
            api_key=settings.openrouter_api_key.get_secret_value(),
            cache_path=settings.vlm_cache_path,
            model=settings.vlm_box_rescue_model,
            fewshot_manifest=settings.vlm_box_rescue_fewshot_manifest,
            timeout_s=settings.vlm_timeout_s,
            retry_attempts=settings.vlm_box_rescue_retry_attempts,
            retry_base_delay_s=settings.vlm_box_rescue_retry_base_delay_s,
            retry_max_delay_s=settings.vlm_box_rescue_retry_max_delay_s,
        )
        identity = hashlib.sha256(
            json.dumps(
                [
                    job_id,
                    frame,
                    digest,
                    auditor.model_identity,
                    sorted(names.values()),
                    annotations["autolabel_revision"],
                ],
                sort_keys=True,
            ).encode()
        ).hexdigest()
        directory = service.workspace.job_dir(job_id) / "manual-claude"
        directory.mkdir(exist_ok=True)
        result_path = directory / f"{identity}.json"
        if result_path.exists():
            return json.loads(result_path.read_text(encoding="utf-8"))
        overlay_path = directory / f"{identity}.png"
        overlay_path.write_bytes(overlay)
        answer, usage, cached = await asyncio.to_thread(
            auditor.propose_missing_boxes,
            overlay_path=overlay_path,
            overlay_sha256=digest,
            classes=tuple(names.values()),
            image_width=width,
            image_height=height,
            existing_boxes=existing,
        )
        proposals = [dict(p.model_dump(mode="json"), status="pending") for p in answer.proposals]
        for index, proposal in enumerate(proposals):
            x1, y1, x2, y2 = proposal["bbox"]
            draw.rectangle([x1 * width, y1 * height, x2 * width, y2 * height], outline="red", width=3)
            draw.text((x1 * width, y1 * height), f"{index + 1}: {proposal['label']}", fill="red")
        preview = io.BytesIO()
        image.save(preview, format="PNG")
        result = dict(
            result_id=identity,
            source_frame=frame,
            width=width,
            height=height,
            revision=annotations["autolabel_revision"],
            proposals=proposals,
            reason=answer.reason,
            model=auditor.model,
            usage=usage.model_dump(mode="json"),
            cache_hit=cached,
            preview="data:image/png;base64," + base64.b64encode(preview.getvalue()).decode(),
        )
        with worker_lock(directory / f"{identity}.lock"):
            if result_path.exists():
                return json.loads(result_path.read_text(encoding="utf-8"))
            _save(result_path, result)
        return result


async def decide_frame(service: Any, job_id: str, result_id: str, index: int, accept: bool) -> dict:
    if not re.fullmatch("[0-9a-f]{64}", result_id):
        raise ValueError("Invalid result ID")
    job = service.store.get_video_job(job_id)
    path = service.workspace.job_dir(job_id) / "manual-claude" / f"{result_id}.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    async with LOCKS.setdefault((job_id, result["source_frame"]), asyncio.Lock()):
        with worker_lock(path.with_suffix(".lock")):
            return await _decide_locked(service, job, job_id, result_id, index, accept, path)


async def _decide_locked(
    service: Any, job: Any, job_id: str, result_id: str, index: int, accept: bool, path: Path
) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not 0 <= index < len(result["proposals"]):
        raise ValueError("Proposal index outside result")
    proposal = result["proposals"][index]
    if proposal["status"] == "accepting" and not accept:
        raise ValueError("Accept outcome is uncertain; Retry Accept before Reject")
    if proposal["status"] not in ("pending", "accepting"):
        return result
    if accept:
        labels = await service.prepare_sequence_labels(job.cvat_job_id, (proposal["label"],))
        label = next((r for r in labels if r["name"] == proposal["label"]), None)
        if label is None:
            raise ValueError("Proposal class is no longer in CVAT")
        x1, y1, x2, y2 = proposal["bbox"]
        uncertain = proposal["status"] == "accepting"
        proposal["status"] = "accepting"
        _save(path, result)
        try:
            revision = await service.track_client.accept_rescue_track(
                cvat_job_id=job.cvat_job_id,
                frame=result["source_frame"],
                label_id=label["id"],
                bbox=BoundingBox2D(
                    xtl=x1 * result["width"],
                    ytl=y1 * result["height"],
                    xbr=x2 * result["width"],
                    ybr=y2 * result["height"],
                ),
                expected_revision=result["revision"],
                proposal_id=f"manual:{result_id}:{index}",
                occluded=proposal["occlusion"] != "none",
                box_extent=proposal["box_extent"],
            )
        except CvatWriteError as error:
            if error.status_code == 409 and not uncertain:
                proposal["status"] = "pending"
                _save(path, result)
            raise
        result["revision"] = revision
    proposal["status"] = "accepted" if accept else "rejected"
    _save(path, result)
    return result


# Context around the drawn box on each side, as a fraction of its size, so a box drawn too
# small still shows the whole object; the floor keeps tiny boxes recognizable.
REFINE_CONTEXT = 1.0
REFINE_MIN_CONTEXT_PX = 48


async def refine_frame_box(
    service: Any, job_id: str, frame: int, label: str, bbox: list[float], instruction: str, *, auditor: Any
) -> dict:
    """Fit an annotator's box to its object on the redacted frame; never writes CVAT."""
    job = service.store.get_video_job(job_id)
    if job.cvat_job_id is None:
        raise ValueError("This job is not linked to CVAT")
    client = service.seed_reader.client
    cvat_job = await client.get_job(job.cvat_job_id)
    if cvat_job.get("dimension", "2d") != "2d":
        raise ValueError("Box refinement requires a 2D CVAT job")
    if not cvat_job.get("start_frame", 0) <= frame <= cvat_job["stop_frame"]:
        raise ValueError("Source frame is outside the CVAT job")
    labels = await client.get_job_labels(job.cvat_job_id)
    if label not in {row["name"] for row in labels}:
        raise ValueError("Class is not in the CVAT job")
    encoded = [data async for data in client.iter_selected_job_frames(job.cvat_job_id, [frame])]
    image = Image.open(io.BytesIO(encoded[0])).convert("RGB")
    width, height = image.size
    x1, y1, x2, y2 = bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height
    pad_x = max((x2 - x1) * REFINE_CONTEXT, REFINE_MIN_CONTEXT_PX)
    pad_y = max((y2 - y1) * REFINE_CONTEXT, REFINE_MIN_CONTEXT_PX)
    left, top = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
    right, bottom = min(width, int(x2 + pad_x + 1)), min(height, int(y2 + pad_y + 1))
    crop = image.crop((left, top, right, bottom))
    ImageDraw.Draw(crop).rectangle([x1 - left, y1 - top, x2 - left, y2 - top], outline="blue", width=2)
    buffer = io.BytesIO()
    crop.save(buffer, format="PNG")
    digest = hashlib.sha256(buffer.getvalue()).hexdigest()
    directory = service.workspace.job_dir(job_id) / "manual-claude" / "refine"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{digest}.png"
    if not path.exists():
        path.write_bytes(buffer.getvalue())
    answer, usage, cached = await asyncio.to_thread(
        auditor.refine_box, crop_path=path, crop_sha256=digest, label=label, instruction=instruction
    )
    refined = None
    if answer["found"]:
        a, b, c, d = answer["bbox"]
        crop_width, crop_height = crop.size
        refined = [
            (left + a * crop_width) / width,
            (top + b * crop_height) / height,
            (left + c * crop_width) / width,
            (top + d * crop_height) / height,
        ]
    return {
        "found": answer["found"],
        "bbox": refined,
        "reason": answer["reason"],
        "source_frame": frame,
        "width": width,
        "height": height,
        "model": auditor.model,
        "cache_hit": cached,
        "usage": usage.model_dump(mode="json"),
    }
