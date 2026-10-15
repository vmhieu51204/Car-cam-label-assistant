"""Append a reviewed-as-pending sequence draft to a CVAT job without erasing work."""

from __future__ import annotations

from typing import Any

import httpx


class SequenceDraftConflictError(ValueError):
    """The draft is stale, unsafe, or lacks mandatory review metadata."""


def prepare_sequence_append(
    draft: dict[str, Any], current: dict[str, Any], labels: list[dict[str, Any]]
) -> dict[str, Any]:
    """Build a CVAT PATCH-create body only for new, pending automatic tracks.

    The full snapshot comparison prevents a stale offline draft from silently
    replacing or duplicating manual work. CVAT also receives its version field
    so concurrent changes after this check must be rejected by the server.
    """
    if not isinstance(draft, dict) or not isinstance(current, dict):
        raise SequenceDraftConflictError("draft and current CVAT annotations must be objects")
    version = current.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 0:
        raise SequenceDraftConflictError("CVAT returned an invalid annotation version")
    if draft.get("version") != version:
        raise SequenceDraftConflictError("CVAT annotations changed since the draft was created")
    for name in ("tags", "shapes", "tracks"):
        if not isinstance(current.get(name), list) or not isinstance(draft.get(name), list):
            raise SequenceDraftConflictError(f"CVAT {name} must be lists")
    if draft["tags"] != current["tags"] or draft["shapes"] != current["shapes"]:
        raise SequenceDraftConflictError("CVAT tags or shapes changed since the draft")
    existing = current["tracks"]
    if draft["tracks"][: len(existing)] != existing:
        raise SequenceDraftConflictError("CVAT tracks changed since the draft")
    generated = draft["tracks"][len(existing) :]
    if not generated or any(track.get("source") != "auto" for track in generated):
        raise SequenceDraftConflictError("draft has no new automatic tracks")
    label_review_ids = {}
    for label in labels:
        status = next(
            (
                attribute
                for attribute in label.get("attributes", [])
                if attribute.get("name") == "review_status" and attribute.get("mutable")
            ),
            None,
        )
        if status is not None:
            label_review_ids[label["id"]] = status["id"]
    for track in generated:
        review_id = label_review_ids.get(track.get("label_id"))
        shapes = track.get("shapes")
        if review_id is None or not isinstance(shapes, list) or not shapes:
            raise SequenceDraftConflictError("CVAT label lacks a mutable review_status attribute")
        if any(
            not any(
                attribute.get("spec_id") == review_id and attribute.get("value") == "pending"
                for attribute in shape.get("attributes", [])
            )
            for shape in shapes
        ):
            raise SequenceDraftConflictError("every automatic keyframe must be pending review")
    return {"version": version, "tags": [], "shapes": [], "tracks": generated}


async def push_sequence_draft(
    *,
    base_url: str,
    token: str,
    cvat_job_id: int,
    draft: dict[str, Any],
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, int]:
    """Read current CVAT state, append only new tracks, and verify the round trip."""
    if cvat_job_id < 1 or not token:
        raise ValueError("a positive CVAT job ID and PAT are required")
    auth = token if token.lower().startswith(("token ", "bearer ")) else f"Token {token}"
    headers = {"Authorization": auth, "Accept": "application/vnd.cvat+json"}
    async with httpx.AsyncClient(
        base_url=base_url.rstrip("/"), headers=headers, timeout=30, transport=transport
    ) as client:
        annotation_path = f"/api/jobs/{cvat_job_id}/annotations"
        before_response = await client.get(annotation_path)
        before_response.raise_for_status()
        labels_response = await client.get("/api/labels", params={"job_id": cvat_job_id, "page_size": 100})
        labels_response.raise_for_status()
        label_page = labels_response.json()
        if label_page.get("next"):
            raise SequenceDraftConflictError("CVAT label list has more than 100 entries")
        current = before_response.json()
        body = prepare_sequence_append(draft, current, label_page["results"])
        response = await client.patch(annotation_path, params={"action": "create"}, json=body)
        response.raise_for_status()
        after_response = await client.get(annotation_path)
        after_response.raise_for_status()
        after = after_response.json()
        before_count = len(current["tracks"])
        added = len(body["tracks"])
        if len(after.get("tracks", [])) != before_count + added:
            raise RuntimeError("CVAT write returned success but track count did not match")
        return {
            "job_id": cvat_job_id,
            "tracks_before": before_count,
            "tracks_added": added,
            "tracks_after": len(after["tracks"]),
        }
