from __future__ import annotations

import asyncio

import httpx
import pytest

from src.services.video.publish_draft import (
    SequenceDraftConflictError,
    prepare_sequence_append,
    push_sequence_draft,
)


def _fixtures():
    original = {"version": 3, "tags": [], "shapes": [], "tracks": [{"id": 10, "source": "manual", "shapes": []}]}
    proposed = {
        "source": "auto",
        "label_id": 7,
        "shapes": [{"frame": 0, "attributes": [{"spec_id": 20, "value": "pending"}]}],
    }
    draft = {**original, "tracks": [*original["tracks"], proposed]}
    labels = [{"id": 7, "attributes": [{"id": 20, "name": "review_status", "mutable": True}]}]
    return original, draft, labels


def test_prepare_append_preserves_manual_track_and_enforces_review() -> None:
    current, draft, labels = _fixtures()
    payload = prepare_sequence_append(draft, current, labels)
    assert len(payload["tracks"]) == 1
    assert payload["tracks"][0]["source"] == "auto"
    assert payload["version"] == 3
    current["version"] += 1
    with pytest.raises(SequenceDraftConflictError, match="changed"):
        prepare_sequence_append(draft, current, labels)
    current["version"] -= 1
    labels[0]["attributes"][0]["mutable"] = False
    with pytest.raises(SequenceDraftConflictError, match="mutable"):
        prepare_sequence_append(draft, current, labels)


def test_live_writer_protocol_adds_only_pending_tracks() -> None:
    current, draft, labels = _fixtures()
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/labels":
            return httpx.Response(200, json={"results": labels, "next": None})
        if request.method == "PATCH":
            body = __import__("json").loads(request.content)
            seen.append(body)
            current["tracks"].extend(body["tracks"])
            return httpx.Response(200, json={"version": 3})
        return httpx.Response(200, json=current)

    result = asyncio.run(
        push_sequence_draft(
            base_url="http://cvat.test",
            token="Bearer placeholder",
            cvat_job_id=5,
            draft=draft,
            transport=httpx.MockTransport(handler),
        )
    )
    assert result == {"job_id": 5, "tracks_before": 1, "tracks_added": 1, "tracks_after": 2}
    assert seen[0]["tracks"] == [draft["tracks"][1]]
    assert seen[0]["shapes"] == []
