import io
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from src.config import Settings
from src.models.video import SequenceAuditUsage
from src.services.cvat import CvatWriteError
from src.services.video import manual_frame_rescue as module
from src.services.video.rescue_matrix import RescueFrameVerdict, RescueProposal


def test_current_rectangle_track_interpolation_and_outside():
    annotations = {
        "tracks": [
            {
                "label_id": 1,
                "shapes": [
                    {"type": "rectangle", "frame": 0, "points": [0, 0, 10, 10]},
                    {"type": "rectangle", "frame": 10, "points": [10, 0, 20, 10]},
                    {"type": "rectangle", "frame": 11, "points": [10, 0, 20, 10], "outside": True},
                ],
            }
        ]
    }
    assert module.rectangles_at_frame(annotations, 5)[0]["bbox"] == [5, 0, 15, 10]
    assert module.rectangles_at_frame(annotations, 11) == []


@pytest.mark.asyncio
async def test_manual_frame_without_sampled_image_preview_then_accept_and_reject(tmp_path, monkeypatch):
    module.LOCKS.clear()
    buffer = io.BytesIO()
    Image.new("RGB", (100, 80), "white").save(buffer, format="PNG")

    class Client:
        get_job = AsyncMock(return_value={"start_frame": 0, "stop_frame": 99})
        get_job_labels = AsyncMock(return_value=[{"id": 1, "name": "car"}])
        get_guarded_job_annotations = AsyncMock(return_value={"autolabel_revision": 5, "tracks": [], "shapes": []})

        async def iter_selected_job_frames(self, job_id, frames):
            assert frames == [7]
            yield buffer.getvalue()

    calls = []

    class Auditor:
        model = "anthropic/claude-sonnet-5.5"
        model_identity = "fixture"

        def __init__(self, **kwargs):
            pass

        def propose_missing_boxes(self, **kwargs):
            calls.append(kwargs)
            p = RescueProposal(
                label="car", bbox=(0.1, 0.2, 0.8, 0.9), confidence="high", occlusion="partial", reason="Missing vehicle"
            )
            return (
                RescueFrameVerdict(missing="yes", proposals=[p, p], reason="Two suggestions"),
                SequenceAuditUsage(),
                False,
            )

    monkeypatch.setattr(module, "ClaudeBoxRescueAuditor", Auditor)
    service = SimpleNamespace(
        box_rescue_enabled=True,
        store=SimpleNamespace(get_video_job=lambda _: SimpleNamespace(cvat_job_id=17)),
        seed_reader=SimpleNamespace(client=Client()),
        workspace=SimpleNamespace(job_dir=lambda _: tmp_path),
        prepare_sequence_labels=AsyncMock(return_value=[{"id": 1, "name": "car"}]),
        track_client=SimpleNamespace(accept_rescue_track=AsyncMock(return_value=6)),
    )
    result = await module.propose_frame(service, Settings(_env_file=None), "old-job", 7)
    assert result["source_frame"] == 7 and result["preview"].startswith("data:image/png;base64,")
    service.track_client.accept_rescue_track.assert_not_called()
    assert await module.propose_frame(service, Settings(_env_file=None), "old-job", 7) == result
    assert len(calls) == 1
    service.track_client.accept_rescue_track.side_effect = CvatWriteError("revision changed", status_code=409)
    with pytest.raises(CvatWriteError):
        await module.decide_frame(service, "old-job", result["result_id"], 0, True)
    saved = json.loads((tmp_path / "manual-claude" / f"{result['result_id']}.json").read_text())
    assert saved["proposals"][0]["status"] == "pending"
    service.track_client.accept_rescue_track.side_effect = None
    service.track_client.accept_rescue_track.reset_mock()
    result = await module.decide_frame(service, "old-job", result["result_id"], 0, True)
    written = service.track_client.accept_rescue_track.call_args.kwargs
    assert written["frame"] == 7 and written["expected_revision"] == 5
    assert written["bbox"].xtl == 10 and written["bbox"].ytl == 16
    await module.decide_frame(service, "old-job", result["result_id"], 0, True)
    result = await module.decide_frame(service, "old-job", result["result_id"], 1, False)
    assert [p["status"] for p in result["proposals"]] == ["accepted", "rejected"]
    assert service.track_client.accept_rescue_track.await_count == 1
    with pytest.raises(ValueError, match="outside"):
        await module.propose_frame(service, Settings(_env_file=None), "old-job", 100)


@pytest.mark.asyncio
async def test_refine_frame_box_maps_crop_back_to_frame_and_checks_class(tmp_path):
    buffer = io.BytesIO()
    Image.new("RGB", (200, 100), "white").save(buffer, format="PNG")

    class Client:
        get_job = AsyncMock(return_value={"start_frame": 0, "stop_frame": 9})
        get_job_labels = AsyncMock(return_value=[{"id": 1, "name": "car"}])

        async def iter_selected_job_frames(self, job_id, frames):
            assert job_id == 17 and frames == [3]
            yield buffer.getvalue()

    seen = {}

    class Auditor:
        model = "vision"

        def refine_box(self, **kwargs):
            seen.update(kwargs)
            with Image.open(kwargs["crop_path"]) as crop:
                seen["size"] = crop.size
            return {"found": True, "bbox": [0.0, 0.0, 1.0, 1.0], "reason": "whole crop"}, SequenceAuditUsage(), False

    service = SimpleNamespace(
        store=SimpleNamespace(get_video_job=lambda _: SimpleNamespace(cvat_job_id=17)),
        seed_reader=SimpleNamespace(client=Client()),
        workspace=SimpleNamespace(job_dir=lambda _: tmp_path),
    )
    # A 20x20 px box at (80, 40): the 48 px context floor gives a 117x100 crop starting at (32, 0).
    result = await module.refine_frame_box(
        service, "job", 3, "car", [0.4, 0.4, 0.5, 0.6], "include mirrors", auditor=Auditor()
    )
    assert seen["label"] == "car" and seen["instruction"] == "include mirrors" and seen["size"] == (117, 100)
    assert result["found"] and result["bbox"] == pytest.approx([32 / 200, 0.0, 149 / 200, 1.0])
    assert result["source_frame"] == 3 and result["model"] == "vision"
    with pytest.raises(ValueError, match="Class is not in the CVAT job"):
        await module.refine_frame_box(service, "job", 3, "truck", [0.4, 0.4, 0.5, 0.6], "", auditor=Auditor())
    with pytest.raises(ValueError, match="outside the CVAT job"):
        await module.refine_frame_box(service, "job", 30, "car", [0.4, 0.4, 0.5, 0.6], "", auditor=Auditor())
