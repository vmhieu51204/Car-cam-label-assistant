import json

import httpx
import pytest

from src.models.video import BoundingBox2D
from src.services.cvat.track_client import CvatTrackClient, CvatWriteError


@pytest.mark.asyncio
async def test_cvat_accept_rescue_detects_revision_race_and_preserves_existing():
    requests = []

    class ReadClient:
        async def get_job_annotations(self, job_id):
            return {
                "version": 8,
                "shapes": [{"frame": 0, "label_id": 1, "type": "rectangle", "points": [1, 1, 5, 5]}],
                "tracks": [],
            }

        async def get_job_labels(self, job_id):
            return [{"id": 1, "name": "car", "attributes": []}]

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"version": 9})

    client = CvatTrackClient(
        base_url="http://cvat.test", token="token", read_client=ReadClient(), transport=httpx.MockTransport(respond)
    )
    box = BoundingBox2D(xtl=10, ytl=10, xbr=30, ybr=30)
    with pytest.raises(CvatWriteError, match="changed"):
        await client.accept_rescue_rectangle(cvat_job_id=1, frame=2, label_id=1, bbox=box, expected_revision=7)
    assert not requests
    revision = await client.accept_rescue_rectangle(cvat_job_id=1, frame=2, label_id=1, bbox=box, expected_revision=8)
    assert revision == 9
    assert requests[0]["version"] == 8 and requests[0]["shapes"][0]["frame"] == 2
    assert requests[0]["tracks"] == []


@pytest.mark.asyncio
async def test_cvat_accept_rescue_track_is_revision_guarded_and_limited_to_one_frame():
    requests = []

    class ReadClient:
        async def get_guarded_job_annotations(self, job_id):
            return {"version": 0, "autolabel_revision": 3, "shapes": [], "tracks": []}

        async def get_job_labels(self, job_id):
            return [{"id": 1, "name": "car", "attributes": [{"name": "autolabel_track_id", "id": 7}]}]

        async def get_job(self, job_id):
            return {"stop_frame": 20}

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"autolabel_revision": 4})

    client = CvatTrackClient(
        base_url="http://cvat.test", token="token", read_client=ReadClient(), transport=httpx.MockTransport(respond)
    )
    box = BoundingBox2D(xtl=10, ytl=12, xbr=40, ybr=50)
    with pytest.raises(CvatWriteError, match="changed"):
        await client.accept_rescue_track(
            cvat_job_id=1, frame=7, label_id=1, bbox=box, expected_revision=2, proposal_id="test"
        )
    assert requests == []
    revision = await client.accept_rescue_track(
        cvat_job_id=1, frame=7, label_id=1, bbox=box, expected_revision=3, proposal_id="test"
    )
    assert revision == 4
    payload = requests[0]
    assert payload["version"] == 0 and payload["shapes"] == []
    assert [shape["frame"] for shape in payload["tracks"][0]["shapes"]] == [7, 8]
    assert payload["tracks"][0]["shapes"][1]["outside"] is True


@pytest.mark.asyncio
async def test_accept_response_lost_uses_proposal_identity_before_revision_check():
    class ReadClient:
        async def get_guarded_job_annotations(self, job_id):
            return {
                "version": 0,
                "autolabel_revision": 10,
                "tracks": [{"label_id": 1, "attributes": [{"spec_id": 7, "value": "vlm-proposal:abc"}]}],
            }

        async def get_job_labels(self, job_id):
            return [{"id": 1, "name": "car", "attributes": [{"name": "autolabel_track_id", "id": 7}]}]

    def no_request(request):
        pytest.fail("accept retry must not create a duplicate track")

    client = CvatTrackClient(
        base_url="http://cvat.test", token="token", read_client=ReadClient(), transport=httpx.MockTransport(no_request)
    )
    assert (
        await client.accept_rescue_track(
            cvat_job_id=1,
            frame=7,
            label_id=1,
            bbox=BoundingBox2D(xtl=10, ytl=12, xbr=40, ybr=50),
            expected_revision=3,
            proposal_id="abc",
        )
        == 10
    )
