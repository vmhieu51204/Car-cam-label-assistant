"""Exercise the exact portable CVAT overlay helper without Django/GPU dependencies."""

import importlib.util
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import RLock
from types import SimpleNamespace

import httpx
import pytest

from src.services.cvat.read_client import CvatReadClient, CvatReadError

path = Path(__file__).parents[2] / "integrations/cvat-autolabel/files/cvat/apps/engine/autolabel_revision.py"
spec = importlib.util.spec_from_file_location("cvat_atomic_proposal", path)
overlay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(overlay)


def test_content_revision_changes_even_when_native_version_stays_zero():
    a = {"version": 0, "tracks": []}
    b = {"version": 0, "tracks": [{"shapes": [{"points": [1, 2, 3, 4]}]}]}
    assert overlay.annotation_revision(a) != overlay.annotation_revision(b)
    assert 0 <= overlay.annotation_revision(a) < 2**52
    assert overlay.annotation_revision(a) == overlay.annotation_revision({**a, "version": 9})


def test_atomic_guard_conflict_and_lost_response_retry():
    state = {"version": 0, "tracks": []}
    lock = RLock()
    writes = []
    objects = SimpleNamespace(select_for_update=lambda: SimpleNamespace(get=lambda **kw: None))
    transaction = SimpleNamespace(atomic=lambda: lock)

    def create(job_id, data, action):
        writes.append(data)
        state["tracks"].extend(data["tracks"])

    revision = str(overlay.annotation_revision(state))

    def accept(identity):
        return overlay.guarded_create(
            job_id=1,
            expected_revision=revision,
            proposal_id=identity,
            data={"tracks": [{"attributes": [{"value": "vlm-proposal:" + identity}]}]},
            transaction=transaction,
            job_model=SimpleNamespace(objects=objects),
            get_annotations=lambda job_id: state,
            patch_annotations=create,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(accept, ("a", "b")))
    assert sorted(code for code, body in responses) == [200, 409]
    assert len(writes) == 1
    identity = state["tracks"][0]["attributes"][0]["value"].removeprefix("vlm-proposal:")
    assert accept(identity)[0] == 200
    assert len(writes) == 1


@pytest.mark.asyncio
async def test_missing_server_extension_fails_closed():
    def handler(request):
        assert request.headers["X-Autolabel-Revision"] == "1"
        return httpx.Response(200, json={"version": 0, "tracks": []})

    client = CvatReadClient(base_url="http://cvat.test", token="secret", transport=httpx.MockTransport(handler))
    with pytest.raises(CvatReadError, match="extension is missing"):
        await client.get_guarded_job_annotations(1)


