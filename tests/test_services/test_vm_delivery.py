"""Exercise operation receipts, native-normalized shapes and concurrent Save semantics."""

import importlib.util
import sys
import threading
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.config import Settings
from src.services.vm.storage import VmStore
from src.services.vm.worker import IncrementalPublisher


@pytest.fixture
def server(monkeypatch):
    root = Path(__file__).resolve().parents[2] / "integrations/cvat-autolabel/files/cvat/apps/engine"

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    revision = load("revision", root / "autolabel_revision.py")
    policy = load("policy", root / "autolabel_stream.py")
    receipts = {}
    row = SimpleNamespace(state={}, save=lambda **kwargs: None)

    def filter_receipt(**kwargs):
        return SimpleNamespace(first=lambda: receipts.get(kwargs["operation_id"]))

    def create_receipt(**kwargs):
        receipts[kwargs["operation_id"]] = SimpleNamespace(
            result=kwargs.get("result", {}), **{k: v for k, v in kwargs.items() if k != "result"}
        )

    models = SimpleNamespace(
        AutoLabelStreamState=SimpleNamespace(objects=SimpleNamespace(get_or_create=lambda **kw: (row, False))),
        AutoLabelStreamOperation=SimpleNamespace(objects=SimpleNamespace(filter=filter_receipt, create=create_receipt)),
    )
    monkeypatch.setitem(sys.modules, "cvat.apps.engine.models", models)
    monkeypatch.setitem(sys.modules, "cvat.apps.engine.autolabel_revision", revision)
    data = {"version": 0, "tags": [], "shapes": [], "tracks": [], "intervals": []}
    next_id = [0]
    lock = threading.RLock()

    @contextmanager
    def atomic():
        with lock:
            yield

    def patch(_job, payload, action):
        returned = {"version": 0, "tags": [], "shapes": [], "tracks": [], "intervals": []}
        for kind in ("tags", "shapes", "tracks", "intervals"):
            for incoming in payload[kind]:
                if action in ("delete", "update"):
                    data[kind] = [r for r in data[kind] if r["id"] != incoming["id"]]
                if action != "delete":
                    record = deepcopy(incoming)
                    if action == "create":
                        next_id[0] += 1
                        record["id"] = next_id[0]
                    for shape in record.get("shapes", []):
                        next_id[0] += 1
                        shape["id"] = next_id[0]
                        shape.pop("keyframe", None)
                        if not shape["attributes"]:
                            shape["attributes"] = [{"spec_id": 9, "value": "pending"}]
                    data[kind].append(record)
                    returned[kind].append(deepcopy(record))
        return returned

    def call(body):
        return policy.handle_stream(
            job_id=1,
            data=body,
            mode=body["mode"],
            transaction=SimpleNamespace(atomic=atomic),
            job_model=SimpleNamespace(
                objects=SimpleNamespace(select_for_update=lambda: SimpleNamespace(get=lambda **kw: None))
            ),
            get_annotations=lambda _id: deepcopy(data),
            patch_annotations=patch,
            put_annotations=None,
        )

    return SimpleNamespace(
        data=lambda: deepcopy(data), call=call, revision=revision.annotation_revision, policy=policy, receipts=receipts
    )


def increment(server, frame=0, op="first"):
    return {
        "mode": "incremental",
        "operation_id": op,
        "expected_revision": server.revision(server.data()),
        "operations": [
            {
                "identity": "vm:fixture:object",
                "frame": frame,
                "points": [1, 2, 10, 12],
                "label_id": 1,
                "total_frames": 8,
                "attributes": [{"spec_id": 4, "value": "vm:fixture:object"}],
            }
        ],
    }


def test_redacted_stream_save_vlm_export_share_media_and_preserve_edits(tmp_path, server):
    import asyncio
    import json
    import subprocess
    from dataclasses import asdict

    import httpx
    import numpy as np
    from cryptography.fernet import Fernet

    from src.services.video.stitching import encode_mask
    from src.services.vm.contracts import file_sha256
    from src.services.vm.privacy import redact_video
    from src.services.vm.reader import VerifiedFrameReader
    from src.services.vm.streaming import StreamingEngine
    from src.services.vm.vlm import VlmProfiles

    settings = Settings(
        _env_file=None,
        db_path=tmp_path / "db",
        vm_work_root=tmp_path / "jobs",
        vlm_settings_encryption_key=Fernet.generate_key().decode(),
        vlm_cache_path=tmp_path / "cache",
        local_frame_cache_max_bytes=1048576,
    )
    store = VmStore(settings.db_path, settings.vm_work_root)
    model = {"face_sha256": "a" * 64, "plate_sha256": "b" * 64, "source_commit": "c" * 40}
    job = store.create(
        1,
        {
            "profile": {"checkpoint": "fixture.pt", "checkpoint_sha256": "d" * 64, "source_commit": "e" * 40},
            "classes": ["car", "person"],
            "manual_regions": [],
            "privacy_profile": {"privacy_" + k: v for k, v in model.items()},
        },
        "fixture",
    )
    directory = store.directory(job["job_id"])
    source = directory / "private/source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x64:rate=10",
            "-frames:v",
            "4",
            "-c:v",
            "libx264rgb",
            "-crf",
            "0",
            str(source),
        ],
        check=True,
    )

    class Detector:
        def detect(self, image):
            return [[20, 20, 40, 40]]

    output = directory / "public/redacted.mp4"
    receipt = redact_video(source, output, Detector(), expected_sha256=file_sha256(source), model_metadata=model)
    job = store.update(
        job["job_id"],
        status="running",
        source_sha256=file_sha256(source),
        total_frames=4,
        cvat_job_id=1,
        in_place=True,
        redacted_sha256=receipt["output_sha256"],
    )
    class Source:
        async def iter_selected_job_frames(self, _job, indexes):
            for _index in indexes:
                raise AssertionError("In-place inference must read private redacted media, not raw CVAT frames")
                yield b""

    reader = VerifiedFrameReader(store, job, Source())
    # An in-place run is bound to the existing raw CVAT job, so frame reads must
    # come from the verified public video even when CVAT has the original frames.
    label = {"id": 1, "name": "car", "attributes": [{"name": "autolabel_track_id", "id": 4, "mutable": False}]}

    class Publisher(IncrementalPublisher):
        async def send(self, batch):
            batch["expected_revision"] = server.revision(server.data())
            assert server.call({"mode": "incremental", **batch})[0] == 200

    publisher = Publisher(settings, store, job, [label, {**label, "id": 2, "name": "person"}], {"sessionid": "fixture"})
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:30, 10:30] = True
    changed = []

    class GPU:
        def __init__(self, *args):
            pass

        def run(self, command):
            for name in command["classes"]:
                for frame in command["frames"]:
                    if name == "car" and frame["frame_index"] == 1:
                        publisher.flush()
                        base = server.data()
                        local = deepcopy(base)
                        local["tracks"][0]["shapes"][0]["points"] = [11, 11, 31, 31]
                        assert (
                            server.call(
                                {"mode": "save", "operation_id": "edit-during-stream", "base": base, "local": local}
                            )[0]
                            == 200
                        )
                        changed.append(True)
                    yield {
                        "type": "frame",
                        "window": command["window"],
                        "frame": frame["frame_index"],
                        "class": name,
                        "image_sha256": frame["sha256"],
                        "observations": []
                        if name == "person"
                        else [
                            {
                                "track_id": "local",
                                "label": "car",
                                "bbox": [10, 10, 30, 30],
                                "mask": asdict(encode_mask(mask)),
                                "score": 0.9,
                            }
                        ],
                    }
            yield {
                "type": "complete",
                "window": command["window"],
                "identity": command["identity"],
                "frames": [f["frame_index"] for f in command["frames"]],
                "runtime": {},
            }

        def close(self):
            pass

    try:
        StreamingEngine(store, settings, reader, publisher, GPU).run(job)
        publisher.flush()
    finally:
        publisher.close()
    assert changed and server.data()["tracks"][0]["shapes"][0]["points"] == [11, 11, 31, 31]
    assert store.get(job["job_id"])["complete_frames"] == 4
    image = directory / "cache/frame_000000.jpg"
    assert asyncio.run(anext(reader.iter_selected_job_frames(1, [0]))) == image.read_bytes()
    profiles = VlmProfiles(store, settings)
    profiles.save(1, model="fixture-vision", base_url="https://example.com/v1", key="fixture-user-key")
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": json.dumps({"missing": "no", "proposals": [], "reason": "fixture"})}}
                ],
                "usage": {},
            },
        )

    auditor = profiles.auditor(profiles.snapshot(1), transport=httpx.MockTransport(respond))
    answer, _, _ = auditor.propose_missing_boxes(
        overlay_path=image,
        overlay_sha256=file_sha256(image),
        classes=("car", "person"),
        image_width=64,
        image_height=64,
        existing_boxes=[],
    )
    assert len(calls) == 1 and not answer.proposals
    export = directory / "public/review-export.json"
    export.write_text(json.dumps(server.data()))
    assert json.loads(export.read_text())["tracks"][0]["shapes"][0]["points"] == [11, 11, 31, 31]


def test_receipt_replay_keeps_manual_save_and_server_shape_defaults(server):
    body = increment(server)
    assert server.call(body)[0] == 200
    baseline = server.data()
    local = deepcopy(baseline)
    local["tracks"][0]["shapes"][0]["points"] = [3, 4, 13, 14]
    save = {"mode": "save", "operation_id": "save-one", "base": baseline, "local": local}
    assert server.call(save)[0] == 200
    deletion_base = server.data()
    body["expected_revision"] = server.revision(server.data())
    assert server.call(body)[1]["replayed"]
    assert len(server.data()["tracks"]) == 1
    assert server.data()["tracks"][0]["shapes"][0]["points"] == [3, 4, 13, 14]
    assert server.call(increment(server, 1, "later"))[0] == 200
    assert any(s["frame"] == 1 and not s["outside"] for s in server.data()["tracks"][0]["shapes"])
    assert server.call(save)[1]["replayed"]
    assert len(server.data()["tracks"]) == 1
    local["tracks"] = []
    assert server.call({"mode": "save", "operation_id": "delete", "base": deletion_base, "local": local})[0] == 200
    assert server.call(increment(server, 2, "after-delete"))[0] == 200
    assert not server.data()["tracks"]


def test_conflict_choices_stale_revision_rejected_and_create_id_returned(server):
    assert server.call(increment(server))[0] == 200
    base = server.data()
    a, b = deepcopy(base), deepcopy(base)
    a["tracks"][0]["shapes"][0]["points"] = [2, 3, 12, 13]
    b["tracks"][0]["shapes"][0]["points"] = [4, 5, 14, 15]
    assert server.call({"mode": "save", "operation_id": "save-a", "base": base, "local": a})[0] == 200
    req = {"mode": "save", "operation_id": "save-b", "base": base, "local": b}
    code, conflict = server.call(req)
    assert code == 409
    req["choices"] = {conflict["conflict"]: "local"}
    req["expected_revision"] = 0
    assert server.call(req)[1]["conflict"] == "revision"
    req["expected_revision"] = conflict["revision"]
    assert server.call(req)[0] == 200
    base = server.data()
    local = deepcopy(base)
    local["shapes"] = [
        {"clientID": 99, "frame": 0, "label_id": 1, "type": "rectangle", "points": [1, 1, 5, 5], "attributes": []}
    ]
    response = server.call({"mode": "save", "operation_id": "create", "base": base, "local": local})[1]
    assert response["created_ids"]["shapes"]["99"] == server.data()["shapes"][0]["id"]


def test_outbox_survives_lost_response_without_duplicate_tracks(tmp_path, server):
    store = VmStore(tmp_path / "db", tmp_path / "jobs")
    job = store.create(1, {"profile": {"variant": "sam3.1"}}, "encrypted-fixture")
    job = store.update(job["job_id"], status="running", total_frames=8, cvat_job_id=1)
    job["owner"] = 1
    label = {"id": 1, "name": "car", "attributes": [{"name": "autolabel_track_id", "id": 4, "mutable": False}]}

    class Publisher(IncrementalPublisher):
        async def send(self, batch):
            batch["expected_revision"] = server.revision(server.data())
            assert server.call({"mode": "incremental", **batch})[0] == 200
            raise TimeoutError("fixture response lost")

    publisher = Publisher(Settings(_env_file=None), store, job, [label], {"sessionid": "fixture"})
    try:
        with pytest.raises(TimeoutError):
            publisher(
                {
                    "frame": 0,
                    "class": "car",
                    "observations": [{"track_id": "local", "bbox": [1, 2, 10, 12], "score": 0.9}],
                },
                {"local": "stable"},
            )
        import json

        intent = json.loads((store.directory(job["job_id"]) / "private/delivery.json").read_text())
        intent["expected_revision"] = server.revision(server.data())
        assert server.call({"mode": "incremental", **intent})[1]["replayed"]
        assert len(server.data()["tracks"]) == 1
    finally:
        publisher.close()


@pytest.mark.asyncio
async def test_in_place_binding_keeps_target_and_adds_review_attributes(monkeypatch, tmp_path):
    from unittest.mock import AsyncMock

    from src.services.vm.worker import VmWorker

    job = {"in_place": True, "cvat_job_id": 15, "cvat_task_id": 19, "classes": ["car"], "owner": 1}
    read = SimpleNamespace(
        get_job=AsyncMock(return_value={"task_id": 19}),
        get_job_labels=AsyncMock(return_value=[{"id": 8, "name": "car", "attributes": []}]),
    )
    writer = SimpleNamespace(ensure_label_attributes=AsyncMock(), create_video_task=AsyncMock())
    authorize = AsyncMock(return_value=1)
    monkeypatch.setattr("src.services.vm.worker.CvatSessionReadClient", lambda *args: read)
    monkeypatch.setattr("src.services.vm.worker.UserTaskClient", lambda *args: writer)
    monkeypatch.setattr("src.services.vm.worker.CvatSessionAuthorizer", lambda *args: SimpleNamespace(authorize=authorize))
    worker = object.__new__(VmWorker)
    worker.settings = Settings(_env_file=None)
    assert await worker.bind(job, {}, {"sessionid": "1"}) == (19, 15)
    writer.create_video_task.assert_not_called()
    writer.ensure_label_attributes.assert_awaited_once()
    assert "autolabel_track_id" in {a.name for a in writer.ensure_label_attributes.call_args.args[1]}
    authorize.assert_awaited_once_with("1", 15, write=True)
    read.get_job.return_value = {"task_id": 20}
    with pytest.raises(ValueError, match="task identity changed"):
        await worker.bind(job, {}, {"sessionid": "1"})


def test_in_place_reader_uses_local_video_and_checks_redecoded_frame_hash(tmp_path, monkeypatch):
    import asyncio
    import hashlib
    import io

    import av
    import numpy as np
    from PIL import Image

    from src.services.vm.reader import VerifiedFrameReader

    store = VmStore(tmp_path / "db", tmp_path / "jobs")
    job = store.create(1, {"in_place": True, "cvat_job_id": 15, "total_frames": 3}, "sealed")
    # Store initializes progress counters; set the verified media length afterwards.
    job = store.update(job["job_id"], total_frames=3)
    video = store.directory(job["job_id"]) / "public/redacted.mp4"
    with av.open(str(video), "w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        for color in (20, 80, 140):
            frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), color, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    expected = []
    with av.open(str(video)) as container:
        for frame in container.decode(video=0):
            output = io.BytesIO()
            Image.fromarray(frame.to_ndarray(format="rgb24")).save(output, format="JPEG", quality=95)
            expected.append(output.getvalue())
    monkeypatch.setattr("src.services.vm.reader.verified_media", lambda *args: (video, {}))

    class RawSource:
        def iter_selected_job_frames(self, *args):
            raise AssertionError("Raw CVAT frames must never reach SAM/VLM")

    reader = VerifiedFrameReader(store, job, RawSource())

    async def collect():
        return [frame async for frame in reader.iter_selected_job_frames(15, [2, 0, 2])]

    assert asyncio.run(collect()) == [expected[2], expected[0], expected[2]]
    store.event(job["job_id"], "input:2", {"sha256": hashlib.sha256(expected[2]).hexdigest()})
    assert asyncio.run(collect()) == [expected[2], expected[0], expected[2]]
    store.event(job["job_id"], "input:0", {"sha256": "0" * 64})
    with pytest.raises(ValueError, match="differs from SAM input"):
        asyncio.run(collect())
