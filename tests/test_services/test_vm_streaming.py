import io
import json

import numpy as np
import pytest
from PIL import Image

from kaggle.sam3_benchmark.main import windows
from src.config import Settings
from src.services.video.stitching import encode_mask
from src.services.vm.frames import FrameProvider, plan_window
from src.services.vm.storage import VmStore
from src.services.vm.streaming import StreamingEngine, boundary_mapping, validate_event


def fixture(tmp_path, count=105):
    store = VmStore(tmp_path / "state.sqlite3", tmp_path / "jobs")
    profile = {
        "variant": "sam3.1",
        "checkpoint": "fixture.pt",
        "checkpoint_sha256": "a" * 64,
        "source_commit": "b" * 40,
    }
    job = store.create(
        1,
        {
            "profile": profile,
            "classes": ["car", "person"],
            "total_frames": count,
            "redacted_sha256": "c" * 64,
            "cvat_job_id": 2,
        },
        "encrypted fixture",
    )
    store.update(job["job_id"], status="running", total_frames=count)
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buffer, format="JPEG")

    class Source:
        delivered = 0

        async def iter_selected_job_frames(self, job, indexes):
            for index in indexes:
                self.delivered += 1
                yield buffer.getvalue()

    return store, store.get(job["job_id"]), Source()


def test_online_planner_equivalent_including_cuts_and_18000_frames():
    for count, cuts in ((1, set()), (48, {48}), (105, {47, 48, 93}), (18000, {48, 49, 500, 17999})):
        records = [{"source_frame_index": i, "scene_cut_before": i in cuts} for i in range(count)]
        expected = windows(records, 48, 4)
        actual, start = [], 0
        while start < count:
            batch, next_start = plan_window(records[start : start + 49], count, 48, 4)
            actual.append((start, batch))
            if next_start is None:
                break
            start = next_start
        assert actual == expected
        assert {r["source_frame_index"] for _, batch in actual for r in batch} == set(range(count))


def test_engine_starts_before_all_input_loads_once_and_resume_loads_none(tmp_path):
    store, job, source = fixture(tmp_path)
    settings = Settings(_env_file=None, local_frame_cache_max_bytes=1048576)
    loads, published = [], []

    class GPU:
        def __init__(self, *args):
            loads.append(source.delivered)

        def run(self, command):
            for label in command["classes"]:
                for frame in command["frames"]:
                    yield {
                        "type": "frame",
                        "window": command["window"],
                        "frame": frame["frame_index"],
                        "class": label,
                        "image_sha256": frame["sha256"],
                        "observations": [],
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

    engine = StreamingEngine(store, settings, source, lambda e, m: published.append((e["frame"], e["class"])), GPU)
    engine.run(job)
    assert len(loads) == 1 and loads[0] < 105
    assert set(published) == {(f, c) for f in range(105) for c in job["classes"]}
    before = len(loads)
    engine.run(store.get(job["job_id"]))
    assert len(loads) == before
    assert json.loads((store.directory(job["job_id"]) / "public/coverage.json").read_text())["complete"]
    checkpoint = store.directory(job["job_id"]) / "checkpoints/000000.json"
    corrupt = json.loads(checkpoint.read_text())
    corrupt["identity"] = "0" * 64
    checkpoint.write_text(json.dumps(corrupt))
    with pytest.raises(ValueError, match="identity"):
        engine.run(store.get(job["job_id"]))


@pytest.mark.asyncio
async def test_cache_pin_evict_reload_hash_and_capacity(tmp_path):
    store, job, source = fixture(tmp_path, 4)
    provider = FrameProvider(store, job["job_id"], source, 2, 1500)
    await provider.acquire([0, 1])
    assert provider.size <= 1500
    provider.release([0, 1])
    await provider.acquire([2, 3])
    assert len(provider.cache) == 2 and 0 not in provider.cache
    provider.release([2, 3])
    await provider.acquire([0])
    assert source.delivered == 5
    provider.release([0])
    with pytest.raises(ValueError, match="active SAM window"):
        await provider.acquire([0, 1, 2])


def test_frame_mask_checksum_and_boundary_keep_id():
    from dataclasses import asdict

    mask = asdict(encode_mask(np.ones((8, 8), dtype=bool)))

    def event(frame, local):
        return {
            "type": "frame",
            "frame": frame,
            "class": "car",
            "window": 0,
            "image_sha256": "a" * 64,
            "observations": [{"track_id": local, "label": "car", "bbox": [0, 0, 8, 8], "mask": mask, "score": 0.9}],
        }

    old, new = [event(i, "old") for i in (1, 2)], [event(i, "new") for i in (1, 2)]
    mapping, _, _ = boundary_mapping(old, new, "car", 1, {"old": "stable-id"}, False)
    assert mapping["new"] == "stable-id"
    validate_event(old[0], {1: {"sha256": "a" * 64, "width": 8, "height": 8}}, ["car"])
    old[0]["observations"][0]["bbox"] = [0, 0, 7, 8]
    with pytest.raises(ValueError, match="mask extent"):
        validate_event(old[0], {1: {"sha256": "a" * 64, "width": 8, "height": 8}}, ["car"])


def test_partial_retry_freezes_delivered_frames_and_relinks_changed_local_ids(tmp_path):
    from dataclasses import asdict

    from src.services.vm.comparison import compare
    from src.services.vm.contracts import file_sha256

    store, job, source = fixture(tmp_path, 6)
    job = store.update(job["job_id"], classes=["car"])
    mask = asdict(encode_mask(np.ones((8, 8), dtype=bool)))
    generations = []
    accepted = []

    def publish(event, mapping):
        if store.event(job["job_id"], f"published:{event['frame']}:car", {"frame": event["frame"], "class": "car"}):
            accepted.append((event["frame"], mapping[event["observations"][0]["track_id"]]))

    class GPU:
        def __init__(self, *args):
            self.generation = len(generations)
            generations.append(self.generation)

        def run(self, command):
            for frame in command["frames"]:
                if not self.generation and frame["frame_index"] == 3:
                    raise ValueError("fixture GPU interruption")
                yield {
                    "type": "frame",
                    "window": 0,
                    "frame": frame["frame_index"],
                    "class": "car",
                    "image_sha256": frame["sha256"],
                    "observations": [
                        {
                            "track_id": "old" if not self.generation else "new",
                            "label": "car",
                            "bbox": [0, 0, 8, 8],
                            "mask": mask,
                            "score": 0.9,
                        }
                    ],
                }
            yield {
                "type": "complete",
                "identity": command["identity"],
                "frames": [r["frame_index"] for r in command["frames"]],
            }

        def close(self):
            pass

    engine = StreamingEngine(store, Settings(_env_file=None, local_frame_cache_max_bytes=1048576), source, publish, GPU)
    with pytest.raises(ValueError, match="interruption"):
        engine.run(job)
    engine.run(store.get(job["job_id"]))
    assert sorted(frame for frame, _id in accepted) == list(range(6))
    assert len({identity for _frame, identity in accepted}) == 1
    root = store.directory(job["job_id"])
    checkpoint = json.loads((root / "checkpoints/000000.json").read_text())
    events = [json.loads((root / entry["path"]).read_text()) for entry in checkpoint["events"]]
    for event in events:
        for row in event["observations"]:
            row["track_id"] = checkpoint["mapping"][row["track_id"]]
    batch = {
        "schema": "autolabel-redacted-batch-reference/v1",
        "input_manifest_sha256": file_sha256(root / "public/input-manifest.json"),
        "profile": json.loads((root / "profile.json").read_text()),
        "events": events,
    }
    assert compare(root, batch)["exact_event_equivalence"]
    batch["input_manifest_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="same verified redacted"):
        compare(root, batch)
