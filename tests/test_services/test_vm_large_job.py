"""18,000-frame coordinator fixture with fake image delivery/storage/GPU, never a quality claim."""

from types import SimpleNamespace

from src.config import Settings
from src.services.vm.contracts import VmProfile
from src.services.vm.streaming import StreamingEngine


def test_streaming_coordinator_covers_18000_frames_and_both_classes(tmp_path, monkeypatch):
    job = {
        "job_id": "fixture",
        "profile": VmProfile(checkpoint="fixture", checkpoint_sha256="a" * 64, source_commit="b" * 40).model_dump(
            by_alias=True
        ),
        "total_frames": 18000,
        "classes": ["car", "person"],
        "redacted_sha256": "c" * 64,
        "cvat_job_id": 1,
        "status": "running",
    }
    calls, published = [], {c: set() for c in job["classes"]}

    def record(i):
        return {
            "source_frame_index": i,
            "frame_index": i,
            "scene_cut_before": i == 8900,
            "sha256": "d" * 64,
            "width": 8,
            "height": 8,
        }

    class Provider:
        size = 512

        def __init__(self, *args):
            pass

        async def acquire(self, indexes):
            calls.extend(indexes)
            return [record(i) for i in indexes]

        def record(self, i):
            return record(i)

        def release(self, *args):
            pass

        def close(self):
            pass

    def update(_id, **values):
        job.update(values)
        return job

    store = SimpleNamespace(
        directory=lambda _id: tmp_path, get=lambda _id: job, update=update, event=lambda *args: True
    )
    artifacts = {}

    def atomic(path, value):
        if path.name == "coverage.json":
            artifacts["coverage"] = value
        return "e" * 64

    loads = []

    class GPU:
        def __init__(self, *args):
            loads.append(len(calls))

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
                "identity": command["identity"],
                "frames": [f["frame_index"] for f in command["frames"]],
            }

        def close(self):
            pass

    monkeypatch.setattr("src.services.vm.streaming.FrameProvider", Provider)
    monkeypatch.setattr("src.services.vm.streaming.atomic_json", atomic)
    StreamingEngine(
        store, Settings(_env_file=None), None, lambda event, _map: published[event["class"]].add(event["frame"]), GPU
    ).run(job)
    assert len(loads) == 1 and loads[0] < 18000
    assert all(frames == set(range(18000)) for frames in published.values())
    assert job["complete_frames"] == 18000 and artifacts["coverage"]["complete"]
