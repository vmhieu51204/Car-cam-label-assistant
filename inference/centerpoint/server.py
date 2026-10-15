"""Loopback-only job runner for the isolated CenterPoint CUDA image."""

import hashlib
import json
import os
import re
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from uuid import UUID

DATA = Path("/datasets/nuscenes")
JOBS = Path("/jobs")
CHECKPOINT = Path("/models/centerpoint_pillar02_nuscenes_191a3822.pth")
CHECKPOINT_SHA = "191a38226d2ce990f26cc3f78c454310cc562d81d29e86d8ca8fa39c0a44215d"
LOCK = threading.Lock()


def digest(path):
    with path.open("rb") as stream:
        value = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
        return value.hexdigest()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def reply(self, code, payload):
        content = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        if self.path != "/health":
            self.reply(404, {"detail": "Not found"})
            return
        self.reply(
            200,
            {
                "ready": True,
                "model": "CenterPoint",
                "checkpoint_sha256": CHECKPOINT_SHA,
                "dataset_manifest_sha256": digest(DATA / "dataset-manifest.json"),
                "bevfusion_checkpoint_sha256": digest(Path("/models/bevfusion-5239b1af.pth")) if Path("/models/bevfusion-5239b1af.pth").is_file() else None,
            },
        )

    def do_POST(self):
        if self.path not in ("/infer", "/infer-upload"):
            self.reply(404, {"detail": "Not found"})
            return
        if not LOCK.acquire(blocking=False):
            self.reply(409, {"detail": "Inference busy"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 4096:
                raise ValueError("Invalid request size")
            body = json.loads(self.rfile.read(size))
            identity = str(UUID(body["job_id"]))
            if self.path == "/infer-upload":
                from upload_service import infer
                root = JOBS / identity
                if not root.is_dir() or root.is_symlink():
                    raise ValueError("Job directory missing")
                self.reply(200, infer(body, root))
                return
            manifest = json.loads((DATA / "dataset-manifest.json").read_text())
            scene = body["scene"]
            count = body["max_frames"]
            if scene not in manifest["scenes"] or type(count) is not int or not 1 <= count <= 200:
                raise ValueError("Unsupported scene/frame count")
            count = min(count, len(manifest["scenes"][scene]["frames"]))
            root = JOBS / identity
            if not root.is_dir() or root.is_symlink():
                raise ValueError("Job directory missing")
            output = root / "private/lidar3d"
            output.mkdir(exist_ok=True)
            target, receipt = output / f"{scene}.json", output / "inference-receipt.json"
            if target.exists() and receipt.exists():
                previous = json.loads(receipt.read_text())
                if previous["scene"] != scene or previous["frames"] != count or previous["sha256"] != digest(target):
                    raise ValueError("Existing inference identity differs")
                self.reply(200, previous)
                return
            with (output / "inference.log").open("w") as log:
                process = subprocess.Popen(
                    [
                        "python",
                        "/workspace/run.py",
                        "--dataset-root",
                        str(DATA),
                        "--scene",
                        scene,
                        "--max-frames",
                        str(count),
                        "--output-dir",
                        str(output),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                deadline = time.monotonic() + 1800
                while process.poll() is None:
                    if (root / "private/cancelled.json").exists() or time.monotonic() > deadline:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        raise ValueError("3D inference cancelled or timed out")
                    time.sleep(0.5)
            if process.returncode:
                raise ValueError("CenterPoint failed; inspect private 3D inference log")
            completed = [
                int(value) for value in re.findall(r"^frame=(\d+) token=", (output / "inference.log").read_text(), re.M)
            ]
            if completed != list(range(count)):
                raise ValueError("Detector did not complete every requested frame")
            result = {"scene": scene, "frames": count, "sha256": digest(target), "checkpoint_sha256": CHECKPOINT_SHA}
            receipt.write_text(json.dumps(result))
            self.reply(200, result)
        except (ValueError, KeyError, OSError) as error:
            self.reply(422, {"detail": str(error)})
        finally:
            LOCK.release()


if __name__ == "__main__":
    if digest(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError("CenterPoint checkpoint checksum mismatch")
    ThreadingHTTPServer(("0.0.0.0", 8013), Handler).serve_forever()
