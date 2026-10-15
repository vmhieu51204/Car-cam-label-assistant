"""Execute only the fixed BEVFusion entry point inside an owner job directory."""

import hashlib
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path

CHECKPOINT_SHA = "e68c6c084d8971f4f879751b0ed3981f575c24ee3cf385a1e39197c26a9bb3ab"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def infer(body, root):
    directory = root / "private/lidar3d"
    manifest_path = directory / "dataset.json"
    manifest_sha = digest(manifest_path)
    if body["manifest_sha256"] != manifest_sha:
        raise ValueError("Dataset manifest identity differs")
    manifest = json.loads(manifest_path.read_text())
    count = len(manifest["frames"])
    if manifest["format"] != "nuscenes" or not 1 <= count <= 200:
        raise ValueError("Unsupported BEVFusion input")
    if body.get("provisioned") is True:
        data = Path("/datasets/nuscenes")
    else:
        input_root = (directory / "input").resolve()
        data = (input_root / manifest["root"]).resolve()
        if not data.is_relative_to(input_root):
            raise ValueError("Dataset escaped the job directory")
    target = directory / "uploaded-predictions.json"
    receipt = directory / "uploaded-inference-receipt.json"
    if target.is_file() and receipt.is_file():
        previous = json.loads(receipt.read_text())
        if previous["manifest_sha256"] != manifest_sha or previous["sha256"] != digest(target) or previous["checkpoint_sha256"] != CHECKPOINT_SHA:
            raise ValueError("Existing BEVFusion output identity differs")
        return previous
    log_path = directory / "inference.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(["python", "/workspace/upload.py", "--manifest", str(manifest_path),
                                    "--dataset-root", str(data)], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + 1800
        while process.poll() is None:
            if (root / "private/cancelled.json").exists() or time.monotonic() > deadline:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise ValueError("BEVFusion cancelled or timed out")
            time.sleep(0.5)
    if process.returncode:
        raise ValueError("BEVFusion failed; inspect private inference.log")
    completed = [int(value) for value in re.findall(r"^frame=(\d+) token=", log_path.read_text(), re.M)]
    if completed != list(range(count)):
        raise ValueError("BEVFusion did not complete every requested frame")
    result = {"frames": count, "manifest_sha256": manifest_sha, "sha256": digest(target), "checkpoint_sha256": CHECKPOINT_SHA}
    receipt.write_text(json.dumps(result))
    return result
