"""One-command healthcheck for the local SAM 3 / SAM 3.1 setup (entry scripts:
``sam3/healthcheck_sam3.py``, ``sam3.1/healthcheck_sam31.py``).

Checks Python, torch/CUDA, GPU, triton, sam3, checkpoint (or Hugging Face access) and the
payload. ``--load-model`` also loads the model and runs one prompt on the first frames.
Prints PASS/WARN/FAIL per item, writes logs/<timestamp>_healthcheck_<model>/healthcheck.json,
and exits 1 if anything FAILs.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from samlocal import DEFAULT_CACHE_DIR, DEFAULT_LOG_DIR, SAM3_PINNED_COMMIT, is_wsl
from samlocal.backend import build_backend_for
from samlocal.logs import create_run_dir, setup_logging, write_json
from samlocal.metrics import GIB, environment, package_version, sam3_commit
from samlocal.models import ModelSpec, get_spec

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


class Checks:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def run(self, name: str, fn: Callable[[], tuple[str, str]]) -> str:
        try:
            status, detail = fn()
        except Exception as exc:  # a crashing check is a failing check
            status, detail = FAIL, f"{type(exc).__name__}: {exc}"
        self.rows.append({"check": name, "status": status, "detail": detail})
        print(f"[{status}] {name:<22} {detail}", flush=True)
        return status

    @property
    def ok(self) -> bool:
        return all(row["status"] != FAIL for row in self.rows)


def main(model: str | ModelSpec, argv: list[str] | None = None) -> int:
    spec = get_spec(model) if isinstance(model, str) else model
    parser = argparse.ArgumentParser(
        description=f"Healthcheck for the local {spec.label} setup.\n\nExample:\n"
                    f"  python healthcheck_{spec.slug}.py --payload ~/data/payload.zip "
                    f"--checkpoint ~/models/{spec.hf_checkpoint} --load-model",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--payload", type=Path, default=None, help="payload.zip / extracted folder / frames folder")
    parser.add_argument("--checkpoint", type=Path, default=None, help=f"local {spec.hf_checkpoint} (else HF download)")
    parser.add_argument("--load-model", action="store_true", help=f"also load {spec.label} and run a 4-frame smoke test")
    parser.add_argument("--prompt", default=None, help="prompt for the smoke test (default: first payload prompt)")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    args = parser.parse_args(argv)

    run_dir = create_run_dir(args.log_dir, f"healthcheck_{spec.slug}")
    log = setup_logging(run_dir)
    checks = Checks()
    state: dict[str, Any] = {}

    def python_version() -> tuple[str, str]:
        ok = sys.version_info >= (3, 12)
        return (PASS if ok else FAIL), f"{platform.python_version()} (SAM3 needs >= 3.12)"

    def os_info() -> tuple[str, str]:
        system = "WSL2" if is_wsl() else platform.system()
        status = WARN if system == "Windows" else PASS
        note = " (native Windows: needs triton-windows; --compile unsupported)" if system == "Windows" else ""
        return status, f"{system} {platform.release()}{note}"

    def torch_cuda() -> tuple[str, str]:
        import torch

        if not torch.cuda.is_available():
            return FAIL, f"torch {torch.__version__} but CUDA unavailable (driver / CPU-only wheel?)"
        state["torch"] = torch
        return PASS, f"torch {torch.__version__}, CUDA {torch.version.cuda}, cuDNN {torch.backends.cudnn.version()}"

    def gpu() -> tuple[str, str]:
        torch = state["torch"]
        props = torch.cuda.get_device_properties(0)
        vram = props.total_memory / GIB
        bf16 = torch.cuda.is_bf16_supported()
        free, _ = torch.cuda.mem_get_info()
        status = PASS if bf16 and vram >= 11 else WARN
        return status, (f"{props.name}, {vram:.1f} GiB VRAM ({free / GIB:.1f} free), cc {props.major}.{props.minor}, "
                        f"bf16={bf16}{fa3_note(props.major)}")

    def fa3_note(major: int) -> str:
        if not spec.multiplex:
            return ""
        return f", FA3={'yes' if major >= 9 else 'no (Ada/Ampere -> use_fa3=False)'}"

    def triton() -> tuple[str, str]:
        import triton as triton_module

        return PASS, f"triton {getattr(triton_module, '__version__', '?')} (sam3 imports it unconditionally)"

    def sam3() -> tuple[str, str]:
        from sam3.model_builder import build_sam3_predictor  # noqa: F401

        commit = sam3_commit()
        status = PASS if commit in (None, SAM3_PINNED_COMMIT) else WARN
        return status, f"sam3 {package_version('sam3')} commit {commit or 'unknown'} (pinned {SAM3_PINNED_COMMIT[:10]})"

    def deps() -> tuple[str, str]:
        missing = []
        for module in ("einops", "scipy", "timm", "ftfy", "iopath", "psutil", "PIL", "cv2", "pkg_resources"):
            try:
                __import__(module)
            except ImportError:
                missing.append(module)
        return (FAIL if missing else PASS), (f"missing: {missing}" if missing else "einops scipy timm ftfy iopath ...")

    def checkpoint() -> tuple[str, str]:
        if args.checkpoint:
            path = args.checkpoint.expanduser()
            if not path.is_file():
                return FAIL, f"not found: {path}"
            size = path.stat().st_size / GIB
            return (PASS if size > 0.5 else FAIL), f"{path} ({size:.2f} GiB)"
        from huggingface_hub import get_token, hf_hub_download

        if not get_token():
            return FAIL, f"no --checkpoint and no HF token: run `hf auth login` and request access to {spec.hf_repo}"
        config = hf_hub_download(spec.hf_repo, "config.json")
        return PASS, f"HF access to {spec.hf_repo} ok ({spec.hf_checkpoint} downloads on first load; {config})"

    def payload() -> tuple[str, str]:
        if not args.payload:
            return WARN, "no --payload given"
        from samlocal.payload import open_payload

        opened = open_payload(args.payload, args.cache_dir)
        state["payload"] = opened
        info = opened.describe()
        status = PASS if opened.prompts or args.prompt else WARN
        return status, (f"{info['layout']}: {info['num_frames']} frames {info['width']}x{info['height']}, "
                        f"prompts={info['prompts'] or 'none (pass --prompts to benchmark)'}")

    def smoke() -> tuple[str, str]:
        from samlocal.staging import remove_window, stage_window

        opened = state.get("payload")
        if opened is None:
            return FAIL, "--load-model needs a readable --payload"
        prompt = args.prompt or (opened.prompts[0] if opened.prompts else "car")
        frames = opened.frames[:4]
        backend = build_backend_for(spec, checkpoint=args.checkpoint, logger=log)
        frame_dir = stage_window(frames, args.cache_dir / "staging" / "healthcheck")
        try:
            info = backend.load()
            backend.reset_peak()
            started = time.perf_counter()
            detections = backend.run_window(frame_dir, len(frames), [prompt], frame_offset=0)[prompt]
            seconds = time.perf_counter() - started
            peak = backend.peak()
        finally:
            remove_window(frame_dir)
            backend.close()
        return PASS, (f"load {info['model_load_seconds']:.1f}s, '{prompt}' on {len(frames)} frames {seconds:.2f}s, "
                      f"{len(detections)} detections, peak {peak['peak_reserved_gib']:.2f} GiB")

    print(f"local-sam healthcheck ({spec.label}) -> {run_dir}")
    checks.run("python", python_version)
    checks.run("os", os_info)
    if checks.run("torch+cuda", torch_cuda) == PASS:
        checks.run("gpu", gpu)
    checks.run("triton", triton)
    checks.run("sam3 import", sam3)
    checks.run("dependencies", deps)
    checks.run("checkpoint", checkpoint)
    checks.run("payload", payload)
    if args.load_model:
        if checks.ok:
            checks.run("smoke inference", smoke)
        else:
            checks.rows.append({"check": "smoke inference", "status": FAIL, "detail": "skipped: earlier FAIL"})
            print(f"[{FAIL}] {'smoke inference':<22} skipped: fix the failures above first")

    try:
        env = environment()
    except Exception:
        env = {"error": traceback.format_exc()}
    write_json(run_dir / "healthcheck.json", {"ok": checks.ok, "checks": checks.rows, "environment": env})
    print(f"\nRESULT: {'OK' if checks.ok else 'FAILED'}  (details: {run_dir / 'healthcheck.json'})")
    return 0 if checks.ok else 1
