"""Build the retained SAM3 notebooks from canonical scripts, without secrets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "kaggle/notebooks"
STAGES = {
    "sam3": "SAM3 video propagation",
    "sam3_seed": "SAM3 image seed",
    "sam3_benchmark": "SAM3 sequence and benchmark",
}
MAIN_GUARD = '\nif __name__ == "__main__":\n    main()\n'


def build_notebook(stage: str) -> dict:
    if stage not in STAGES:
        raise ValueError(f"Unsupported Kaggle stage: {stage}")
    script = (ROOT / "kaggle" / stage / "main.py").read_text(encoding="utf-8")
    if not script.endswith(MAIN_GUARD):
        raise ValueError("unexpected main guard")
    probe = (ROOT / "kaggle/runtime_probe.py").read_text(encoding="utf-8")
    contract = (ROOT / "kaggle/checkpoint_contract.py").read_text(encoding="utf-8")
    bootstrap = (
        "import sys, types\n" + "_package = types.ModuleType('kaggle')\nsys.modules.setdefault('kaggle', _package)\n"
    )
    modules = [("runtime_probe", probe), ("checkpoint_contract", contract)]
    if stage == "sam3_benchmark":
        modules.append(("tracking_parameters", (ROOT / "kaggle/tracking_parameters.py").read_text(encoding="utf-8")))
        modules.append(("dms_memory", (ROOT / "kaggle/dms_memory.py").read_text(encoding="utf-8")))
        modules.append(("native_memory", (ROOT / "kaggle/native_memory.py").read_text(encoding="utf-8")))
        bootstrap += "_benchmark_package = types.ModuleType('kaggle.sam3_benchmark')\nsys.modules['kaggle.sam3_benchmark'] = _benchmark_package\nsys.modules['kaggle.sam3_benchmark.main'] = sys.modules[__name__]\n"
    for module, source in modules:
        bootstrap += f"_module = types.ModuleType('kaggle.{module}')\nexec({source!r}, _module.__dict__)\nsys.modules['kaggle.{module}'] = _module\n"

    def cell(kind: str, source: str, name: str) -> dict[str, Any]:
        row: dict[str, Any] = {"cell_type": kind, "id": name, "metadata": {}, "source": source.splitlines(keepends=True)}
        if kind == "code":
            row.update(execution_count=None, outputs=[])
        return row

    return {
        "cells": [
            cell(
                "markdown",
                f"# {STAGES[stage]}\nPrivate Kaggle GPU notebook. Attach exactly one sam3.pt checkpoint and one private payload.",
                "autolabel-setup",
            ),
            cell("code", script[: -len(MAIN_GUARD)], "autolabel-model"),
            cell("code", bootstrap, "autolabel-runtime-probe"),
            cell("code", "main()\n", "autolabel-run"),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for stage in STAGES:
        path = OUTPUT / f"autolabel_{stage}.ipynb"
        path.write_text(json.dumps(build_notebook(stage), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(path.name)


if __name__ == "__main__":
    main()
