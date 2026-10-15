"""Score already-evaluated runs with the official TrackEval (KITTI 2D box: car, pedestrian) and compare with ours.

TrackEval needs numpy < 1.24 (it still uses ``np.float``), so it lives in its own venv, outside this repo:

    ~/venvs/trackeval/bin/python -I kitti/trackeval_kitti.py --payload ~/data/kitti_0014.zip \
        --eval logs/kitti/<stamp>/eval

Reads ``<eval>/kitti_eval.json`` and the ``<eval>/trackeval/<run>/data/<seq>.txt`` files written by eval_kitti.py,
builds a TrackEval ``Kitti2DBox`` folder per run in ``<eval>/trackeval_official/`` (GT cropped to the scored frames
and renumbered from 0), runs HOTA + CLEAR + Identity, writes ``<eval>/trackeval_official.json`` and appends a
section to ``<eval>/kitti_report.md``.

With a ``kitti5`` payload TrackEval still follows the KITTI rules (Van is a distractor of car, our van/truck/cyclist
classes are not scored), so its car numbers differ from ours by design; compare on ``kitti2`` payloads.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))  # kitti_metrics / eval_kitti, also under python -I

from eval_kitti import _fmt, _safe, load_payload  # noqa: E402
from kitti_metrics import PayloadReader  # noqa: E402

REPORT_MARKER = "<!-- trackeval_kitti.py -->"
TRACKEVAL_CLASSES = ("car", "pedestrian")
HOTA_FIELDS = ("HOTA", "DetA", "DetRe", "DetPr", "AssA", "AssRe", "AssPr", "LocA")
OURS = {"HOTA": "hota", "DetA": "det_a", "DetRe": "det_re", "DetPr": "det_pr", "AssA": "ass_a", "AssRe": "ass_re",
        "AssPr": "ass_pr", "LocA": "loc_a"}


def _renumber(lines: list[str], frame_to_index: dict[int, int]) -> list[str]:
    """Keep rows of the scored frames, first column renumbered to 0..N-1."""
    out = []
    for line in lines:
        fields = line.split()
        if fields and int(fields[0]) in frame_to_index:
            out.append(" ".join([str(frame_to_index[int(fields[0])]), *fields[1:]]))
    return out


def build_folder(root: Path, sequence: str, gt_text: str, tracker_file: Path, tracker: str,
                 frames: list[int]) -> None:
    if root.exists():
        shutil.rmtree(root)
    index = {frame: i for i, frame in enumerate(frames)}
    gt_dir = root / "gt" / "label_02"
    gt_dir.mkdir(parents=True)
    (gt_dir / f"{sequence}.txt").write_text("\n".join(_renumber(gt_text.splitlines(), index)) + "\n", encoding="utf-8")
    (root / "gt" / "evaluate_tracking.seqmap.training").write_text(
        f"{sequence} empty 000000 {len(frames):06d}\n", encoding="utf-8")
    data = root / "trackers" / tracker / "data"
    data.mkdir(parents=True)
    rows = _renumber(tracker_file.read_text(encoding="utf-8").splitlines(), index)
    (data / f"{sequence}.txt").write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def run_trackeval(root: Path, tracker: str) -> dict[str, dict[str, Any]]:
    import numpy as np
    import trackeval

    eval_config = {**trackeval.Evaluator.get_default_eval_config(), "USE_PARALLEL": False, "PRINT_RESULTS": False,
                   "PRINT_CONFIG": False, "TIME_PROGRESS": False, "OUTPUT_SUMMARY": False, "OUTPUT_DETAILED": False,
                   "PLOT_CURVES": False, "LOG_ON_ERROR": str(root / "error_log.txt")}
    dataset_config = {**trackeval.datasets.Kitti2DBox.get_default_dataset_config(),
                      "GT_FOLDER": str(root / "gt"), "TRACKERS_FOLDER": str(root / "trackers"),
                      "OUTPUT_FOLDER": str(root / "output"), "TRACKERS_TO_EVAL": [tracker],
                      "CLASSES_TO_EVAL": list(TRACKEVAL_CLASSES), "SPLIT_TO_EVAL": "training", "PRINT_CONFIG": False}
    metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR({"PRINT_CONFIG": False}),
               trackeval.metrics.Identity({"PRINT_CONFIG": False})]
    with contextlib.redirect_stdout(io.StringIO()):  # TrackEval prints progress even when told not to
        results, messages = trackeval.Evaluator(eval_config).evaluate(
            [trackeval.datasets.Kitti2DBox(dataset_config)], metrics)
    message = messages["Kitti2DBox"][tracker]
    if message != "Success":
        raise SystemExit(f"TrackEval failed for {tracker}: {message}")
    combined = results["Kitti2DBox"][tracker]["COMBINED_SEQ"]
    out: dict[str, dict[str, Any]] = {}
    for cls in TRACKEVAL_CLASSES:
        h, c, i = combined[cls]["HOTA"], combined[cls]["CLEAR"], combined[cls]["Identity"]
        row = {field: float(np.mean(h[field])) for field in HOTA_FIELDS}
        row.update({"MOTA": float(c["MOTA"]), "MOTP": float(c["MOTP"]), "IDSW": int(c["IDSW"]),
                    "Frag": int(c["Frag"]), "MT": int(c["MT"]), "ML": int(c["ML"]), "CLR_TP": int(c["CLR_TP"]),
                    "CLR_FN": int(c["CLR_FN"]), "CLR_FP": int(c["CLR_FP"]), "IDF1": float(i["IDF1"]),
                    "IDR": float(i["IDR"]), "IDP": float(i["IDP"])})
        out[cls] = row
    return out


def our_class(manifest: dict, cls: str) -> str:
    """TrackEval class -> our class name of the payload profile (kitti2 calls pedestrians ``person``)."""
    return "person" if cls == "pedestrian" and manifest.get("profile", "kitti2") == "kitti2" else cls


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--payload", type=Path, required=True, help="kitti_<seq>.zip the runs were scored on")
    parser.add_argument("--eval", type=Path, required=True, help="eval folder written by eval_kitti.py")
    args = parser.parse_args(argv)
    try:
        import trackeval  # noqa: F401
    except ImportError:
        raise SystemExit("trackeval not importable: run this with the TrackEval venv (~/venvs/trackeval/bin/python)")

    manifest, _, position_to_source = load_payload(args.payload)
    gt_text = PayloadReader(args.payload).read_text(manifest["labels"])
    sequence = manifest["sequence"]
    evaluated = json.loads((args.eval / "kitti_eval.json").read_text(encoding="utf-8"))
    output: dict[str, Any] = {"schema": "local-sam-kitti-trackeval/v1", "sequence": sequence,
                              "profile": manifest.get("profile", "kitti2"), "runs": {}}
    for run in evaluated["runs"]:
        label = run["run_label"]
        count = run["num_frames"] or len(position_to_source)
        frames = [position_to_source[p] for p in range(min(count, len(position_to_source)))]
        tracker = _safe(label)
        root = args.eval / "trackeval_official" / tracker
        build_folder(root, sequence, gt_text, args.eval / "trackeval" / tracker / "data" / f"{sequence}.txt",
                     tracker, frames)
        official = run_trackeval(root, tracker)
        per_class = run["metrics"]["per_class"]
        for cls, row in official.items():
            ours = per_class.get(our_class(manifest, cls))
            if ours:
                row["ours"] = {field: ours["hota"][key] for field, key in OURS.items()}
                row["ours"]["MOTA"] = ours["clear"]["mota"]
                row["ours"]["IDF1"] = ours["identity"]["idf1"]
        output["runs"][label] = official
        print(f"{label}: " + ", ".join(f"{cls} HOTA {row['HOTA']:.3f} (ours {row.get('ours', {}).get('HOTA', 0):.3f})"
                                      for cls, row in official.items()))
    (args.eval / "trackeval_official.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    append_report(args.eval / "kitti_report.md", output)
    print(f"written: {args.eval}/trackeval_official.json, section in kitti_report.md")
    return 0


def append_report(path: Path, output: dict[str, Any]) -> None:
    lines = [REPORT_MARKER, "", f"**Official TrackEval** (Kitti2DBox, profile of payload `{output['profile']}`; "
             "value / ours; KITTI rules: Van is a car distractor, only car and pedestrian are scored)", "",
             "| run | class | HOTA | DetRe | AssRe | DetA | AssA | LocA | MOTA | IDF1 | MT/ML |", "|" + "---|" * 11]
    for label, classes in output["runs"].items():
        for cls, row in classes.items():
            ours = row.get("ours", {})

            def pair(field: str) -> str:
                return f"{_fmt(row[field])} / {_fmt(ours.get(field))}"

            lines.append(f"| {label} | {cls} | {pair('HOTA')} | {pair('DetRe')} | {pair('AssRe')} | {pair('DetA')} | "
                         f"{pair('AssA')} | {_fmt(row['LocA'])} | {pair('MOTA')} | {pair('IDF1')} | "
                         f"{row['MT']}/{row['ML']} |")
    replace_section(path, REPORT_MARKER, lines)


def replace_section(path: Path, marker: str, lines: list[str]) -> None:
    """Put ``lines`` (starting with ``marker``) at the end of the report, dropping an older copy of that section
    but keeping sections of other scripts (each starts with its own ``<!-- ... -->`` marker)."""
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    head, found, tail = text.partition(marker)
    if found:
        later = tail.find("<!-- ")
        text = head.rstrip("\n") + ("\n\n" + tail[later:].rstrip("\n") if later >= 0 else "")
    path.write_text(text.rstrip("\n") + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
