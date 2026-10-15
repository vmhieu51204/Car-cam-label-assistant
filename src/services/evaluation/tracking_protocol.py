"""Offline protocol boundaries and candidate gate. Never enables production features."""

import hashlib
import json
import math
from pathlib import Path

from src.services.evaluation.phase0 import file_hash

VIEWS = ("stitched_sampled", "interpolated_timeline")


def select_runs(protocol: dict, split: str = "dev") -> list[dict]:
    if "splits" not in protocol:
        runs = protocol["runs"]
        if split != "dev" or any(r["sequence"] not in {"0013", "0020"} for r in runs):
            raise ValueError("explicit phase1 split protocol required")
        return runs
    splits = protocol["splits"]
    dev, held = splits["dev"], splits["held_out"]
    if (
        len(dev) != len(set(dev))
        or len(held) != len(set(held))
        or set(dev) & set(held)
        or not {"0013", "0020"} <= set(dev)
    ):
        raise ValueError("invalid DEV/HELD-OUT split")
    if split not in {"dev", "held_out"}:
        raise ValueError("unknown evaluation split")
    runs = protocol["runs"]
    if len({r["sequence"] for r in runs}) != len(runs):
        raise ValueError("duplicate sequence in prediction registry")
    if any(r["sequence"] not in set(dev) | set(held) for r in runs):
        raise ValueError("prediction sequence outside protocol split")
    return [r for r in runs if r["sequence"] in splits[split]]


def _comparisons(baseline: dict, candidate: dict, expected: set[str], *, dev: bool) -> tuple[list[dict], bool]:
    left = {r["sequence"]: r for r in baseline["runs"]}
    right = {r["sequence"]: r for r in candidate["runs"]}
    if len(left) != len(baseline["runs"]) or len(right) != len(candidate["runs"]):
        raise ValueError("duplicate report sequence")
    if not set(left) <= expected or not set(right) <= expected:
        raise ValueError("report contains sequences outside requested split")
    checks = []
    complete = bool(expected) and set(left) == set(right) == expected
    for sequence in sorted(set(left) & set(right)):
        if any("views" not in row or "input_hashes" not in row for row in (left[sequence], right[sequence])):
            complete = False
            continue
        if left[sequence]["input_hashes"] != right[sequence]["input_hashes"]:
            raise ValueError("baseline/candidate frozen input hashes differ")
        for view in VIEWS:
            for threshold in ("0.5", "0.7"):
                if any(
                    view not in row["views"] or threshold not in row["views"][view]["metrics"]["thresholds"]
                    for row in (left[sequence], right[sequence])
                ):
                    complete = False
                    continue
                a = left[sequence]["views"][view]["metrics"]["thresholds"][threshold]
                b = right[sequence]["views"][view]["metrics"]["thresholds"][threshold]
                support = len(a["tracking"]["tracks"])
                if support != len(b["tracking"]["tracks"]):
                    raise ValueError("GT track support changed")

                def gt_signature(metric: dict) -> list:
                    return sorted(
                        (str(t.get("video_id")), str(t["track_id"]), t.get("gt_frames"))
                        for t in metric["tracking"]["tracks"]
                    )

                if gt_signature(a) != gt_signature(b):
                    raise ValueError("GT track identities/exposure changed")
                available = support >= 2
                tests = {}
                for field in ("precision", "recall", "mean_iou", "idf1"):
                    aa = a["tracking"].get(field) if field == "idf1" else a.get(field)
                    bb = b["tracking"].get(field) if field == "idf1" else b.get(field)
                    valid = all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in (aa, bb))
                    available &= valid
                    tests[field + "_non_decreasing"] = bb >= aa if valid else None
                for field in ("id_switches", "fragmentations"):
                    aa, bb = a["tracking"][field], b["tracking"][field]
                    if any(type(v) is not int or v < 0 for v in (aa, bb)):
                        raise ValueError("invalid temporal event count")
                    tests[field + "_non_increasing"] = bb <= aa
                before = a["tracking"]["id_switches"] + a["tracking"]["fragmentations"]
                after = b["tracking"]["id_switches"] + b["tracking"]["fragmentations"]
                tests["combined_events_strictly_decreased"] = after < before if dev else after <= before
                checks.append(
                    {
                        "sequence": sequence,
                        "view": view,
                        "iou_threshold": threshold,
                        "gt_tracks": support,
                        "availability": "available" if available else "unavailable",
                        "checks": tests,
                        "passed": available and all(tests.values()),
                    }
                )
                complete &= available
    return checks, complete


def assess_gate(
    protocol: dict,
    baseline: dict | None = None,
    candidate: dict | None = None,
    *,
    held_baseline: dict | None = None,
    held_candidate: dict | None = None,
    final_authorized: bool = False,
) -> dict:
    """Compare ready reports only. Missing evaluation is never labeled a quality failure."""
    expected_dev = {r["sequence"] for r in select_runs(protocol)}
    result = {
        "schema": "autolabel-phase1-tracking-gate/v1",
        "status": "not_evaluated",
        "production_flag": False,
        "change_production_default": False,
        "dev_checks": [],
        "held_out_checks": [],
        "held_out_history_status": protocol.get("held_out_history_status", "unverified"),
        "note": "Proposed uncalibrated gate; no threshold tuning to pass",
    }
    if baseline is None or candidate is None:
        return result
    checks, complete = _comparisons(baseline, candidate, expected_dev, dev=True)
    result["dev_checks"] = checks
    result["dev_sequence_count"] = len({c["sequence"] for c in checks})
    result["dev_gt_tracks_by_view"] = {
        v: sum(c["gt_tracks"] for c in checks if c["view"] == v and c["iou_threshold"] == "0.5") for v in VIEWS
    }
    if not complete:
        return result
    if not all(c["passed"] for c in checks):
        result["status"] = "dev_failed"
        return result
    result["status"] = "dev_passed_unverified"
    if held_baseline is None or held_candidate is None:
        return result
    if not final_authorized:
        raise ValueError("held-out reports require explicit final evaluation and DEV freeze")
    checks, complete = _comparisons(held_baseline, held_candidate, set(protocol["splits"]["held_out"]), dev=False)
    result["held_out_checks"] = checks
    result["held_out_sequence_count"] = len({c["sequence"] for c in checks})
    if any(c["availability"] == "available" and not c["passed"] for c in checks):
        result["status"] = "held_out_failed"
    elif complete and protocol.get("held_out_history_status") == "confirmed_unused":
        result["status"] = "eligible_for_review"
    return result


def freeze_dev(
    protocol_path: Path, baseline_path: Path, candidate_path: Path, parameters_path: Path, output: Path
) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    baseline, candidate = [json.loads(p.read_text(encoding="utf-8")) for p in (baseline_path, candidate_path)]
    protocol_sha = file_hash(protocol_path)
    if any(report.get("protocol_sha256") != protocol_sha for report in (baseline, candidate)):
        raise ValueError("DEV reports do not bind current protocol")
    if candidate.get("parameters_sha256") != file_hash(parameters_path):
        raise ValueError("candidate parameters do not match frozen selection")
    if assess_gate(protocol, baseline, candidate)["status"] != "dev_passed_unverified":
        raise ValueError("DEV gate must pass before parameter freeze")
    record = {
        "schema": "autolabel-dev-tracking-freeze/v1",
        "protocol_sha256": protocol_sha,
        **{
            name: {"path": str(path.resolve()), "sha256": file_hash(path)}
            for name, path in (
                ("baseline", baseline_path),
                ("candidate", candidate_path),
                ("parameters", parameters_path),
            )
        },
    }
    data = json.dumps(record, sort_keys=True, indent=2).encode()
    if output.exists() and output.read_bytes() != data:
        raise ValueError("existing DEV freeze differs")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)
    return record


def verify_freeze(protocol_path: Path, freeze_path: Path) -> dict:
    record = json.loads(freeze_path.read_text(encoding="utf-8"))
    if record.get("schema") != "autolabel-dev-tracking-freeze/v1" or record.get("protocol_sha256") != file_hash(
        protocol_path
    ):
        raise ValueError("DEV freeze does not match current protocol")
    for name in ("baseline", "candidate", "parameters"):
        if file_hash(Path(record[name]["path"])) != record[name]["sha256"]:
            raise ValueError("frozen DEV evidence changed")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    reports = [json.loads(Path(record[name]["path"]).read_text(encoding="utf-8")) for name in ("baseline", "candidate")]
    if reports[1].get("parameters_sha256") != record["parameters"]["sha256"]:
        raise ValueError("DEV freeze parameter binding invalid")
    if assess_gate(protocol, *reports)["status"] != "dev_passed_unverified":
        raise ValueError("frozen DEV gate no longer passes")
    return record


def final_ledger_path(protocol_path: Path) -> Path:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    root = (protocol_path.resolve().parent / protocol.get("final_ledger_root", "final-ledgers")).resolve()
    population = {
        "held_out": protocol["splits"]["held_out"],
        "dataset_root": protocol.get("dataset_root"),
        "inputs": {n: protocol.get("input_hashes", {}).get(n) for n in protocol["splits"]["held_out"]},
    }
    identity = hashlib.sha256(json.dumps(population, sort_keys=True).encode()).hexdigest()
    return root / (identity + ".json")


def begin_final(
    protocol_path: Path, freeze_path: Path, output: Path, *, root: Path = Path(".")
) -> tuple[dict, dict | None]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("held_out_history_status") != "confirmed_unused":
        raise ValueError("HELD-OUT history unverified; final evaluation blocked")
    frozen = verify_freeze(protocol_path, freeze_path)
    runs = select_runs(protocol, "held_out")
    if {r["sequence"] for r in runs} != set(protocol["splits"]["held_out"]):
        raise ValueError("missing HELD-OUT predictions; final evaluation blocked")
    for run in runs:
        for role in ("result", "mapping", "labels"):
            if file_hash(root / run[role]) != run["sha256"][role]:
                raise ValueError("HELD-OUT frozen input hash mismatch")
    identity = {
        "protocol_sha256": file_hash(protocol_path),
        "dev_freeze_sha256": file_hash(freeze_path),
        "parameters_sha256": frozen["parameters"]["sha256"],
        "runs_sha256": hashlib.sha256(json.dumps(runs, sort_keys=True).encode()).hexdigest(),
    }
    ledger = final_ledger_path(protocol_path)
    if ledger.exists():
        previous = json.loads(ledger.read_text(encoding="utf-8"))
        if (
            previous.get("identity") != identity
            or previous.get("status") != "completed"
            or previous.get("output") != str(output.resolve())
        ):
            raise ValueError("final evaluation already reserved/incompatible; cannot score again")
        report = previous["reports"].get("diagnosis") or previous["reports"]["baseline"]
        if file_hash(Path(report["path"])) != report["sha256"]:
            raise ValueError("final cached report hash mismatch")
        return identity, json.loads(Path(report["path"]).read_text(encoding="utf-8"))
    if (output / "diagnosis.json").exists() or (output / "events.csv").exists():
        raise ValueError("final evaluation needs separate output; existing report must be preserved")
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("x", encoding="utf-8") as stream:
        json.dump({"identity": identity, "status": "reserved", "output": str(output.resolve())}, stream)
    return identity, None


def finish_final(protocol_path: Path, output: Path, identity: dict, reports: dict[str, Path]) -> dict:
    ledger = final_ledger_path(protocol_path)
    previous = json.loads(ledger.read_text(encoding="utf-8"))
    if (
        previous["status"] != "reserved"
        or previous["identity"] != identity
        or previous["output"] != str(output.resolve())
    ):
        raise ValueError("final reservation does not match output/identity")
    if "diagnosis" not in reports and not {"baseline", "candidate"} <= reports.keys():
        raise ValueError("final reports require diagnosis or baseline/candidate pair")
    sealed = {}
    for role, path in reports.items():
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("protocol_sha256") != identity["protocol_sha256"] or report.get("evaluation_split") != "held_out":
            raise ValueError("final report provenance does not match reservation")
        if role == "candidate" and report.get("parameters_sha256") != identity["parameters_sha256"]:
            raise ValueError("final candidate parameters differ from DEV freeze")
        sealed[role] = {"path": str(path.resolve()), "sha256": file_hash(path)}
    record = {**previous, "status": "completed", "reports": sealed}
    temporary = ledger.with_suffix(".pending")
    temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
    temporary.replace(ledger)
    output.mkdir(parents=True, exist_ok=True)
    (output / "final-held-out-ledger.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def verify_gate_reports(protocol_path: Path, freeze_path: Path | None, paths: list[Path | None]) -> bool:
    expected = file_hash(protocol_path)
    for index, path in enumerate(paths):
        if path is not None:
            report = json.loads(path.read_text(encoding="utf-8"))
            if report.get("protocol_sha256") != expected or report.get("evaluation_split") != (
                "dev" if index < 2 else "held_out"
            ):
                raise ValueError("gate report has stale protocol or wrong evaluation split")
    if freeze_path is None:
        if any(paths[2:]):
            raise ValueError("held-out reports require explicit final evaluation and DEV freeze")
        return False
    frozen = verify_freeze(protocol_path, freeze_path)
    for index, role in enumerate(("baseline", "candidate")):
        path = paths[index]
        if path is None or file_hash(path) != frozen[role]["sha256"]:
            raise ValueError("DEV gate reports differ from frozen evidence")
    if not any(paths[2:]):
        return False
    ledger = final_ledger_path(protocol_path)
    if not ledger.exists():
        raise ValueError("held-out report is not sealed by final evaluation")
    receipt = json.loads(ledger.read_text(encoding="utf-8"))
    if receipt.get("status") != "completed" or receipt["identity"]["dev_freeze_sha256"] != file_hash(freeze_path):
        raise ValueError("held-out final evaluation is incomplete or belongs to another freeze")
    for index, role in ((2, "baseline"), (3, "candidate")):
        sealed = receipt["reports"].get(role)
        path = paths[index]
        if path is None or sealed is None or file_hash(path) != sealed["sha256"]:
            raise ValueError("held-out gate report does not match sealed final evidence")
    return True
