import copy
import json

import pytest

from src.services.evaluation.phase0 import file_hash
from src.services.evaluation.tracking_protocol import (
    assess_gate,
    begin_final,
    finish_final,
    freeze_dev,
    select_runs,
    verify_freeze,
    verify_gate_reports,
)


def protocol():
    return {
        "splits": {"dev": ["0013", "0020"], "held_out": ["0004"]},
        "runs": [{"sequence": "0013"}, {"sequence": "0020"}],
        "held_out_history_status": "unverified",
    }


def report(names, *, candidate=False):
    metric = {
        "precision": 0.8,
        "recall": 0.8,
        "mean_iou": 0.8,
        "tracking": {
            "idf1": 0.8,
            "id_switches": 0,
            "fragmentations": 1 if candidate else 2,
            "tracks": [{"track_id": "a"}, {"track_id": "b"}],
        },
    }
    return {
        "runs": [
            {
                "sequence": n,
                "input_hashes": {"labels": "same"},
                "views": {
                    v: {"metrics": {"thresholds": {t: copy.deepcopy(metric) for t in ("0.5", "0.7")}}}
                    for v in ("stitched_sampled", "interpolated_timeline")
                },
            }
            for n in names
        ]
    }


def test_gate_statuses_and_never_changes_production():
    p = protocol()
    a, b = report(["0013", "0020"]), report(["0013", "0020"], candidate=True)
    assert assess_gate(p)["status"] == "not_evaluated"
    assert assess_gate(p, a, b)["status"] == "dev_passed_unverified"
    worse = copy.deepcopy(b)
    worse["runs"][0]["views"]["stitched_sampled"]["metrics"]["thresholds"]["0.7"]["recall"] -= 0.01
    assert assess_gate(p, a, worse)["status"] == "dev_failed"
    ha, hb = report(["0004"]), report(["0004"], candidate=True)
    with pytest.raises(ValueError, match="explicit final"):
        assess_gate(p, a, b, held_baseline=ha, held_candidate=hb)
    assert (
        assess_gate(p, a, b, held_baseline=ha, held_candidate=hb, final_authorized=True)["status"]
        == "dev_passed_unverified"
    )
    p["held_out_history_status"] = "confirmed_unused"
    assert (
        assess_gate(p, a, b, held_baseline=ha, held_candidate=hb, final_authorized=True)["status"]
        == "eligible_for_review"
    )
    hb["runs"][0]["views"]["interpolated_timeline"]["metrics"]["thresholds"]["0.5"]["precision"] -= 0.01
    gate = assess_gate(p, a, b, held_baseline=ha, held_candidate=hb, final_authorized=True)
    assert gate["status"] == "held_out_failed" and not gate["production_flag"] and not gate["change_production_default"]


def test_registry_extension_and_low_support_unavailable():
    p = protocol()
    p["splits"]["dev"].append("0001")
    p["runs"].append({"sequence": "0001"})
    assert [r["sequence"] for r in select_runs(p)] == ["0013", "0020", "0001"]
    p["runs"].append({"sequence": "0004"})
    assert "0004" not in {r["sequence"] for r in select_runs(p)}
    a, b = report(["0013", "0020", "0001"]), report(["0013", "0020", "0001"], candidate=True)
    a["runs"][0]["views"]["stitched_sampled"]["metrics"]["thresholds"]["0.5"]["tracking"]["tracks"].pop()
    b["runs"][0]["views"]["stitched_sampled"]["metrics"]["thresholds"]["0.5"]["tracking"]["tracks"].pop()
    gate = assess_gate(p, a, b)
    assert gate["status"] == "not_evaluated" and any(c["availability"] == "unavailable" for c in gate["dev_checks"])


def frozen_files(tmp_path):
    p = protocol()
    p["held_out_history_status"] = "confirmed_unused"
    p["runs"].append({"sequence": "0004"})
    for role in ("result", "mapping", "labels"):
        source = tmp_path / (role + ".json")
        source.write_text("synthetic frozen input")
        p["runs"][-1][role] = str(source)
    p["runs"][-1]["sha256"] = {role: file_hash(tmp_path / (role + ".json")) for role in ("result", "mapping", "labels")}
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(p))
    parameters = tmp_path / "params.json"
    parameters.write_text('{"example_only":true}')
    a, b = report(["0013", "0020"]), report(["0013", "0020"], candidate=True)
    a["protocol_sha256"] = b["protocol_sha256"] = file_hash(path)
    a["evaluation_split"] = b["evaluation_split"] = "dev"
    b["parameters_sha256"] = file_hash(parameters)
    ap, bp = tmp_path / "a.json", tmp_path / "b.json"
    ap.write_text(json.dumps(a))
    bp.write_text(json.dumps(b))
    freeze = tmp_path / "freeze.json"
    freeze_dev(path, ap, bp, parameters, freeze)
    return path, freeze, parameters


def test_freeze_hashes_final_once_cache_and_input_changes(tmp_path):
    path, freeze, parameters = frozen_files(tmp_path)
    assert verify_freeze(path, freeze)["parameters"]["sha256"] == file_hash(parameters)
    output = tmp_path / "final"
    identity, cached = begin_final(path, freeze, output)
    assert cached is None
    with pytest.raises(ValueError, match="already reserved"):
        begin_final(path, freeze, output)
    with pytest.raises(ValueError, match="already reserved"):
        begin_final(path, freeze, tmp_path / "another-final-directory")
    output.mkdir()
    report_path = output / "diagnosis.json"
    cached_report = {"cached": True, "protocol_sha256": file_hash(path), "evaluation_split": "held_out"}
    report_path.write_text(json.dumps(cached_report))
    finish_final(path, output, identity, {"diagnosis": report_path})
    assert begin_final(path, freeze, output)[1] == cached_report
    parameters.write_text("changed")
    with pytest.raises(ValueError, match="evidence changed"):
        begin_final(path, freeze, output)


def test_unverified_history_missing_runs_and_existing_output_block_final(tmp_path):
    path, freeze, _ = frozen_files(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    (output / "diagnosis.json").write_text("preserve")
    with pytest.raises(ValueError, match="separate output"):
        begin_final(path, freeze, output)
    p = json.loads(path.read_text())
    p["held_out_history_status"] = "unverified"
    path.write_text(json.dumps(p))
    with pytest.raises(ValueError, match="history unverified"):
        begin_final(path, freeze, tmp_path / "new")
    assert not (tmp_path / "new").exists()


def test_gate_missing_overlap_evidence_is_incomplete():
    p = protocol()
    a, b = report(["0013", "0020"]), report(["0013", "0020"], candidate=True)
    b["runs"][0] = {"sequence": "0013", "status": "missing_overlap_predictions"}
    assert assess_gate(p, a, b)["status"] == "not_evaluated"
    a, b = report(["0013", "0020"]), report(["0013", "0020"], candidate=True)
    ha = report(["0004"])
    hb = {"runs": [{"sequence": "0004", "status": "missing_overlap_predictions"}]}
    assert (
        assess_gate(p, a, b, held_baseline=ha, held_candidate=hb, final_authorized=True)["status"]
        == "dev_passed_unverified"
    )


def test_gate_rejects_stale_substituted_unsealed_reports(tmp_path):
    path, freeze, parameters = frozen_files(tmp_path)
    files = [tmp_path / "a.json", tmp_path / "b.json"]
    assert not verify_gate_reports(path, freeze, [*files, None, None])
    stale = tmp_path / "stale.json"
    report_a = json.loads(files[0].read_text())
    report_a["protocol_sha256"] = "0" * 64
    stale.write_text(json.dumps(report_a))
    with pytest.raises(ValueError, match="stale protocol"):
        verify_gate_reports(path, freeze, [stale, files[1], None, None])
    substituted = tmp_path / "substituted.json"
    report_a = json.loads(files[0].read_text())
    report_a["extra"] = "different evidence"
    substituted.write_text(json.dumps(report_a))
    with pytest.raises(ValueError, match="frozen evidence"):
        verify_gate_reports(path, freeze, [substituted, files[1], None, None])
    held = []
    for role in ("baseline", "candidate"):
        r = report(["0004"], candidate=role == "candidate")
        r.update(protocol_sha256=file_hash(path), evaluation_split="held_out", parameters_sha256=file_hash(parameters))
        hp = tmp_path / ("held-" + role + ".json")
        hp.write_text(json.dumps(r))
        held.append(hp)
    with pytest.raises(ValueError, match="not sealed"):
        verify_gate_reports(path, freeze, [*files, *held])
    output = tmp_path / "final"
    identity, _ = begin_final(path, freeze, output)
    finish_final(path, output, identity, {"baseline": held[0], "candidate": held[1]})
    assert verify_gate_reports(path, freeze, [*files, *held])
    held[1].write_text(json.dumps({**json.loads(held[1].read_text()), "changed": True}))
    with pytest.raises(ValueError, match="sealed final evidence"):
        verify_gate_reports(path, freeze, [*files, *held])
