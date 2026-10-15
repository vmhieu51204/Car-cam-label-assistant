"""Offline benchmark CLI supporting prepare-reference, build-bindings, evaluate, and report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from src.models.evaluation import (
    BenchmarkManifest,
    BindingOptions,
    DatasetReference,
    FrameBinding,
    KittiReferenceOptions,
    NuScenesReferenceOptions,
    PredictionNormalizationOptions,
    ProjectionPolicy,
    ReferenceProfile,
    RunEvaluation,
    SequenceKey,
    canonical_json,
    content_hash,
    parse_document,
)
from src.services.evaluation.engine import evaluate_run
from src.services.evaluation.geometry import write_geometry_analysis
from src.services.evaluation.manual import load_manual_reference
from src.services.evaluation.misses import write_miss_analysis
from src.services.evaluation.phase0 import canonical_predictions, compare_baselines, write_baseline
from src.services.evaluation.predictions.sam3 import load_sam3_predictions
from src.services.evaluation.provenance import build_frame_bindings
from src.services.evaluation.references.kitti import load_kitti_reference
from src.services.evaluation.references.nuscenes import load_nuscenes_reference
from src.services.evaluation.temporal import write_temporal_analysis
from src.services.evaluation.tracking import tracking_diagnostics


def cmd_prepare_reference(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol)
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    protocol_data = json.loads(protocol_path.read_text(encoding="utf-8"))
    dataset_type = protocol_data.get("dataset", "kitti")

    if dataset_type == "kitti":
        profile = ReferenceProfile.model_validate(protocol_data["profile"])
        options = KittiReferenceOptions(
            reference_revision=protocol_data.get("reference_revision", "kitti-ref/v1"),
            split=protocol_data.get("split", "training"),
            sequences=tuple(protocol_data["sequences"]),
            profiles=(profile,),
            source_fps=protocol_data.get("source_fps", 10.0),
            timestamp_provenance=protocol_data.get("timestamp_provenance", "declared_fps"),
        )
        ref = load_kitti_reference(dataset_root, options=options)
    elif dataset_type == "nuscenes":
        profile = ReferenceProfile.model_validate(protocol_data["profile"])
        proj = ProjectionPolicy.model_validate(protocol_data.get("projection", {}))
        options = NuScenesReferenceOptions(
            reference_revision=protocol_data.get("reference_revision", "nuscenes-ref/v1"),
            version=protocol_data.get("version", "v1.0-mini"),
            split=protocol_data.get("split", "mini_train"),
            scenes=tuple(protocol_data["scenes"]),
            cameras=tuple(protocol_data.get("cameras", ("CAM_FRONT",))),
            profile=profile,
            projection=proj,
        )
        ref = load_nuscenes_reference(dataset_root, options=options)
    else:
        raise ValueError(f"unsupported dataset in protocol: {dataset_type}")

    ref_path = output_dir / "reference.json"
    ref_path.write_bytes(canonical_json(ref))

    manifest_dict = protocol_data.get("manifest")
    if manifest_dict:
        manifest = BenchmarkManifest.model_validate(manifest_dict)
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_bytes(canonical_json(manifest))

    print(f"Prepared reference saved to {ref_path} ({len(ref.frames)} frames, {len(ref.boxes)} boxes)")


def cmd_build_bindings(args: argparse.Namespace) -> None:
    parse_document(BenchmarkManifest, Path(args.manifest).read_bytes())
    reference_path = Path(args.reference)
    payload_path = Path(args.payload)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ref = parse_document(DatasetReference, reference_path.read_bytes())
    seq_uid = args.sequence
    seq_key = SequenceKey(
        dataset_id=ref.frames[0].key.sequence.dataset_id, split=ref.frames[0].key.sequence.split, sequence_uid=seq_uid
    )

    options = BindingOptions(
        input_kind=args.input_kind,
        sequence=seq_key,
        mapping_id=f"payload-{seq_uid}/v1",
        endpoint_policy="include_final_source_frame" if args.input_kind == "sequence-native" else "sampled_only",
    )

    source_manifests = tuple(Path(p) for p in args.source_manifest) if getattr(args, "source_manifest", None) else ()
    bindings = build_frame_bindings(
        ref,
        payload_path,
        options=options,
        source_manifests=source_manifests,
        result=Path(args.result) if args.result else None,
    )

    output_path.write_text(
        json.dumps([b.model_dump() for b in bindings], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Generated {len(bindings)} bindings to {output_path}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest)
    reference_path = Path(args.reference)
    result_path = Path(args.result)
    payload_path = Path(args.payload) if args.payload else None
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = parse_document(BenchmarkManifest, manifest_path.read_bytes())
    reference = parse_document(DatasetReference, reference_path.read_bytes())

    if args.bindings:
        bindings_data = json.loads(Path(args.bindings).read_text(encoding="utf-8"))
        bindings = tuple(FrameBinding.model_validate(b) for b in bindings_data)
        manifest = BenchmarkManifest.model_validate({**manifest.model_dump(), "bindings": bindings})

    views = tuple(args.views.split(","))
    norm_options = PredictionNormalizationOptions.model_validate({"views": views})

    runs = load_sam3_predictions(result_path, manifest=manifest, payload=payload_path, options=norm_options)
    evaluations: list[RunEvaluation] = []

    for run in runs:
        eval_run = evaluate_run(reference, run, manifest=manifest, dataset_root=output_dir)
        evaluations.append(eval_run)

    report_dict: dict[str, Any] = {
        "schema": "autolabel-evaluation-report/v1",
        "evaluations": [e.model_dump() for e in evaluations],
        "manifest_sha256": content_hash(manifest),
        "reference_sha256": content_hash(reference),
    }
    if args.tracking_diagnostics:
        diagnostics = []
        for evaluation in evaluations:
            for scoped in evaluation.scopes:
                prepared = scoped.prepared_view
                if prepared is None:
                    continue
                native = {x.source_id: x for x in prepared.native_kitti_rows}
                occlusions = {
                    x.annotation_id: ("none", "partial", "heavy", "unknown")[native[x.metadata.kitti_source_id].occluded]
                    for x in prepared.reference_boxes if x.metadata.kitti_source_id in native
                }
                diagnostics.append({"config_id": evaluation.config_id, "view": evaluation.view,
                                    "scope": scoped.scope.scope_id,
                                    "metrics": tracking_diagnostics(prepared.reference_boxes, prepared.prediction_boxes,
                                                                    frames=scoped.scope.frames, occlusions=occlusions)})
        report_dict["tracking_diagnostics"] = diagnostics

    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report_dict, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Evaluated {len(evaluations)} runs. Report written to {report_path}")


def cmd_report(args: argparse.Namespace) -> None:
    report_path = Path(args.report)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(report_path.read_text(encoding="utf-8"))
    evaluations = data.get("evaluations", [])

    rows = []
    for ev in evaluations:
        config_id = ev["config_id"]
        view = ev["view"]
        status = ev["status"]
        for sc in ev.get("scopes", []):
            scope_id = sc["scope"]["scope_id"]
            for grp in sc.get("groups", []):
                if grp.get("aggregation") == "micro":
                    m = grp["diagnostic"]
                    rows.append({"config_id": config_id, "view": view, "status": status, "scope": scope_id,
                                 **{k: m[k] for k in ("tp", "fp", "fn")},
                                 **{k: m[k]["value"] for k in ("precision", "recall", "f1")}})

    csv_path = output_dir / "report.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["config_id", "view", "status", "scope", "tp", "fp", "fn", "precision", "recall", "f1"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary CSV report written to {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline evaluation benchmark runner")
    subparsers = parser.add_subparsers(dest="command", required=True)
    p_phase0 = subparsers.add_parser("phase0", help="Evaluate frozen local artifacts; no service calls")
    p_phase0.add_argument("--protocol", type=Path, required=True)
    p_phase0.add_argument("--root", type=Path, default=Path("."))
    p_phase0.add_argument("--output", type=Path, required=True)
    p_misses = subparsers.add_parser("phase0-misses", help="Diagnose frozen raw SAM3 misses; no service calls")
    p_misses.add_argument("--protocol", type=Path, required=True)
    p_misses.add_argument("--root", type=Path, default=Path("."))
    p_misses.add_argument("--output", type=Path, required=True)
    p_misses.add_argument("--seed", type=int, default=42)
    p_misses.add_argument("--bootstrap-repeats", type=int, default=10000)
    p_misses.add_argument("--image-count", type=int, default=30)
    p_geometry = subparsers.add_parser("phase0-geometry", help="Geometry of frozen raw misses; no service calls")
    p_geometry.add_argument("--protocol", type=Path, required=True)
    p_geometry.add_argument("--misses", type=Path, required=True)
    p_geometry.add_argument("--root", type=Path, default=Path("."))
    p_geometry.add_argument("--output", type=Path, required=True)
    p_geometry.add_argument("--seed", type=int, default=42)
    p_temporal = subparsers.add_parser("phase0-temporal", help="Offline temporal bridge and size-memory diagnostics")
    p_temporal.add_argument("--protocol", type=Path, required=True)
    p_temporal.add_argument("--geometry", type=Path, required=True)
    p_temporal.add_argument("--root", type=Path, default=Path("."))
    p_temporal.add_argument("--output", type=Path, required=True)
    p_temporal.add_argument("--bootstrap-repeats", type=int, default=10000)
    p_manual = subparsers.add_parser("manual", help="Evaluate predictions on explicitly annotated frames")
    p_manual.add_argument("--ground-truth", type=Path, required=True)
    p_manual.add_argument("--predictions", type=Path, required=True)
    p_manual.add_argument("--output", type=Path, required=True)
    p_compare = subparsers.add_parser("compare-phase0", help="Verify equivalent metrics on immutable artifacts")
    p_compare.add_argument("--before", type=Path, required=True)
    p_compare.add_argument("--after", type=Path, required=True)
    p_compare.add_argument("--output", type=Path, required=True)

    p_tracking = subparsers.add_parser("phase1-tracking-diagnosis", help="Frozen tracking event signals")
    p_tracking.add_argument("--protocol", type=Path, required=True)
    p_tracking.add_argument("--root", type=Path, default=Path("."))
    p_tracking.add_argument("--output", type=Path, required=True)
    p_tracking.add_argument("--final-held-out", action="store_true")
    p_tracking.add_argument("--dev-freeze", type=Path)
    p_gate = subparsers.add_parser("phase1-tracking-gate", help="Offline proposed gate; never enable production")
    p_gate.add_argument("--protocol", type=Path, required=True)
    p_gate.add_argument("--baseline", type=Path)
    p_gate.add_argument("--candidate", type=Path)
    p_gate.add_argument("--held-baseline", type=Path)
    p_gate.add_argument("--held-candidate", type=Path)
    p_gate.add_argument("--dev-freeze", type=Path)
    p_gate.add_argument("--output", type=Path, required=True)
    p_freeze = subparsers.add_parser("phase1-tracking-freeze", help="Freeze a passed DEV comparison before final evaluation")
    for field in ("protocol", "baseline", "candidate", "parameters", "output"):
        p_freeze.add_argument("--"+field, type=Path, required=True)
    p_export = subparsers.add_parser("export-experiments", help="Offline reversible-filter/sparse-keyframe counterfactual")
    p_export.add_argument("--protocol", type=Path, required=True)
    p_export.add_argument("--root", type=Path, default=Path("."))
    p_export.add_argument("--output", type=Path, required=True)

    # prepare-reference
    p_prep = subparsers.add_parser("prepare-reference")
    p_prep.add_argument("--protocol", required=True)
    p_prep.add_argument("--dataset-root", required=True)
    p_prep.add_argument("--output", required=True)

    # build-bindings
    p_bind = subparsers.add_parser("build-bindings")
    p_bind.add_argument("--manifest", required=True)
    p_bind.add_argument("--reference", required=True)
    p_bind.add_argument("--payload", required=True)
    p_bind.add_argument("--result", default=None)
    p_bind.add_argument("--input-kind", default="sequence-native")
    p_bind.add_argument("--sequence", required=True)
    p_bind.add_argument("--source-manifest", action="append", default=[])
    p_bind.add_argument("--output", required=True)

    # evaluate
    p_eval = subparsers.add_parser("evaluate")
    p_eval.add_argument("--manifest", required=True)
    p_eval.add_argument("--reference", required=True)
    p_eval.add_argument("--result", required=True)
    p_eval.add_argument("--payload", default=None)
    p_eval.add_argument("--bindings", default=None)
    p_eval.add_argument("--views", default="raw,stitched")
    p_eval.add_argument("--backend", default="none")
    p_eval.add_argument("--tracking-diagnostics", action="store_true")
    p_eval.add_argument("--output", required=True)

    # report
    p_rep = subparsers.add_parser("report")
    p_rep.add_argument("--report", required=True)
    p_rep.add_argument("--dataset-root", required=True)
    p_rep.add_argument("--output", required=True)
    p_rep.add_argument("--examples-per-category", type=int, default=5)

    args = parser.parse_args()
    if args.command == "phase1-tracking-diagnosis":
        from src.services.evaluation.tracking_diagnosis import diagnose
        report = diagnose(args.protocol,args.root,args.output,final_held_out=args.final_held_out,dev_freeze=args.dev_freeze)
        print(json.dumps({"events":report["event_count"],"services_called":[]}))
    elif args.command == "phase1-tracking-freeze":
        from src.services.evaluation.tracking_protocol import freeze_dev
        report = freeze_dev(args.protocol,args.baseline,args.candidate,args.parameters,args.output)
        print(json.dumps({"schema":report["schema"],"services_called":[]}))
    elif args.command == "phase1-tracking-gate":
        from src.services.evaluation.tracking_protocol import assess_gate, verify_gate_reports
        config = json.loads(args.protocol.read_text(encoding="utf-8"))
        reports = [json.loads(path.read_text(encoding="utf-8")) if path else None
                   for path in (args.baseline,args.candidate,args.held_baseline,args.held_candidate)]
        authorized = verify_gate_reports(args.protocol,args.dev_freeze,
                                         [args.baseline,args.candidate,args.held_baseline,args.held_candidate])
        report = assess_gate(config,*reports[:2],held_baseline=reports[2],held_candidate=reports[3],final_authorized=authorized)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
        print(json.dumps({"status":report["status"],"production_flag":False,"services_called":[]}))
    elif args.command == "export-experiments":
        from src.services.evaluation.export_experiments import export_experiments
        report = export_experiments(args.protocol,args.root,args.output)
        print(json.dumps({"runs":len(report["runs"]),"services_called":[]}))
    elif args.command == "phase0":
        report = write_baseline(args.protocol, root=args.root, output=args.output)
        print(json.dumps({"output": str(args.output), "runs": len(report["runs"]), "services_called": []}))
    elif args.command == "phase0-misses":
        report = write_miss_analysis(args.protocol, root=args.root, output=args.output, seed=args.seed,
                                     bootstrap_repeats=args.bootstrap_repeats, image_count=args.image_count)
        print(json.dumps({"output": str(args.output), "population": report["population"], "services_called": []}))
    elif args.command == "phase0-geometry":
        report = write_geometry_analysis(args.protocol, misses_csv=args.misses, root=args.root,
                                         output=args.output, seed=args.seed)
        print(json.dumps({"output": str(args.output), "population": report["population"],
                          "geometry_counts": report["geometry_counts"], "services_called": []}))
    elif args.command == "phase0-temporal":
        report = write_temporal_analysis(args.protocol, geometry_csv=args.geometry, root=args.root,
                                         output=args.output, bootstrap_repeats=args.bootstrap_repeats)
        print(json.dumps({"output": str(args.output), "population": report["population"], "services_called": []}))
    elif args.command == "manual":
        cmd_manual(args)
    elif args.command == "compare-phase0":
        report = compare_baselines(args.before, args.after)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"inputs_unchanged": report["inputs_unchanged"], "metrics_unchanged": report["metrics_unchanged"]}))
    elif args.command == "prepare-reference":
        cmd_prepare_reference(args)
    elif args.command == "build-bindings":
        cmd_build_bindings(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "report":
        cmd_report(args)


def cmd_manual(args: argparse.Namespace) -> None:
    from scripts.eval_autolabel import Box, read_boxes, read_state_tracks

    reference, occlusions = load_manual_reference(args.ground_truth)
    if args.predictions.is_dir() or args.predictions.suffix.lower() == ".xml":
        document = None
    else:
        document = json.loads(args.predictions.read_text(encoding="utf-8"))
    if document is None or "predictions" not in document:
        videos = {x.key.sequence.sequence_uid for x in reference.frames}
        if len(videos) != 1:
            raise ValueError("CVAT/COCO/job predictions need a single-video GT or explicit video_id predictions")
        boxes = read_state_tracks(args.predictions)[0] if args.predictions.is_dir() else read_boxes(args.predictions)
        video_id = next(iter(videos))
        annotated = {x.source_ordinal for x in reference.frames}
        document = {"predictions": [{"video_id": video_id, "frame": x.frame, "track_id": x.track_id,
                                     "class": x.label, "box": x.xyxy} for x in boxes if x.frame in annotated]}
    index = {(x.key.sequence.sequence_uid, x.source_ordinal): x.key for x in reference.frames}
    predictions = []
    available = True
    for i, row in enumerate(document["predictions"]):
        key = index.get((row["video_id"], row["frame"]))
        if key is None:
            raise ValueError("prediction is outside explicitly annotated frames")
        available &= row.get("track_id") is not None
        local = canonical_predictions([Box(row["frame"], row["class"], tuple(row["box"]), str(row["track_id"]) if row.get("track_id") is not None else None)],
                                      {row["frame"]: key}, "manual")
        predictions.append(local[0].model_copy(update={"prediction_id": f"manual/{i}", "scoring_ordinal": i}))
    report = tracking_diagnostics(reference.boxes, tuple(predictions), frames=tuple(x.key for x in reference.frames),
                                  occlusions=occlusions, tracking_ids_available=available)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "annotated_frames": len(reference.frames)}))


if __name__ == "__main__":
    main()
