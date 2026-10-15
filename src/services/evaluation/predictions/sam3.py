"""Normalization adapter for existing SAM3 benchmark, window, and track results."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

from src.models.evaluation import (
    BBox,
    BenchmarkManifest,
    FrameBinding,
    FrameKey,
    FrameProcessing,
    MaskReference,
    ObservationSource,
    PayloadSeed,
    PredictionBox,
    PredictionNormalizationOptions,
    PredictionRun,
    RunProvenance,
    RunStatus,
    RuntimeRecord,
    Score,
    TrackKey,
    TrackStateObservation,
    WindowOutcome,
    content_hash,
)
from src.services.evaluation.predictions.views import geometry_views, interpolate_view


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_json_bytes(data: bytes) -> dict[str, Any]:
    parsed = json.loads(data.decode("utf-8"), object_pairs_hook=_pairs)
    if not isinstance(parsed, dict):
        raise ValueError("prediction result must be a JSON object")
    return parsed


def _validate_rle(mask_rle: dict[str, Any]) -> tuple[int, int, str]:
    if not isinstance(mask_rle, dict):
        raise ValueError("invalid mask RLE record")
    width = mask_rle.get("width")
    height = mask_rle.get("height")
    counts = mask_rle.get("counts")
    if (
        type(width) is not int
        or type(height) is not int
        or width <= 0
        or height <= 0
        or not isinstance(counts, list)
        or any(type(x) is not int or x < 0 for x in counts)
        or sum(counts) != width * height
    ):
        raise ValueError("corrupt MaskRLE")
    digest = hashlib.sha256(
        json.dumps({"counts": counts, "height": height, "width": width}, sort_keys=True).encode()
    ).hexdigest()
    return width, height, digest


def _resolve_bindings(
    bindings: tuple[FrameBinding, ...],
) -> tuple[dict[tuple[str, int], FrameKey], dict[tuple[str, str, int], FrameKey], dict[int, FrameKey]]:
    by_domain: dict[tuple[str, int], FrameKey] = {}
    by_window: dict[tuple[str, str, int], FrameKey] = {}
    by_payload_index: dict[int, FrameKey] = {}
    for binding in bindings:
        steps = (binding, *binding.upstream)
        for step in steps:
            if step.domain == "window_local" and step.config_id and step.window_id:
                by_window[(step.config_id, step.window_id, step.index)] = binding.frame
            else:
                by_domain[(step.domain, step.index)] = binding.frame
            if step.domain in ("payload", "selected") or step.mapping_id.startswith("payload"):
                by_payload_index[step.index] = binding.frame
    return by_domain, by_window, by_payload_index


def _load_payload_metadata(
    payload: Path | None,
    manifest: BenchmarkManifest,
) -> tuple[str | None, str | None, tuple[PayloadSeed, ...], set[int]]:
    if payload is None:
        return None, None, (), set()
    if not payload.is_file():
        raise FileNotFoundError(f"payload file not found: {payload}")
    payload_bytes = payload.read_bytes()
    payload_sha256 = hashlib.sha256(payload_bytes).hexdigest()
    config_sha256: str | None = None
    seeds: list[PayloadSeed] = []
    packaged_source_indices: set[int] = set()

    if zipfile.is_zipfile(payload):
        with zipfile.ZipFile(payload, "r") as archive:
            names = set(archive.namelist())
            if "benchmark-config.json" in names:
                config_bytes = archive.read("benchmark-config.json")
                config_sha256 = hashlib.sha256(config_bytes).hexdigest()
            if "frames.json" in names:
                frames_doc = json.loads(archive.read("frames.json").decode("utf-8"))
                for frame_info in frames_doc.get("frames", []):
                    if "original_source_index" in frame_info:
                        packaged_source_indices.add(int(frame_info["original_source_index"]))
            if "seed.json" in names:
                seed_doc = json.loads(archive.read("seed.json").decode("utf-8"))
                for idx, item in enumerate(seed_doc.get("seeds", [])):
                    model_seed_id = str(item.get("track_id", item.get("seed_id", idx)))
                    frame_idx = item.get("frame_index", item.get("source_frame_index", 0))
                    # Find matching binding for seed frame
                    matched_frame = None
                    for b in manifest.bindings:
                        if b.index == frame_idx and (
                            b.domain in ("payload", "selected") or b.mapping_id.startswith("payload")
                        ):
                            matched_frame = b.frame
                            break
                    if matched_frame is None and manifest.planned_frames:
                        matched_frame = manifest.planned_frames[0]
                    if matched_frame is not None:
                        bbox = tuple(float(x) for x in item["bbox"])
                        seeds.append(
                            PayloadSeed(
                                seed_id=f"seed-{idx}",
                                model_seed_id=model_seed_id,
                                frame=matched_frame,
                                class_name=item.get("label", item.get("category", "car")),
                                bbox_xyxy=(bbox[0], bbox[1], bbox[2], bbox[3]),
                                mapping_id=manifest.bindings[0].mapping_id if manifest.bindings else "payload/v1",
                            )
                        )
    return payload_sha256, config_sha256, tuple(seeds), packaged_source_indices


def load_sam3_predictions(
    result: Path,
    *,
    manifest: BenchmarkManifest,
    payload: Path | None = None,
    options: PredictionNormalizationOptions,
) -> tuple[PredictionRun, ...]:
    """Offline adapter normalizing benchmark, window, and track results to PredictionRun."""
    if not result.is_file():
        raise FileNotFoundError(f"result file not found: {result}")
    result_bytes = result.read_bytes()
    result_sha256 = hashlib.sha256(result_bytes).hexdigest()
    data = _parse_json_bytes(result_bytes)
    schema = data.get("schema")

    payload_sha256, config_sha256, payload_seeds, packaged_indices = _load_payload_metadata(payload, manifest)
    if config_sha256 is None:
        config_sha256 = hashlib.sha256(
            json.dumps(manifest.normalization.model_dump(), sort_keys=True).encode()
        ).hexdigest()

    # Verify endpoint completeness if payload contains native endpoints
    if packaged_indices:
        binding_native_indices = {
            step.index for b in manifest.bindings for step in (b, *b.upstream) if step.domain == "native_export"
        }
        if binding_native_indices and not packaged_indices <= binding_native_indices:
            raise ValueError("payload frames contain unrepresented endpoint or source indices in bindings")

    by_domain, by_window, by_payload_index = _resolve_bindings(manifest.bindings)

    # Helper to resolve frame key
    def get_frame(
        index: int, domain: str = "payload", config_id: str | None = None, window_id: str | None = None
    ) -> tuple[FrameKey, str]:
        if config_id and window_id and (config_id, window_id, index) in by_window:
            return by_window[(config_id, window_id, index)], "window_local/v1"
        if (domain, index) in by_domain:
            return by_domain[(domain, index)], f"{domain}/v1"
        if index in by_payload_index:
            return by_payload_index[index], "payload/v1"
        raise ValueError(f"unknown frame index {index} in domain {domain}")

    # Build frame timeline positions for sequence-based geometry views
    frame_positions: dict[FrameKey, int] = {}
    for binding in manifest.bindings:
        for step in (binding, *binding.upstream):
            if step.domain == "native_export":
                frame_positions[binding.frame] = step.index
    if len(frame_positions) != len(manifest.planned_frames):
        for i, frame in enumerate(manifest.planned_frames):
            if frame not in frame_positions:
                frame_positions[frame] = i

    runs: list[PredictionRun] = []

    # Case A: autolabel-sam3-benchmark/v1
    if schema == "autolabel-sam3-benchmark/v1":
        configurations = data.get("configurations")
        if not isinstance(configurations, list) or not configurations:
            raise ValueError("benchmark result contains no configurations")

        for item in configurations:
            config_id = item["id"]
            raw_status_str = item.get("status", "completed")
            if raw_status_str not in (
                "completed",
                "failed",
                "timed_out",
                "unsupported",
                "skipped_budget",
                "pending",
                "invalid_input",
            ):
                raise ValueError(f"unknown config status: {raw_status_str}")
            raw_status: RunStatus = raw_status_str
            status: RunStatus = raw_status

            failure_reason = item.get("error")
            if status in ("failed", "timed_out", "unsupported", "invalid_input") and not failure_reason:
                failure_reason = "execution terminated with error"

            # Parse observations
            observations = item.get("window_predictions") or item.get("predictions") or []
            selected_indices = item.get("selected_source_frames", [])
            evaluated_indices = item.get("evaluated_source_frames", [])

            # Track processing records
            evaluated_frames: set[FrameKey] = set()
            for idx in evaluated_indices:
                f_key, m_id = get_frame(idx)
                evaluated_frames.add(f_key)

            selected_frames: set[FrameKey] = set()
            for idx in selected_indices:
                f_key, m_id = get_frame(idx)
                selected_frames.add(f_key)

            # If no explicit evaluated list but completed empty
            if not evaluated_frames and status == "completed":
                for frame in manifest.planned_frames:
                    evaluated_frames.add(frame)
                    selected_frames.add(frame)

            processing: list[FrameProcessing] = []
            for frame in manifest.planned_frames:
                if frame in evaluated_frames:
                    processing.append(
                        FrameProcessing(
                            frame=frame,
                            selected=True,
                            state="processed",
                            evidence="wrapper_declared",
                            mapping_id="payload/v1",
                        )
                    )
                elif frame in selected_frames:
                    processing.append(
                        FrameProcessing(
                            frame=frame,
                            selected=True,
                            state="unknown",
                            evidence="unavailable",
                            mapping_id="payload/v1",
                        )
                    )
                else:
                    processing.append(
                        FrameProcessing(
                            frame=frame,
                            selected=False,
                            state="not_processed",
                            evidence="unavailable",
                            mapping_id="payload/v1",
                        )
                    )

            candidate_boxes: list[PredictionBox] = []
            track_states: list[TrackStateObservation] = []
            masks_lookup: dict[str, Any] = {}

            for obs_idx, obs in enumerate(observations):
                frame_idx = obs.get("frame_index", obs.get("frame"))
                if frame_idx is None:
                    raise ValueError(f"observation {obs_idx} lacks frame_index")
                window_id = str(obs["window_id"]) if "window_id" in obs and obs["window_id"] is not None else None
                frame_key, mapping_id = get_frame(
                    int(frame_idx), domain="payload", config_id=config_id, window_id=window_id
                )

                label = obs.get("label", obs.get("category", obs.get("class_name")))
                if not isinstance(label, str):
                    raise ValueError(f"observation {obs_idx} missing class label")
                if label not in manifest.profile.classes:
                    if options.unknown_class_policy == "reject":
                        raise ValueError(f"unknown prediction class: {label}")
                    continue

                raw_box = obs.get("bbox", obs.get("bbox_xyxy"))
                if not raw_box or len(raw_box) != 4:
                    raise ValueError(f"observation {obs_idx} invalid bbox")
                box_xyxy: BBox = (float(raw_box[0]), float(raw_box[1]), float(raw_box[2]), float(raw_box[3]))
                if box_xyxy[2] <= box_xyxy[0] or box_xyxy[3] <= box_xyxy[1]:
                    if options.bounds_policy == "reject":
                        raise ValueError("bbox must have positive continuous xyxy extents")

                raw_track_id = str(obs.get("track_id", "0"))
                source = ObservationSource(
                    mapping_id=mapping_id,
                    source_ordinal=obs_idx,
                    config_id=config_id,
                    source_observation_id=f"row-{obs_idx}" if window_id is None else f"w{window_id}-row-{obs_idx}",
                    window_id=window_id,
                )

                # Score semantics
                score_val = obs.get("score")
                if score_val is not None:
                    mode = item.get("config", {}).get("mode", "text_prompt")
                    semantics = "probability" if mode == "text_prompt" else "sigmoid_logit"
                    score = Score(value=float(score_val), semantics=semantics, source="model")
                else:
                    score = Score(value=None, semantics="missing", source="missing")

                mask_ref: MaskReference | None = None
                if "mask_rle" in obs and obs["mask_rle"]:
                    width, height, rle_hash = _validate_rle(obs["mask_rle"])
                    mask_ref = MaskReference(
                        relative_path=f"masks/mask_{obs_idx:06d}.rle",
                        sha256=rle_hash,
                        width=width,
                        height=height,
                    )
                    masks_lookup[f"pred-{config_id}-{obs_idx}"] = obs["mask_rle"]

                if obs.get("outside", False):
                    track_states.append(
                        TrackStateObservation(
                            observation_id=f"ts-{config_id}-{obs_idx}",
                            frame=frame_key,
                            track=TrackKey(
                                sequence=frame_key.sequence,
                                track_id=raw_track_id,
                                config_id=config_id,
                                window_id=window_id,
                            ),
                            class_name=label,
                            source=source,
                            outside=True,
                            score=score,
                        )
                    )
                else:
                    pred_id = f"pred-{config_id}-{obs_idx}"
                    candidate_boxes.append(
                        PredictionBox(
                            prediction_id=pred_id,
                            frame=frame_key,
                            class_name=label,
                            bbox_xyxy=box_xyxy,
                            raw_track_id=raw_track_id,
                            source=source,
                            scoring_ordinal=0,
                            score=score,
                            origin="model",
                            mask=mask_ref,
                        )
                    )

            windows: list[WindowOutcome] = []
            for w in item.get("window_checkpoints", []):
                windows.append(
                    WindowOutcome(
                        window_id=str(w["window_id"]),
                        status="completed",
                        checkpoint_sha256=w.get("output_sha256"),
                        resumed=w.get("window_id") in item.get("resumed_windows", []),
                    )
                )

            # Runtimes
            runtime_info = item.get("runtime", {})
            runtime = RuntimeRecord(
                inference_s=runtime_info.get("total_seconds"),
                hardware=runtime_info.get("device"),
            )

            cuts: dict[FrameKey, str | None] = {
                b.frame: b.source for b in options.reset_boundaries if b.config_id == config_id
            }

            raw_boxes, stitched_boxes = geometry_views(
                tuple(candidate_boxes),
                options=options,
                frame_positions=frame_positions,
                cuts=cuts,
                masks=masks_lookup,
            )

            provenance = RunProvenance(
                result_sha256=result_sha256,
                payload_sha256=payload_sha256,
                config_sha256=config_sha256,
                model_revision=str(runtime_info.get("model_revision", "sam3/v1")),
                evaluator_revision=manifest.evaluator_revision,
                inference_variant="text",
            )

            for requested_view in options.views:
                if requested_view == "raw":
                    boxes_view = raw_boxes
                elif requested_view == "stitched":
                    boxes_view = stitched_boxes
                elif requested_view == "interpolated":
                    boxes_view = interpolate_view(
                        stitched_boxes,
                        tuple(track_states),
                        options=options,
                        bindings=manifest.bindings,
                        selected_frames=selected_frames,
                        cuts=cuts,
                    )
                else:
                    raise ValueError(f"unsupported view {requested_view}")

                run = PredictionRun(
                    config_id=config_id,
                    view=requested_view,
                    status=status,
                    raw_status=raw_status,
                    processing=tuple(processing),
                    boxes=boxes_view,
                    track_states=tuple(track_states),
                    payload_seeds=payload_seeds,
                    normalization=options,
                    normalization_sha256=content_hash(options),
                    provenance=provenance,
                    windows=tuple(windows),
                    runtime=runtime,
                    failure_reason=failure_reason,
                )
                runs.append(PredictionRun.model_validate({**run.model_dump(), "content_sha256": content_hash(run)}))

    # Case B: autolabel-stitch-comparison/v1
    elif schema == "autolabel-stitch-comparison/v1":
        config_id = data.get("config_id", "stitch-comparison")
        candidate_boxes = []
        obs_counter = 0
        for track in data.get("tracks", []):
            label = track.get("category", track.get("label", "car"))
            for obs in track.get("observations", []):
                frame_idx = obs["frame"]
                frame_key, mapping_id = get_frame(int(frame_idx), domain="payload", config_id=config_id)
                raw_box = obs["bbox"]
                box_xyxy = (float(raw_box[0]), float(raw_box[1]), float(raw_box[2]), float(raw_box[3]))
                score_val = obs.get("score")
                score = (
                    Score(value=float(score_val), semantics="probability", source="model")
                    if score_val is not None
                    else Score(value=None, semantics="missing", source="missing")
                )
                source = ObservationSource(
                    mapping_id=mapping_id,
                    source_ordinal=obs_counter,
                    config_id=config_id,
                    source_observation_id=f"row-{obs_counter}",
                )
                candidate_boxes.append(
                    PredictionBox(
                        prediction_id=f"pred-{config_id}-{obs_counter}",
                        frame=frame_key,
                        class_name=label,
                        bbox_xyxy=box_xyxy,
                        raw_track_id=str(track.get("track_id", "0")),
                        stitched_track_id=str(track.get("track_id")),
                        source=source,
                        scoring_ordinal=0,
                        score=score,
                        origin="model",
                    )
                )
                obs_counter += 1

        proc_tuple = tuple(
            FrameProcessing(
                frame=frame,
                selected=True,
                state="processed",
                evidence="wrapper_declared",
                mapping_id="payload/v1",
            )
            for frame in manifest.planned_frames
        )
        cuts = {b.frame: b.source for b in options.reset_boundaries if b.config_id == config_id}
        raw_boxes, stitched_boxes = geometry_views(
            tuple(candidate_boxes),
            options=options,
            frame_positions=frame_positions,
            cuts=cuts,
            masks={},
        )
        provenance = RunProvenance(
            result_sha256=result_sha256,
            payload_sha256=payload_sha256,
            config_sha256=config_sha256,
            model_revision="sam3/v1",
            evaluator_revision=manifest.evaluator_revision,
            inference_variant="text",
        )
        for requested_view in options.views:
            boxes_view = raw_boxes if requested_view == "raw" else stitched_boxes
            run = PredictionRun(
                config_id=config_id,
                view=requested_view,
                status="completed",
                raw_status="completed",
                processing=proc_tuple,
                boxes=boxes_view,
                normalization=options,
                normalization_sha256=content_hash(options),
                provenance=provenance,
            )
            runs.append(PredictionRun.model_validate({**run.model_dump(), "content_sha256": content_hash(run)}))

    # Case C: autolabel-track-set/v1
    elif schema == "autolabel-track-set/v1" or "track_set" in data:
        track_set_doc = data.get("track_set", data)
        config_id = "track-set"
        candidate_boxes = []
        obs_counter = 0
        for track in track_set_doc.get("tracks", []):
            label = track.get("label_name", "car")
            for obs in track.get("observations", []):
                frame_idx = obs["frame_index"]
                frame_key, mapping_id = get_frame(int(frame_idx), domain="payload", config_id=config_id)
                raw_box = obs["bbox"]
                box_xyxy = (float(raw_box[0]), float(raw_box[1]), float(raw_box[2]), float(raw_box[3]))
                score_val = obs.get("score")
                score = (
                    Score(value=float(score_val), semantics="probability", source="model")
                    if score_val is not None
                    else Score(value=1.0, semantics="fixed_visible", source="inherited_export")
                )
                source = ObservationSource(
                    mapping_id=mapping_id,
                    source_ordinal=obs_counter,
                    config_id=config_id,
                    source_observation_id=f"row-{obs_counter}",
                )
                candidate_boxes.append(
                    PredictionBox(
                        prediction_id=f"pred-{config_id}-{obs_counter}",
                        frame=frame_key,
                        class_name=label,
                        bbox_xyxy=box_xyxy,
                        raw_track_id=str(track.get("track_id", "0")),
                        source=source,
                        scoring_ordinal=0,
                        score=score,
                        origin="model",
                    )
                )
                obs_counter += 1

        proc_tuple = tuple(
            FrameProcessing(
                frame=frame,
                selected=True,
                state="processed",
                evidence="wrapper_declared",
                mapping_id="payload/v1",
            )
            for frame in manifest.planned_frames
        )
        cuts = {b.frame: b.source for b in options.reset_boundaries if b.config_id == config_id}
        raw_boxes, stitched_boxes = geometry_views(
            tuple(candidate_boxes),
            options=options,
            frame_positions=frame_positions,
            cuts=cuts,
            masks={},
        )
        provenance = RunProvenance(
            result_sha256=result_sha256,
            payload_sha256=payload_sha256,
            config_sha256=config_sha256,
            model_revision="sam3/v1",
            evaluator_revision=manifest.evaluator_revision,
            inference_variant="text",
        )
        for requested_view in options.views:
            boxes_view = raw_boxes if requested_view == "raw" else stitched_boxes
            run = PredictionRun(
                config_id=config_id,
                view=requested_view,
                status="completed",
                raw_status="completed",
                processing=proc_tuple,
                boxes=boxes_view,
                normalization=options,
                normalization_sha256=content_hash(options),
                provenance=provenance,
            )
            runs.append(PredictionRun.model_validate({**run.model_dump(), "content_sha256": content_hash(run)}))
    else:
        raise ValueError(f"unsupported prediction schema: {schema}")

    return tuple(runs)
