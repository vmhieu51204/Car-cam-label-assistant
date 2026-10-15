"""Provenance, source verification, and explicit frame binding composition."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from src.models.evaluation import (
    BindingOptions,
    BindingStep,
    DatasetReference,
    FrameBinding,
    FrameKey,
)


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_manifest_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_bytes(), object_pairs_hook=_pairs)
    if not isinstance(data, dict):
        raise ValueError("manifest must be a JSON object")
    return data


def build_frame_bindings(
    reference: DatasetReference,
    payload: Path,
    *,
    options: BindingOptions,
    source_manifests: tuple[Path, ...] = (),
    result: Path | None = None,
) -> tuple[FrameBinding, ...]:
    """Compose explicit multi-step bindings from payload/converted indices to native frames."""
    if not payload.is_file():
        raise FileNotFoundError(f"payload file not found: {payload}")

    ref_frames_by_index: dict[int, FrameKey] = {}
    for frame in reference.frames:
        if frame.key.sequence == options.sequence:
            ref_frames_by_index[frame.source_ordinal] = frame.key

    if not ref_frames_by_index:
        raise ValueError(f"reference contains no frames for sequence {options.sequence}")

    # Case A: sequence-native payload zip
    if options.input_kind == "sequence-native":
        if not zipfile.is_zipfile(payload):
            raise ValueError("sequence-native payload must be a zip file")

        with zipfile.ZipFile(payload, "r") as archive:
            if "frames.json" not in archive.namelist():
                raise ValueError("sequence-native payload missing frames.json")
            frames_data = json.loads(archive.read("frames.json").decode("utf-8"))

        records = frames_data.get("frames", [])
        if not records:
            raise ValueError("frames.json contains no frame records")

        bindings: list[FrameBinding] = []
        seen_native_indices: set[int] = set()

        for rec in records:
            payload_idx = rec["source_frame_index"]
            native_idx = rec.get("original_source_index", payload_idx)
            seen_native_indices.add(native_idx)

            if native_idx not in ref_frames_by_index:
                raise ValueError(f"payload native index {native_idx} not found in reference")

            frame_key = ref_frames_by_index[native_idx]
            upstream_step = BindingStep(
                mapping_id=f"native/{options.sequence.sequence_uid}/v1",
                domain="native_export",
                index=native_idx,
            )
            binding = FrameBinding(
                mapping_id=options.mapping_id,
                domain="payload",
                index=payload_idx,
                frame=frame_key,
                upstream=(upstream_step,),
            )
            bindings.append(binding)

        # Endpoint verification for sequence-native
        if options.endpoint_policy == "include_final_source_frame":
            final_ref_index = max(ref_frames_by_index.keys())
            if final_ref_index not in seen_native_indices:
                raise ValueError(
                    f"endpoint policy include_final_source_frame requires native index {final_ref_index} in payload"
                )

        return tuple(bindings)

    # Case B: public-converted (requires source manifests chain)
    elif options.input_kind == "public-converted":
        if not source_manifests:
            raise ValueError("public-converted requires at least one source manifest")

        # First manifest: converted manifest with frame_index -> source_frame_index
        conv_manifest = parse_manifest_json(source_manifests[0])
        conv_frames = conv_manifest.get("frames", [])
        conv_map = {row["frame_index"]: row["source_frame_index"] for row in conv_frames}

        # If a second manifest is provided: benchmark manifest
        if len(source_manifests) > 1:
            bench_manifest = parse_manifest_json(source_manifests[1])
            bench_frames = bench_manifest.get("frames", [])
            bench_map = {row["frame_index"]: row["source_frame_index"] for row in bench_frames}
        else:
            bench_map = {idx: idx for idx in conv_map}

        bindings = []
        for bench_idx, conv_idx in bench_map.items():
            if conv_idx not in conv_map:
                raise ValueError(f"benchmark index {bench_idx} maps to unknown converted index {conv_idx}")
            native_idx = conv_map[conv_idx]
            if native_idx not in ref_frames_by_index:
                raise ValueError(f"converted index maps to unknown native index {native_idx}")

            frame_key = ref_frames_by_index[native_idx]
            native_step = BindingStep(
                mapping_id=f"native/{options.sequence.sequence_uid}/v1",
                domain="native_export",
                index=native_idx,
            )
            conv_step = BindingStep(
                mapping_id="public-converted/v1",
                domain="converted",
                index=conv_idx,
            )
            binding = FrameBinding(
                mapping_id=options.mapping_id,
                domain="payload",
                index=bench_idx,
                frame=frame_key,
                upstream=(native_step, conv_step),
            )
            bindings.append(binding)

        return tuple(bindings)

    # Case C: nuScenes native
    elif options.input_kind == "nuscenes-native":
        bindings = []
        for ordinal, key in sorted(ref_frames_by_index.items()):
            native_step = BindingStep(
                mapping_id=f"native/{options.sequence.sequence_uid}/v1",
                domain="native_export",
                index=ordinal,
            )
            binding = FrameBinding(
                mapping_id=options.mapping_id,
                domain="payload",
                index=ordinal,
                frame=key,
                upstream=(native_step,),
            )
            bindings.append(binding)
        return tuple(bindings)

    # Case D: COCO explicit
    elif options.input_kind == "coco-explicit":
        bindings = []
        for ordinal, key in sorted(ref_frames_by_index.items()):
            native_step = BindingStep(
                mapping_id="coco-native/v1",
                domain="native_export",
                index=ordinal,
            )
            binding = FrameBinding(
                mapping_id=options.mapping_id,
                domain="payload",
                index=ordinal,
                frame=key,
                upstream=(native_step,),
            )
            bindings.append(binding)
        return tuple(bindings)

    raise ValueError(f"unsupported binding input_kind: {options.input_kind}")
