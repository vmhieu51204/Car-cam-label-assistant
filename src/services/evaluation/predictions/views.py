"""Offline geometry views using the existing stitcher and interpolation policies.

Producer ordinals remain evidence; final per-frame ordinals are assigned only after
overlap selection. No score is inferred by either identity or geometry matching.
"""

from __future__ import annotations

import io
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Any

from scripts.compare_video_optimization import interpolate_tracks
from src.models.evaluation import (
    BBox,
    FrameBinding,
    FrameKey,
    PredictionBox,
    PredictionNormalizationOptions,
    Score,
    TrackStateObservation,
)
from src.services.evaluation.legacy import read_cvat_xml
from src.services.video.stitching import ChunkObservation, ChunkTrack, StitchConfig, UnifiedTrack, stitch_tracks


def _replace(box: PredictionBox, **changes: Any) -> PredictionBox:
    return PredictionBox.model_validate({**box.model_dump(), **changes})


def scoring_ordinals(boxes: list[PredictionBox]) -> tuple[PredictionBox, ...]:
    counts: dict[FrameKey, int] = defaultdict(int)
    result = []
    for box in boxes:
        result.append(_replace(box, scoring_ordinal=counts[box.frame]))
        counts[box.frame] += 1
    return tuple(result)


def geometry_views(
    boxes: tuple[PredictionBox, ...],
    *,
    options: PredictionNormalizationOptions,
    frame_positions: dict[FrameKey, int],
    cuts: dict[FrameKey, str | None],
    masks: dict[str, Any],
) -> tuple[tuple[PredictionBox, ...], tuple[PredictionBox, ...]]:
    """Associate only inside a sequence/cut partition with the shared stitcher.

    Raw overlap retains the first window as a complete frame, including its empty
    result. The caller additionally removes rows superseded by explicit empty
    first-window evidence. Stitched geometry is selected per stitched identity.
    """
    by_sequence: dict[Any, list[PredictionBox]] = defaultdict(list)
    for box in boxes:
        by_sequence[box.frame.sequence].append(box)
    raw: list[PredictionBox] = []
    stitched: list[PredictionBox] = []
    for sequence_index, (sequence, sequence_boxes) in enumerate(by_sequence.items()):
        boundaries = sorted(frame_positions[frame] for frame in cuts if frame.sequence == sequence)
        parts: dict[int, list[PredictionBox]] = defaultdict(list)
        for box in sequence_boxes:
            segment = sum(boundary <= frame_positions[box.frame] for boundary in boundaries)
            parts[segment].append(box)
        for segment, rows in parts.items():
            window_names = list(dict.fromkeys(box.source.window_id or "unwindowed" for box in rows))
            # Caller preserves producer window order; window IDs need not sort lexically.
            windows = {name: index for index, name in enumerate(window_names)}
            first: dict[FrameKey, int] = {}
            for box in rows:
                window = windows[box.source.window_id or "unwindowed"]
                first[box.frame] = min(first.get(box.frame, window), window)
            raw.extend(box for box in rows if windows[box.source.window_id or "unwindowed"] == first[box.frame])
            grouped: dict[tuple[int, str, str], list[PredictionBox]] = defaultdict(list)
            for box in rows:
                grouped[(windows[box.source.window_id or "unwindowed"], box.raw_track_id, box.class_name)].append(box)
            chunks = [
                ChunkTrack(
                    window,
                    identity,
                    label,
                    [
                        ChunkObservation(
                            frame_positions[box.frame], box.bbox_xyxy, masks.get(box.prediction_id), box.score.value
                        )
                        for box in sorted(observations, key=lambda box: frame_positions[box.frame])
                    ],
                )
                for (window, identity, label), observations in grouped.items()
            ]
            result = stitch_tracks(
                chunks,
                StitchConfig(
                    options.stitch.min_overlap_iou, options.stitch.ambiguity_margin, options.stitch.min_common_frames
                ),
            )
            # Select a source-observation sidecar, not a score synthesized from the
            # merged object. Every output observation has exactly one source row.
            assignments: dict[tuple[int, str], str] = {}
            for decision in result.decisions:
                assignments[(decision.window_id, decision.local_track_id)] = decision.global_track_id
            initial = [chunk for chunk in chunks if chunk.window_id == min(windows.values())]
            for chunk, track in zip(initial, result.tracks, strict=False):
                assignments[(chunk.window_id, chunk.local_track_id)] = track.track_id
            chosen: dict[tuple[str, FrameKey], PredictionBox] = {}
            geometry = (
                options.identity_geometry_view
                if options.stitched_overlap_geometry == "identity_only"
                else "last_window"
            )
            for box in rows:
                window = windows[box.source.window_id or "unwindowed"]
                identity = assignments[(window, box.raw_track_id)]
                key = (identity, box.frame)
                if geometry == "last_window" or key not in chosen:
                    chosen[key] = _replace(box, stitched_track_id=f"s{sequence_index}/cut{segment}/{identity}")
            stitched.extend(chosen.values())
    return scoring_ordinals(raw), scoring_ordinals(stitched)


def interpolate_view(
    boxes: tuple[PredictionBox, ...],
    states: tuple[TrackStateObservation, ...],
    *,
    options: PredictionNormalizationOptions,
    bindings: tuple[FrameBinding, ...],
    selected_frames: set[FrameKey],
    cuts: dict[FrameKey, str | None],
) -> tuple[PredictionBox, ...]:
    """Apply a named existing algorithm on an explicitly bound native timeline.

    Native-export indices are required even for token datasets. Outside rows split
    intervals, as do declared cuts and configured maximum gaps. Interpolation never
    changes processing evidence and never manufactures confidence.
    """
    native: dict[FrameKey, int] = {}
    for binding in bindings:
        for step in (binding, *binding.upstream):
            if step.domain == "native_export":
                if binding.frame in native and native[binding.frame] != step.index:
                    raise ValueError("ambiguous interpolation timeline")
                native[binding.frame] = step.index
    needed = {box.frame for box in boxes} | selected_frames
    if not needed <= native.keys():
        raise ValueError("interpolation requires exhaustive explicit native-export bindings")
    grouped: dict[tuple[Any, str], list[PredictionBox]] = defaultdict(list)
    for box in boxes:
        grouped[(box.frame.sequence, box.stitched_track_id or box.raw_track_id)].append(box)
    output = list(boxes)
    for (sequence, identity), track_boxes in grouped.items():
        timeline = {index: frame for frame, index in native.items() if frame.sequence == sequence}
        if len(timeline) != sum(frame.sequence == sequence for frame in native):
            raise ValueError("native-export mapping is not bijective")
        ordered = sorted(track_boxes, key=lambda box: native[box.frame])
        # Explicit outside evidence and cuts terminate intervals even when absent
        # from the selected-frame list.
        stops = sorted(
            {native[frame] for frame in cuts if frame.sequence == sequence}
            | {
                native[state.frame]
                for state in states
                if state.frame.sequence == sequence
                and state.outside
                and state.track.track_id == ordered[0].raw_track_id
            }
        )
        partitions: list[list[PredictionBox]] = []
        for box in ordered:
            if not partitions:
                partitions.append([box])
                continue
            previous = partitions[-1][-1]
            gap = native[box.frame] - native[previous.frame]
            if any(native[previous.frame] < stop <= native[box.frame] for stop in stops) or (
                options.interpolation.max_gap_source_frames is not None
                and gap > options.interpolation.max_gap_source_frames
            ):
                partitions.append([box])
            else:
                partitions[-1].append(box)
        for part in partitions:
            first, final = native[part[0].frame], native[part[-1].frame]
            following_stop = next((stop for stop in stops if stop > final), None)
            endpoint = final
            if options.interpolation.endpoint_policy == "legacy_final_extent":
                endpoint = max(timeline)
            if following_stop is not None:
                endpoint = min(endpoint, following_stop - 1)
            eligible = {index for index in timeline if first <= index <= endpoint}
            if options.interpolation.algorithm == "video_optimization_linear":
                track = UnifiedTrack(
                    identity, part[0].class_name, [ChunkObservation(native[box.frame], box.bbox_xyxy) for box in part]
                )
                selected = sorted(native[frame] for frame in selected_frames if frame.sequence == sequence)
                generated = interpolate_tracks([track], selected, eligible)
            else:
                root = ET.Element("annotations")
                ET.SubElement(root, "version").text = "1.1"
                track_element = ET.SubElement(root, "track", id=identity, label=part[0].class_name)
                for box in part:
                    _xml_box(track_element, native[box.frame], box.bbox_xyxy, False)
                if following_stop is not None:
                    _xml_box(track_element, following_stop, part[-1].bbox_xyxy, True)
                elif endpoint > final:
                    _xml_box(track_element, endpoint, part[-1].bbox_xyxy, False)
                generated = read_cvat_xml(io.BytesIO(ET.tostring(root)))  # type: ignore[arg-type]
            observed = {native[box.frame] for box in part}
            for generated_box in generated:
                if generated_box.frame in observed or generated_box.frame not in eligible:
                    continue
                left = max(
                    (box for box in part if native[box.frame] <= generated_box.frame), key=lambda box: native[box.frame]
                )
                output.append(
                    _replace(
                        left,
                        prediction_id=f"{left.prediction_id}/interpolated/{generated_box.frame}",
                        frame=timeline[generated_box.frame],
                        bbox_xyxy=generated_box.xyxy,
                        origin="interpolated",
                        score=Score(value=None, semantics="missing", source="interpolation-no-confidence"),
                        mask=None,
                        interpolation_revision=options.interpolation.revision,
                    )
                )
    return scoring_ordinals(output)


def _xml_box(parent: ET.Element, frame: int, box: BBox, outside: bool) -> None:
    attrib = {
        "frame": str(frame),
        "outside": str(int(outside)),
        **dict(zip(("xtl", "ytl", "xbr", "ybr"), map(str, box), strict=True)),
    }
    ET.SubElement(parent, "box", attrib=attrib)
