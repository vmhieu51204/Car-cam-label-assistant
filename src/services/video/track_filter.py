"""Reversible export triage. These thresholds are uncalibrated, not truth labels."""

from enum import StrEnum

from pydantic import Field, model_validator

from src.models.video import StrictContract, VideoTrack


class SuspectReason(StrEnum):
    TOO_SHORT = "too_short"
    TINY_BOX = "tiny_box"
    LOW_CONFIDENCE_LABEL = "low_confidence_label"
    IN_IGNORE_REGION = "in_ignore_region"
    EDGE_TRUNCATED = "edge_truncated"


class ClassFilter(StrictContract):
    min_frames: int = Field(default=3, ge=1, strict=True)
    min_area_px: float = Field(default=144, ge=0, allow_inf_nan=False)
    min_height_px: float = Field(default=12, ge=0, allow_inf_nan=False)
    # Scores are creation scores in older artifacts, not calibrated per-frame probabilities.
    min_score: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)


class TrackFilterConfig(StrictContract):
    enabled: bool = False
    defaults: ClassFilter = Field(default_factory=ClassFilter)
    classes: dict[str, ClassFilter] = Field(default_factory=dict)
    ignore_regions: list[tuple[float, float, float, float]] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_regions(self) -> "TrackFilterConfig":
        for x1, y1, x2, y2 in self.ignore_regions:
            if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                raise ValueError("ignore regions must be normalized, ordered rectangles")
        return self


def classify_tracks(
    tracks: list[VideoTrack],
    config: TrackFilterConfig,
    *,
    width: int,
    height: int,
) -> dict[str, list[str]]:
    """Return reasons only. Never remove or mutate tracks; human seeds are untouched."""
    if width <= 0 or height <= 0:
        raise ValueError("positive image dimensions required")
    if not config.enabled:
        return {}
    result = {}
    for track in tracks:
        if track.source != "auto":
            continue
        rule = config.classes.get(track.label_name, config.defaults)
        visible = [o for o in track.observations if not o.outside]
        reasons = []
        if len(visible) < rule.min_frames:
            reasons.append(SuspectReason.TOO_SHORT.value)
        if visible and all(
            (o.bbox.xbr - o.bbox.xtl) * (o.bbox.ybr - o.bbox.ytl) < rule.min_area_px
            or o.bbox.ybr - o.bbox.ytl < rule.min_height_px
            for o in visible
        ):
            reasons.append(SuspectReason.TINY_BOX.value)
        scores = [o.score for o in visible if o.score is not None]
        if rule.min_score is not None and scores and max(scores) < rule.min_score:
            reasons.append(SuspectReason.LOW_CONFIDENCE_LABEL.value)
        if visible and any(
            all(
                x1 <= o.bbox.xtl / width
                and y1 <= o.bbox.ytl / height
                and o.bbox.xbr / width <= x2
                and o.bbox.ybr / height <= y2
                for o in visible
            )
            for x1, y1, x2, y2 in config.ignore_regions
        ):
            reasons.append(SuspectReason.IN_IGNORE_REGION.value)
        if visible and all(o.truncated for o in visible):
            reasons.append(SuspectReason.EDGE_TRUNCATED.value)
        if reasons:
            result[track.track_id] = reasons
    return result
