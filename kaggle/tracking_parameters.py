"""Read scalar tracking configuration only; never copy state/memory or set defaults."""

import math
from typing import Any

VIDEO_FIELDS = (
    "score_threshold_detection", "det_nms_thresh", "new_det_thresh", "assoc_iou_thresh",
    "hotstart_delay", "hotstart_unmatch_thresh", "hotstart_dup_thresh",
    "recondition_every_nth_frame", "init_trk_keep_alive", "max_trk_keep_alive",
    "min_trk_keep_alive", "decrease_keep_alive_for_empty_masklets",
)
MEMORY_FIELDS = (
    "num_maskmem", "use_memory_selection", "mf_threshold", "memory_temporal_stride_for_eval", "max_cond_frames_in_attn",
    "max_obj_ptrs_in_encoder", "trim_past_non_cond_mem_for_eval",
)


def tracking_parameters(model: Any) -> dict:
    """Paths are explicit. Missing/unsupported attributes stay unavailable, never inferred."""
    predictor = getattr(model, "predictor", None)
    owner = getattr(predictor, "model", model) if predictor is not None else model
    tracker = getattr(owner, "tracker", None)
    result = {}
    for prefix, obj, fields in (("model", owner, VIDEO_FIELDS), ("model.tracker", tracker, MEMORY_FIELDS)):
        for name in fields:
            value = getattr(obj, name, None) if obj is not None else None
            valid = type(value) in (bool, int, float, str)
            if isinstance(value, float) and not math.isfinite(value):
                valid = False
            result[name] = {"value": value if valid else None,
                            "availability": "observed" if valid else "unavailable",
                            "source_attribute": f"{prefix}.{name}"}
    return {"schema": "autolabel-sam3-tracking-parameters/v1", "parameters": result,
            "scope": "Observed scalar attributes; no inference/state probing; historical artifacts unchanged"}
