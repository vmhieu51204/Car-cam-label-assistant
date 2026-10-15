"""Instance-local workaround for SAM3 shared-conditioning removal (upstream #572/#573).

Removing a bucket's conditioning owner must not reset the surviving objects' memory.
Keep the upstream clear/remove algorithm and intercept only its destructive reset when
a different, still-registered object has a usable tracked frame. No package files change.
"""
from __future__ import annotations

import functools
import inspect


def install_shared_conditioning_guard(tracker, logger) -> dict:
    # SAM3.1 hands the base model a Sam3MultiplexPredictorWrapper whose __getattr__ forwards to
    # `.model`. remove_objects then runs on that inner model and calls
    # self.clear_all_points_in_frame there, so patching the wrapper would never fire.
    tracker = getattr(getattr(tracker, "clear_all_points_in_frame", None), "__self__", tracker)
    existing = getattr(tracker, "_local_shared_conditioning_guard", None)
    if existing is not None:
        return existing
    info = {"patch_id": "shared-conditioning-anchor-v2", "installed": False, "promotions": 0,
            "target": type(tracker).__name__,
            "upstream_issue": "https://github.com/facebookresearch/sam3/issues/572"}
    clear = getattr(tracker, "clear_all_points_in_frame", None)
    reset = getattr(tracker, "_reset_tracking_results", None)
    if not callable(clear) or not callable(reset):
        info["reason"] = "tracker does not expose the multiplex clear/reset interface"
        return info
    signature = inspect.signature(clear)
    if not {"inference_state", "obj_id"}.issubset(signature.parameters):
        info["reason"] = "unsupported tracker clear signature"
        return info
    contexts = []

    @functools.wraps(clear)
    def guarded_clear(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        contexts.append((bound.arguments["inference_state"], bound.arguments["obj_id"]))
        try:
            return clear(*args, **kwargs)
        finally:
            contexts.pop()

    @functools.wraps(reset)
    def guarded_reset(inference_state, *args, **kwargs):
        if contexts and contexts[-1][0] is inference_state:
            removed_id = contexts[-1][1]
            outputs = inference_state.get("output_dict", {})
            cond = outputs.get("cond_frame_outputs")
            noncond = outputs.get("non_cond_frame_outputs", {})
            surviving = set(inference_state.get("obj_ids", [])) - {removed_id}
            if isinstance(cond, dict) and not cond and surviving:
                for frame in sorted(noncond):
                    candidate = noncond[frame]
                    if surviving.intersection(candidate.get("local_obj_id_to_idx", {})):
                        cond[frame] = noncond.pop(frame)
                        for obj_outputs in inference_state.get("output_dict_per_obj", {}).values():
                            sliced = obj_outputs.get("non_cond_frame_outputs", {}).pop(frame, None)
                            if sliced is not None:
                                obj_outputs.setdefault("cond_frame_outputs", {})[frame] = sliced
                        info["promotions"] += 1
                        logger.warning("SAM3 shared-conditioning guard: retained frame %s after clearing "
                                       "object %s; surviving objects=%s (promotion %d)",
                                       frame, removed_id, sorted(surviving), info["promotions"])
                        return None
        return reset(inference_state, *args, **kwargs)

    tracker.clear_all_points_in_frame = guarded_clear
    tracker._reset_tracking_results = guarded_reset
    tracker._local_shared_conditioning_guard = info
    info["installed"] = True
    return info
