"""Experimental singleton tracker groups, following FudanCVL/SAM3-DMS.

Only initialization grouping changes. The installed tracker still selects memory
using its own score/threshold, including its mandatory adjacent frame.
"""
from __future__ import annotations

import inspect
from types import MethodType
from typing import Any

SOURCE_COMMIT = "88475d9fc2b6267f8542710375dec92b40590353"


def validate_evidence(item: dict) -> None:
    evidence = item.get("runtime", {}).get("memory_selection", {})
    groups = evidence.get("verified_singleton_groups")
    if (evidence.get("policy") != "dms_singleton" or evidence.get("author_commit") != SOURCE_COMMIT
            or type(groups) is not int or groups < 0 or (item.get("predictions") and groups == 0)):
        raise ValueError("missing effective DMS singleton evidence")


class DecoupledMemory:
    def __init__(self, model: Any):
        self.model = model
        self.original = model._tracker_add_new_objects
        expected = ("frame_idx", "num_frames", "new_obj_ids", "new_obj_masks",
                    "tracker_states_local", "orig_vid_height", "orig_vid_width", "feature_cache")
        if tuple(inspect.signature(self.original).parameters) != expected:
            raise NotImplementedError("unsupported SAM3 tracker initialization API")
        if not model.tracker.use_memory_selection:
            raise NotImplementedError("DMS requires native memory selection enabled")
        self.groups = 0
        self.active = False

    def install(self) -> None:
        if self.active:
            raise RuntimeError("DMS already installed")
        owner = self

        def initialize(_model: Any, frame_idx: int, num_frames: int, new_obj_ids: list[int], new_obj_masks: Any,
                       tracker_states_local: list[Any], orig_vid_height: int, orig_vid_width: int,
                       feature_cache: dict) -> list[Any]:
            if len(new_obj_ids) != new_obj_masks.size(0):
                raise ValueError("object IDs and masks disagree")
            for index, object_id in enumerate(new_obj_ids):
                before = len(tracker_states_local)
                owner.original(frame_idx, num_frames, [object_id], new_obj_masks[index:index + 1],
                               tracker_states_local, orig_vid_height, orig_vid_width, feature_cache)
                if len(tracker_states_local) != before + 1:
                    raise RuntimeError("SAM3 did not create one tracker group per object")
                if list(tracker_states_local[-1]["obj_ids"]) != [object_id]:
                    raise RuntimeError("DMS singleton tracker verification failed")
                owner.groups += 1
            return tracker_states_local

        self.model._tracker_add_new_objects = MethodType(initialize, self.model)
        self.active = True

    def restore(self) -> None:
        self.model._tracker_add_new_objects = self.original
        self.active = False

    def evidence(self) -> dict:
        return {"policy": "dms_singleton", "author_commit": SOURCE_COMMIT,
                "verified_singleton_groups": self.groups,
                "memory_threshold": self.model.tracker.mf_threshold}
