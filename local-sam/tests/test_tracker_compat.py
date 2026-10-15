from __future__ import annotations

import logging
from copy import deepcopy
import unittest

from samlocal.compat import install_shared_conditioning_guard


class FakeTracker:
    def __init__(self, *, fail_reset: bool = False) -> None:
        self.reset_calls = 0
        self.fail_reset = fail_reset

    def clear_all_points_in_frame(self, inference_state, frame_idx, obj_id, need_output=False):
        inference_state["output_dict"]["cond_frame_outputs"].pop(frame_idx, None)
        inference_state["output_dict_per_obj"][obj_id]["cond_frame_outputs"].pop(
            frame_idx, None
        )
        self._reset_tracking_results(inference_state)

    def remove_objects(self, inference_state, obj_id, frame_idx=10):
        # Upstream remove_objects reaches clear through ``self``, not through any wrapper.
        self.clear_all_points_in_frame(inference_state, frame_idx, obj_id, need_output=False)

    def _reset_tracking_results(self, inference_state):
        self.reset_calls += 1
        if self.fail_reset:
            raise RuntimeError("reset failed")
        inference_state["output_dict"]["cond_frame_outputs"].clear()
        inference_state["output_dict"]["non_cond_frame_outputs"].clear()


class FakeWrapper:
    """Mirrors Sam3MultiplexTrackerPredictor: attribute access is forwarded to ``.model``."""

    def __init__(self, model) -> None:
        self.model = model

    def __getattr__(self, name):
        return getattr(self.__dict__["model"], name)


def make_state(*, ids=(0, 1, 2), noncond=()):
    per_obj = {
        obj_id: {"cond_frame_outputs": {10: {"obj": obj_id}},
                 "non_cond_frame_outputs": {}}
        for obj_id in ids
    }
    all_noncond = {}
    for frame_idx, frame_ids in noncond:
        frame = {"local_obj_id_to_idx": {obj_id: i for i, obj_id in enumerate(frame_ids)},
                 "pred_masks": {obj_id: f"frame-{frame_idx}-object-{obj_id}" for obj_id in frame_ids}}
        all_noncond[frame_idx] = frame
        for obj_id in frame_ids:
            if obj_id in per_obj:
                per_obj[obj_id]["non_cond_frame_outputs"][frame_idx] = {
                    "obj": obj_id, "frame": frame_idx
                }
    return {
        "obj_ids": list(ids),
        "output_dict": {"cond_frame_outputs": {10: {"owner": 0}},
                        "non_cond_frame_outputs": all_noncond},
        "output_dict_per_obj": per_obj,
    }


class SharedConditioningGuardTests(unittest.TestCase):
    def test_removing_conditioning_owner_promotes_earliest_frame_with_survivor_memory(self):
        tracker = FakeTracker()
        state = make_state(noncond=[(2, [0]), (4, [1, 2]), (6, [0, 1, 2])])
        original_second_candidate = deepcopy(state["output_dict"]["non_cond_frame_outputs"][6])
        install_shared_conditioning_guard(tracker, logging.getLogger("test"))

        tracker.clear_all_points_in_frame(state, frame_idx=10, obj_id=0)

        self.assertEqual(tracker.reset_calls, 0)
        self.assertEqual(set(state["output_dict"]["cond_frame_outputs"]), {4})
        self.assertEqual(set(state["output_dict"]["non_cond_frame_outputs"]), {2, 6})
        self.assertEqual(state["output_dict"]["cond_frame_outputs"][4]["local_obj_id_to_idx"], {1: 0, 2: 1})
        self.assertIn(4, state["output_dict_per_obj"][1]["cond_frame_outputs"])
        self.assertIn(4, state["output_dict_per_obj"][2]["cond_frame_outputs"])
        self.assertNotIn(4, state["output_dict_per_obj"][0]["cond_frame_outputs"])
        self.assertEqual(state["output_dict"]["non_cond_frame_outputs"][6], original_second_candidate)


    def test_guard_installed_through_wrapper_fires_on_internal_remove_path(self):
        inner = FakeTracker()
        wrapper = FakeWrapper(inner)
        state = make_state(noncond=[(2, [0]), (4, [1, 2])])

        info = install_shared_conditioning_guard(wrapper, logging.getLogger("test"))
        wrapper.remove_objects(state, 0)

        self.assertEqual(info["target"], "FakeTracker")
        self.assertNotIn("clear_all_points_in_frame", wrapper.__dict__)
        self.assertEqual(inner.reset_calls, 0)
        self.assertEqual(info["promotions"], 1)
        self.assertEqual(set(state["output_dict"]["cond_frame_outputs"]), {4})
        self.assertIs(install_shared_conditioning_guard(wrapper, logging.getLogger("test")), info)

    def test_reset_runs_when_no_objects_survive(self):
        tracker = FakeTracker()
        state = make_state(ids=(0,), noncond=[(2, [0])])
        install_shared_conditioning_guard(tracker, logging.getLogger("test"))

        tracker.clear_all_points_in_frame(state, frame_idx=10, obj_id=0)

        self.assertEqual(tracker.reset_calls, 1)
        self.assertEqual(state["output_dict"]["cond_frame_outputs"], {})
        self.assertEqual(state["output_dict"]["non_cond_frame_outputs"], {})


    def test_owner_only_nonconditioning_candidate_is_not_promoted(self):
        tracker = FakeTracker()
        state = make_state(ids=(0, 1), noncond=[(2, [0])])
        install_shared_conditioning_guard(tracker, logging.getLogger("test"))

        tracker.clear_all_points_in_frame(state, frame_idx=10, obj_id=0)

        self.assertEqual(tracker.reset_calls, 1)
        self.assertEqual(state["output_dict"]["cond_frame_outputs"], {})


    def test_unrelated_explicit_reset_is_not_intercepted(self):
        tracker = FakeTracker()
        state = make_state(noncond=[(2, [1, 2])])
        install_shared_conditioning_guard(tracker, logging.getLogger("test"))

        tracker._reset_tracking_results(state)

        self.assertEqual(tracker.reset_calls, 1)
        self.assertEqual(state["output_dict"]["cond_frame_outputs"], {})
        self.assertEqual(state["output_dict"]["non_cond_frame_outputs"], {})


    def test_original_reset_exception_propagates_and_context_is_cleared(self):
        tracker = FakeTracker(fail_reset=True)
        state = make_state(noncond=[])
        install_shared_conditioning_guard(tracker, logging.getLogger("test"))

        with self.assertRaisesRegex(RuntimeError, "reset failed"):
            tracker.clear_all_points_in_frame(state, frame_idx=10, obj_id=0)

        tracker.fail_reset = False
        tracker._reset_tracking_results(state)
        self.assertEqual(tracker.reset_calls, 2)
        self.assertEqual(state["output_dict"]["cond_frame_outputs"], {})


    def test_install_is_idempotent(self):
        tracker = FakeTracker()
        logger = logging.getLogger("test")
        first = install_shared_conditioning_guard(tracker, logger)
        clear_method = tracker.clear_all_points_in_frame
        reset_method = tracker._reset_tracking_results
        second = install_shared_conditioning_guard(tracker, logger)

        self.assertIs(first, second)
        self.assertIs(tracker.clear_all_points_in_frame, clear_method)
        self.assertIs(tracker._reset_tracking_results, reset_method)
        self.assertTrue(first["installed"])
        self.assertEqual(first["promotions"], 0)
