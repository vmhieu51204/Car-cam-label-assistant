"""The GPU adapter must discover each class on every window and retain masks."""

import numpy as np
import pytest

from inference.sam3_sequence import collect_text_window, require_video_text_api


class FakeVideoModel:
    def __init__(self):
        self.prompts = []
        self.states = 0

    def init_state(self, *, resource_path):
        self.states += 1
        return {"path": resource_path}

    def add_prompt(self, *, inference_state, frame_idx, text_str):
        assert frame_idx == 0
        inference_state["prompt"] = text_str
        self.prompts.append(text_str)

    def propagate_in_video(self, state, *, start_frame_idx, max_frame_num_to_track, reverse):
        assert start_frame_idx == 0 and not reverse
        for index in range(max_frame_num_to_track):
            mask = np.zeros((12, 16), dtype=bool)
            mask[2:7, index : index + 5] = True
            yield index, {"out_obj_ids": [1], "out_binary_masks": [mask], "out_probs": [0.7]}


def test_text_prompt_uses_fresh_model_state_per_class(tmp_path):
    model = FakeVideoModel()
    tracks = collect_text_window(model, directory=tmp_path, window_id=3, frames=[45, 46], labels=("car", "person"))
    assert model.states == 2 and model.prompts == ["car", "person"]
    assert {(track.label, track.local_track_id) for track in tracks} == {("car", "car:1"), ("person", "person:1")}
    assert all([obs.frame for obs in track.observations] == [45, 46] for track in tracks)
    assert all(obs.mask is not None and obs.mask.sha256 for track in tracks for obs in track.observations)
    assert tracks[0].observations[0].bbox == (0.0, 2.0, 5.0, 7.0)


def test_missing_video_text_api_fails_explicitly():
    with pytest.raises(NotImplementedError, match="video text-prompt API"):
        require_video_text_api(object())
