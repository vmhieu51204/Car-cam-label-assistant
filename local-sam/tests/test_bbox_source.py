from __future__ import annotations

import unittest

import numpy as np

from samlocal.backend import Sam31Backend


def _outputs(masks: np.ndarray, boxes_xyxy: list[tuple[float, float, float, float]]) -> dict:
    height, width = masks.shape[-2:]
    xywh = [((x1) / width, (y1) / height, (x2 - x1) / width, (y2 - y1) / height) for x1, y1, x2, y2 in boxes_xyxy]
    return {"out_obj_ids": np.arange(len(masks)), "out_probs": np.full(len(masks), 0.9, dtype=np.float32),
            "out_binary_masks": masks, "out_boxes_xywh": np.asarray(xywh, dtype=np.float32)}


class BboxSourceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = Sam31Backend(checkpoint=None)

    def test_box_comes_from_final_mask_not_inflated_model_box(self) -> None:
        masks = np.zeros((1, 48, 80), dtype=bool)
        masks[0, 20:24, 10:14] = True  # final mask: a small car
        # model box computed before the non-overlap step spans to the frame corner
        detections = self.backend._detections(_outputs(masks, [(10, 10, 79, 47)]), 0, 0, "car")
        self.assertEqual(detections[0].bbox, (10.0, 20.0, 14.0, 24.0))
        row = self.backend.bbox_audit["by_label"]["car"]
        self.assertEqual((row["compared"], row["model_box_differs"]), (1, 1))
        self.assertEqual(self.backend.bbox_audit["examples"][0]["model_box"], [10.0, 10.0, 79.0, 47.0])

    def test_object_with_empty_final_mask_is_dropped_and_counted(self) -> None:
        masks = np.zeros((1, 48, 80), dtype=bool)
        detections = self.backend._detections(_outputs(masks, [(10, 10, 20, 20)]), 0, 0, "car")
        self.assertEqual(detections, [])
        self.assertEqual(self.backend.bbox_audit["by_label"]["car"]["empty_final_mask"], 1)

    def test_edge_speck_is_dropped_before_the_box(self) -> None:
        masks = np.zeros((1, 48, 80), dtype=bool)
        masks[0, 20:30, 10:20] = True  # the car
        masks[0, 45, 78] = True  # one-pixel artifact near the frame corner
        detections = self.backend._detections(_outputs(masks, [(10, 20, 79, 46)]), 0, 0, "car")
        self.assertEqual(detections[0].bbox, (10.0, 20.0, 20.0, 30.0))
        row = self.backend.bbox_audit["by_label"]["car"]
        self.assertEqual((row["masks_with_specks"], row["specks_removed"], row["sparse_mask"]), (1, 1, 0))

    def test_two_distant_blobs_are_flagged_sparse(self) -> None:
        masks = np.zeros((1, 48, 80), dtype=bool)
        masks[0, 0:3, 0:3] = True
        masks[0, 44:48, 76:80] = True
        detections = self.backend._detections(_outputs(masks, [(0, 0, 80, 48)]), 0, 5, "car")
        self.assertEqual(detections[0].frame, 5)
        row = self.backend.bbox_audit["by_label"]["car"]
        self.assertEqual((row["model_box_differs"], row["sparse_mask"]), (0, 1))


if __name__ == "__main__":
    unittest.main()
