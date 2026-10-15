from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kitti"))

import numpy as np  # noqa: E402
from kitti_metrics import Box, detection, evaluate, first_detection, hota, parse_labels  # noqa: E402
from mots_kitti import decode_rle  # noqa: E402
from trackeval_kitti import replace_section  # noqa: E402

from samlocal.backend import Sam31Backend, mask_rle  # noqa: E402

BOX = (100.0, 100.0, 200.0, 200.0)


def _track(track: str, frames: range | list[int], xyxy: tuple[float, ...] = BOX, label: str = "car") -> list[Box]:
    return [Box(frame, label, xyxy, track, 1.0) for frame in frames]


def _label(frame: int, track: int, native: str, xyxy: tuple[float, ...], *, occluded: int = 0,
           truncated: float = 0.0, depth: float = 10.0) -> str:
    x1, y1, x2, y2 = xyxy
    return (f"{frame} {track} {native} {truncated} {occluded} 0 {x1} {y1} {x2} {y2} "
            f"1.5 1.6 4.0 0.0 1.6 {depth} 0.0")


class HotaTest(unittest.TestCase):
    def test_half_detected_track(self) -> None:
        # GT on 2 frames, prediction only on the first: DetRe = AssRe = 0.5, DetPr = AssPr = 1, HOTA = 0.5
        result = hota(_track("gt", [0, 1]), _track("p", [0]), {0, 1})
        self.assertAlmostEqual(result["det_re"], 0.5)
        self.assertAlmostEqual(result["det_pr"], 1.0)
        self.assertAlmostEqual(result["ass_re"], 0.5)
        self.assertAlmostEqual(result["ass_pr"], 1.0)
        self.assertAlmostEqual(result["hota"], 0.5)
        self.assertAlmostEqual(result["loc_a"], 1.0)

    def test_id_switch_halves_association(self) -> None:
        # one GT track covered by two predicted tracks of 2 frames each: DetA = 1, AssA = AssRe = 0.5
        result = hota(_track("gt", range(4)), _track("a", [0, 1]) + _track("b", [2, 3]), set(range(4)))
        self.assertAlmostEqual(result["det_a"], 1.0)
        self.assertAlmostEqual(result["ass_a"], 0.5)
        self.assertAlmostEqual(result["ass_re"], 0.5)
        self.assertAlmostEqual(result["ass_pr"], 1.0)
        self.assertAlmostEqual(result["hota"], math.sqrt(0.5))

    def test_no_predictions(self) -> None:
        result = hota(_track("gt", range(3)), [], set(range(3)))
        self.assertEqual(result["hota"], 0.0)
        self.assertEqual(result["det_re"], 0.0)


class RecallTest(unittest.TestCase):
    def test_f2_weights_recall(self) -> None:
        result = detection(_track("gt", [0, 1]), _track("p", [0]), {0, 1}, 0.5)
        self.assertAlmostEqual(result["f2"], 5 * 1 * 0.5 / (4 * 1 + 0.5))

    def test_first_detection_delay_and_coverage(self) -> None:
        frames = set(range(6))
        gt = _track("present", range(6)) + _track("enters", range(2, 6), (300.0, 100.0, 400.0, 200.0))
        pred = _track("p", range(4, 6), (300.0, 100.0, 400.0, 200.0))
        result = first_detection(gt, pred, frames)
        self.assertEqual(result["entered"]["delays"], [2])  # enters at frame 2, first matched at frame 4
        self.assertEqual(result["at_start"]["never_detected"], 1)
        self.assertAlmostEqual(result["coverage"], 0.5)
        self.assertAlmostEqual(result["entered"]["delay_ms_median"], 200.0)

    def test_breakdown_by_distance_and_occlusion(self) -> None:
        near, far, hidden = (0, 100, 100, 200), (300, 100, 330, 130), (500, 100, 600, 200)
        labels = "\n".join([_label(0, 1, "Car", near, depth=10.0), _label(0, 2, "Car", far, depth=55.0),
                            _label(0, 3, "Car", hidden, occluded=3, depth=25.0)])
        reference = parse_labels(labels, profile="kitti2")
        predictions = [Box(0, "car", tuple(map(float, near)), "p1", 0.9),
                       Box(0, "car", tuple(map(float, hidden)), "p3", 0.9)]
        car = evaluate(reference, predictions, {0})["per_class"]["car"]["recall_breakdown"]
        self.assertEqual(car["distance"], {"<20m": {"gt": 1, "tp": 1}, ">=40m": {"gt": 1, "tp": 0}})
        self.assertEqual(car["height"]["25-40px"], {"gt": 1, "tp": 0})
        # occlusion uses unfiltered GT: the occ3 car is scored there although KITTI ignores it
        self.assertEqual(car["occlusion"]["occ3"], {"gt": 1, "tp": 1})
        self.assertEqual(car["occlusion"]["occ0"], {"gt": 2, "tp": 1})


class MotsAndReportTest(unittest.TestCase):
    def test_mask_rle_round_trip(self) -> None:
        mask = np.zeros((5, 7), dtype=bool)
        mask[0, 0] = mask[1:3, 2:6] = mask[4, 6] = True
        np.testing.assert_array_equal(decode_rle(mask_rle(mask)), mask)

    def test_replace_section_keeps_other_sections(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "report.md"
            path.write_text("# report\n", encoding="utf-8")
            replace_section(path, "<!-- a -->", ["<!-- a -->", "A1"])
            replace_section(path, "<!-- b -->", ["<!-- b -->", "B1"])
            replace_section(path, "<!-- a -->", ["<!-- a -->", "A2"])
            text = path.read_text(encoding="utf-8")
        self.assertIn("B1", text)
        self.assertIn("A2", text)
        self.assertNotIn("A1", text)
        self.assertTrue(text.startswith("# report"))


class ObjectCapTest(unittest.TestCase):
    def test_prompt_window_at_cap_is_counted(self) -> None:
        backend = Sam31Backend(checkpoint=None, max_objects=2)
        backend.object_cap["max_objects"] = backend.max_objects_cap
        dets = [type("D", (), {"local_id": i})() for i in range(2)]
        with self.assertLogs("local-sam", level="WARNING"):
            backend._count_objects("car", dets, frame_offset=40, num_frames=48)
        backend._count_objects("van", dets[:1], frame_offset=40, num_frames=48)
        self.assertEqual(backend.object_cap["prompt_windows_at_cap"], 1)
        self.assertEqual(backend.object_cap["max_ids_per_prompt_window"], {"car": 2, "van": 1})

    def test_upstream_cap_warnings_are_counted_once(self) -> None:
        import logging

        from samlocal.backend import _install_cap_counter

        target = {"limit_warnings": 0}
        _install_cap_counter(target)
        _install_cap_counter(target)  # a second load() must not double count
        upstream = logging.getLogger("sam3.model.sam3_video_base")
        upstream.propagate = False  # like sam3.logger.get_logger
        upstream.warning("hitting self.max_num_objects=16 with new_det_num=3 and prev_obj_num=15")
        upstream.info("`setting max_num_objects` to 16")
        self.assertEqual(target["limit_warnings"], 1)


if __name__ == "__main__":
    unittest.main()
