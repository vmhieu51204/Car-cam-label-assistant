"""Compatibility wrappers for the local KITTI diagnostic (not official HOTA)."""

from __future__ import annotations

from scripts.eval_autolabel import Box, tracking_metrics
from src.services.evaluation.references.kitti import (
    KITTI_CLASSES as KITTI_CLASSES,
)
from src.services.evaluation.references.kitti import (
    TARGET_CLASSES as TARGET_CLASSES,
)
from src.services.evaluation.references.kitti import (
    KittiItem as KittiItem,
)
from src.services.evaluation.references.kitti import (
    KittiReference as KittiReference,
)
from src.services.evaluation.references.kitti import (
    _ioa as _ioa,
)
from src.services.evaluation.references.kitti import (
    load_reference as load_reference,
)
from src.services.evaluation.references.kitti import (
    preprocess_kitti_style,
)


def evaluate_kitti_style(reference: KittiReference, predictions: list[Box], frames: set[int]) -> dict:
    valid_gt, retained, removed = preprocess_kitti_style(reference, predictions, frames)
    return {
        "metrics": tracking_metrics(valid_gt, retained, frames),
        "ignored_predictions": removed,
        "valid_ground_truth": len(valid_gt),
        "retained_predictions": len(retained),
    }
