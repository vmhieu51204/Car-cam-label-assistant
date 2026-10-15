import math

from shapely import affinity
from shapely.geometry import Polygon, box

from src.models.perception import Box3D


def footprint(value: Box3D) -> Polygon:
    """Return the rotated ground-plane footprint in the box's declared frame."""
    length, width, _ = value.size_lwh
    polygon = box(-length / 2, -width / 2, length / 2, width / 2)
    polygon = affinity.rotate(polygon, value.yaw, origin=(0, 0), use_radians=True)
    return affinity.translate(polygon, xoff=value.center[0], yoff=value.center[1])


def iou3d(a: Box3D, b: Box3D) -> float:
    """Compute intersection over union for two upright boxes."""
    pa, pb = footprint(a), footprint(b)
    low = max(a.center[2] - a.size_lwh[2] / 2, b.center[2] - b.size_lwh[2] / 2)
    high = min(a.center[2] + a.size_lwh[2] / 2, b.center[2] + b.size_lwh[2] / 2)
    intersection = pa.intersection(pb).area * max(0.0, high - low)
    return float(intersection / (math.prod(a.size_lwh) + math.prod(b.size_lwh) - intersection))


def giou3d(a: Box3D, b: Box3D) -> float:
    """Upright 3D GIoU with a convex-hull footprint enclosing prism.

    This enclosing prism is explicit; it is not a minimum-volume 3D hull.
    """
    pa, pb = footprint(a), footprint(b)
    a_low, a_high = a.center[2] - a.size_lwh[2] / 2, a.center[2] + a.size_lwh[2] / 2
    b_low, b_high = b.center[2] - b.size_lwh[2] / 2, b.center[2] + b.size_lwh[2] / 2
    overlap_height = max(0.0, min(a_high, b_high) - max(a_low, b_low))
    intersection = pa.intersection(pb).area * overlap_height
    union = math.prod(a.size_lwh) + math.prod(b.size_lwh) - intersection
    enclosure = pa.union(pb).convex_hull.area * (max(a_high, b_high) - min(a_low, b_low))
    return float(intersection / union - (enclosure - union) / enclosure)
