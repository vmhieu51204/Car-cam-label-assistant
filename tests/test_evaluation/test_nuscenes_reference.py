"""Small native-table fixtures and independently calculated geometry answers."""

import json
import math

import numpy as np
import pytest
from PIL import Image

from src.models.evaluation import (
    ClassMapping,
    NuScenesReferenceOptions,
    ProjectionPolicy,
    ReferenceProfile,
    canonical_json,
    content_hash,
)
from src.services.evaluation.references.nuscenes import load_nuscenes_reference
from src.services.geometry.nuscenes_projection import (
    box_corners_global,
    global_points_to_camera,
    intersect_projected_hull,
    project_camera_points,
    project_nuscenes_box,
)

CAMERAS = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT")
IDENTITY = {"translation": [0.0, 0.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0]}
INTRINSIC = [[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]]


def token(number):
    return f"{number:032x}"


def options(cameras=("CAM_FRONT",)):
    policy = ProjectionPolicy()
    return NuScenesReferenceOptions(
        split="mini_train",
        scenes=("scene-controlled",),
        cameras=cameras,
        reference_revision="controlled/v1",
        projection=policy,
        profile=ReferenceProfile(
            profile_id="nuscenes-projected",
            revision="nuscenes-projected/v1",
            designation="raw",
            classes=("car", "person"),
            class_map=(
                ClassMapping(native_class="vehicle.car", class_name="car"),
                ClassMapping(native_class="human.pedestrian.adult", class_name="person"),
            ),
            eligibility_policy="devkit-target-map/v1",
            geometry_policy=policy.revision,
            projection=policy,
        ),
    )


def fixture(root):
    """Two keyframes, all six cameras, one car; second frame is negative."""
    scene, first, last = token(1), token(2), token(3)
    annotation = {
        "token": token(7),
        "sample_token": first,
        "instance_token": token(8),
        "translation": [0.0, 0.0, 10.0],
        "size": [2.0, 4.0, 2.0],
        "rotation": [1.0, 0.0, 0.0, 0.0],
        "visibility_token": "4",
        "num_lidar_pts": 1,
        "num_radar_pts": 0,
        "attribute_tokens": [],
        "prev": "",
        "next": "",
    }
    tables = {
        "scene": [
            {
                "token": scene,
                "name": "scene-controlled",
                "first_sample_token": first,
                "last_sample_token": last,
                "nbr_samples": 2,
                "log_token": token(6),
            }
        ],
        "sample": [
            {"token": first, "scene_token": scene, "timestamp": 1000000, "prev": "", "next": last},
            {"token": last, "scene_token": scene, "timestamp": 1500000, "prev": first, "next": ""},
        ],
        "sample_data": [],
        "sensor": [],
        "calibrated_sensor": [],
        "ego_pose": [{"token": token(4), **IDENTITY}],
        "sample_annotation": [annotation],
        "instance": [
            {
                "token": token(8),
                "category_token": token(9),
                "first_annotation_token": token(7),
                "last_annotation_token": token(7),
            }
        ],
        "category": [{"token": token(9), "name": "vehicle.car"}],
        "log": [{"token": token(6)}],
        "attribute": [],
        "visibility": [{"token": "4"}],
    }
    for camera_index, channel in enumerate(CAMERAS):
        sensor, calibrated = token(20 + camera_index), token(30 + camera_index)
        tables["sensor"].append({"token": sensor, "channel": channel, "modality": "camera"})
        tables["calibrated_sensor"].append(
            {"token": calibrated, "sensor_token": sensor, **IDENTITY, "camera_intrinsic": INTRINSIC}
        )
        for frame_index, sample in enumerate((first, last)):
            filename = f"samples/{channel}/{frame_index}.png"
            path = root / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (100, 80), "white").save(path)
            tables["sample_data"].append(
                {
                    "token": token(100 + camera_index * 2 + frame_index),
                    "sample_token": sample,
                    "calibrated_sensor_token": calibrated,
                    "ego_pose_token": token(4),
                    "is_key_frame": True,
                    "timestamp": 1000010 + frame_index * 500000 + camera_index,
                    "width": 100,
                    "height": 80,
                    "filename": filename,
                    "prev": "",
                    "next": "",
                }
            )
    write_tables(root, tables)
    return tables


def write_tables(root, tables):
    version = root / "v1.0-mini"
    version.mkdir(exist_ok=True)
    for name, rows in tables.items():
        (version / f"{name}.json").write_text(json.dumps(rows), encoding="utf-8")


def test_known_answer_inverse_camera_ego_transforms_and_pixels():
    half = math.sqrt(0.5)
    ego = {"translation": [10.0, 20.0, 30.0], "rotation": [half, 0.0, 0.0, half]}
    camera = {"translation": [3.0, 4.0, 5.0], "rotation": [half, half, 0.0, 0.0]}
    # Rx90(1,2,10)+(3,4,5)=(4,-6,7); Rz90 plus ego=(16,24,37).
    points = global_points_to_camera(np.array([[16.0, 24.0, 37.0]]), ego, camera)
    assert points == pytest.approx(np.array([[1.0, 2.0, 10.0]]))
    assert project_camera_points(points, np.asarray(INTRINSIC)) == pytest.approx(np.array([[60.0, 60.0]]))


def test_full_rotation_corners_known_answer_not_yaw():
    # Quaternion (.5,.5,.5,.5) cyclically permutes (x,y,z) into (z,x,y).
    corners = box_corners_global(
        {"translation": [10.0, 20.0, 30.0], "rotation": [0.5, 0.5, 0.5, 0.5], "size": [2.0, 4.0, 6.0]}
    )
    expected = np.array(
        [[13, 22, 31], [13, 22, 29], [7, 22, 29], [7, 22, 31], [13, 18, 31], [13, 18, 29], [7, 18, 29], [7, 18, 31]],
        dtype=float,
    )
    assert corners == pytest.approx(expected)


def test_projected_box_known_pixels_and_depth_rejections():
    camera = {**IDENTITY, "camera_intrinsic": INTRINSIC}
    annotation = {"translation": [0.0, 0.0, 10.0], "size": [2.0, 4.0, 2.0], "rotation": [1.0, 0.0, 0.0, 0.0]}
    result = project_nuscenes_box(annotation, IDENTITY, camera, width=100, height=80)
    assert result.bbox_xyxy == pytest.approx((50 - 200 / 9, 40 - 100 / 9, 50 + 200 / 9, 40 + 100 / 9))
    assert not result.clipped
    for center, reason in ((-10, "behind_camera"), (0, "near_plane"), (1.05, "near_plane")):
        result = project_nuscenes_box(
            {**annotation, "translation": [0.0, 0.0, center]}, IDENTITY, camera, width=100, height=80
        )
        assert result.rejection_reason == reason
    outside = project_nuscenes_box(
        {**annotation, "translation": [100.0, 0.0, 10.0]}, IDENTITY, camera, width=100, height=80
    )
    assert outside.rejection_reason == "outside_canvas"


def test_hull_intersection_is_not_corner_clamping():
    # Triangle's bbox overlaps canvas, but triangle lies beyond x+y=250.
    assert (
        intersect_projected_hull(np.array([[50, 200], [200, 50], [200, 200]]), 100, 100).rejection_reason
        == "outside_canvas"
    )
    triangle = intersect_projected_hull(np.array([[-10, 50], [50, -10], [50, 50]]), 100, 100)
    assert triangle.bbox_xyxy == (0.0, 0.0, 50.0, 50.0)
    assert triangle.original_bbox_xyxy == (-10.0, -10.0, 50.0, 50.0)
    assert triangle.clipped
    assert (
        intersect_projected_hull(np.array([[0, 0], [1, 1], [2, 2]]), 100, 100).rejection_reason
        == "degenerate_projection"
    )
    assert (
        intersect_projected_hull(np.array([[np.nan, 1], [2, 2], [3, 3]]), 100, 100).rejection_reason
        == "invalid_projection"
    )


def test_all_six_camera_joins_native_identity_negative_frames_and_hash(tmp_path):
    tables = fixture(tmp_path)
    reference = load_nuscenes_reference(tmp_path, options=options(CAMERAS))
    assert len(reference.frames) == 12
    assert len(reference.boxes) == 6
    assert len(reference.native_nuscenes_annotations) == 1
    assert len({box.annotation_id for box in reference.boxes}) == 6
    assert len({box.native_annotation_id for box in reference.boxes}) == 1
    assert all(
        box.reference_kind == "projected_3d_box" and box.metadata.camera_visibility_revision is None
        for box in reference.boxes
    )
    assert all(box.metadata.nuscenes.size_wlh == (2.0, 4.0, 2.0) for box in reference.boxes)
    assert all(frame.camera.sample_timestamp_us == 1000000 for frame in reference.frames if frame.source_ordinal == 0)
    assert [frame.camera.sensor_timestamp_us - frame.camera.sample_timestamp_us for frame in reference.frames[::2]] == [
        10,
        11,
        12,
        13,
        14,
        15,
    ]
    assert all(box.frame.frame_uid != token(3) for box in reference.boxes)
    assert reference.content_sha256 == content_hash(reference)
    assert canonical_json(reference) == canonical_json(load_nuscenes_reference(tmp_path, options=options(CAMERAS)))
    assert json.loads((tmp_path / "v1.0-mini/sample_annotation.json").read_text()) == tables["sample_annotation"]


def test_loader_uses_each_camera_observation_pose(tmp_path):
    tables = fixture(tmp_path)
    half = math.sqrt(0.5)
    tables["ego_pose"].append(
        {"token": token(5), "translation": [10.0, 20.0, 30.0], "rotation": [half, 0.0, 0.0, half]}
    )
    tables["sample_data"][0]["ego_pose_token"] = token(5)
    tables["calibrated_sensor"][0].update(translation=[3.0, 4.0, 5.0], rotation=[half, half, 0.0, 0.0])
    tables["sample_annotation"][0].update(translation=[16.0, 24.0, 37.0], size=[2.0, 2.0, 2.0])
    write_tables(tmp_path, tables)
    reference = load_nuscenes_reference(tmp_path, options=options())
    assert reference.frames[0].camera.ego_pose_token == token(5)
    assert reference.boxes[0].bbox_xyxy == pytest.approx((50.0, 40 + 100 / 11, 50 + 200 / 9, 40 + 300 / 9))


@pytest.mark.parametrize(
    "table",
    [
        "scene",
        "sample",
        "sample_data",
        "sensor",
        "calibrated_sensor",
        "ego_pose",
        "sample_annotation",
        "instance",
        "category",
        "log",
        "attribute",
        "visibility",
    ],
)
def test_duplicate_tokens_fail_instead_of_overwriting(tmp_path, table):
    tables = fixture(tmp_path)
    row = tables[table][0] if tables[table] else {"token": token(200)}
    if not tables[table]:
        tables[table].append(row)
    tables[table].append(dict(row))
    write_tables(tmp_path, tables)
    with pytest.raises(ValueError, match="Duplicate"):
        load_nuscenes_reference(tmp_path, options=options())


@pytest.mark.parametrize(
    "fault",
    [
        "foreign_key",
        "chain",
        "timestamp",
        "duplicate_camera",
        "missing_camera",
        "dimensions",
        "pose",
        "intrinsic",
        "annotation",
        "traversal",
        "orphan_sample",
    ],
)
def test_structural_faults_reject_input(tmp_path, fault):
    tables = fixture(tmp_path)
    if fault == "foreign_key":
        tables["sample_data"][0]["ego_pose_token"] = "unknown"
    elif fault == "chain":
        tables["sample"][1]["prev"] = ""
    elif fault == "timestamp":
        tables["sample"][1]["timestamp"] = 1000000
    elif fault == "duplicate_camera":
        tables["sample_data"].append({**tables["sample_data"][0], "token": token(999)})
    elif fault == "missing_camera":
        tables["sample_data"].pop(0)
    elif fault == "dimensions":
        tables["sample_data"][0]["width"] = 99
    elif fault == "pose":
        tables["ego_pose"][0]["rotation"] = [2.0, 0.0, 0.0, 0.0]
    elif fault == "intrinsic":
        tables["calibrated_sensor"][0]["camera_intrinsic"] = [[0.0, 0.0, 0.0]] * 3
    elif fault == "annotation":
        tables["sample_annotation"][0]["size"] = [0.0, 2.0, 2.0]
    elif fault == "traversal":
        tables["sample_data"][0]["filename"] = "../outside.png"
    elif fault == "orphan_sample":
        tables["sample"].append({**tables["sample"][1], "token": token(999)})
    write_tables(tmp_path, tables)
    with pytest.raises(ValueError):
        load_nuscenes_reference(tmp_path, options=options())


@pytest.mark.parametrize(
    "center,reason",
    [([-100.0, 0.0, 10.0], "outside_canvas"), ([0.0, 0.0, -10.0], "behind_camera"), ([0.0, 0.0, 0.0], "near_plane")],
)
def test_rejected_annotation_retained_with_frame_reason_support(tmp_path, center, reason):
    tables = fixture(tmp_path)
    tables["sample_annotation"][0]["translation"] = center
    write_tables(tmp_path, tables)
    reference = load_nuscenes_reference(tmp_path, options=options())
    assert len(reference.frames) == 2 and not reference.boxes
    assert reference.native_nuscenes_annotations[0].translation_global == tuple(center)
    assert reference.rejections[0].native_annotation_id == token(7)
    assert reference.rejections[0].frame == reference.frames[0].key
    assert reference.rejections[0].reason == reason
    assert reference.rejection_counts[0].count == 1


def test_unmapped_pedestrian_is_preserved_but_never_scored(tmp_path):
    tables = fixture(tmp_path)
    tables["category"][0]["name"] = "human.pedestrian.personal_mobility"
    write_tables(tmp_path, tables)
    reference = load_nuscenes_reference(tmp_path, options=options())
    assert not reference.boxes and reference.rejections[0].reason == "out_of_task"
    assert reference.native_nuscenes_annotations[0].category_name == "human.pedestrian.personal_mobility"


def test_clipped_box_preserves_original_extent(tmp_path):
    tables = fixture(tmp_path)
    tables["sample_annotation"][0]["translation"] = [-4.0, 0.0, 10.0]
    write_tables(tmp_path, tables)
    reference = load_nuscenes_reference(tmp_path, options=options())
    box = reference.boxes[0]
    assert box.bbox_xyxy[0] == 0.0
    assert box.metadata.original_bbox_xyxy[0] < 0
    assert box.metadata.clipping_policy == options().projection.revision
    assert reference.rejection_counts[0].reason == "canvas_clipped"
    assert not reference.rejections


def test_duplicate_json_keys_rejected(tmp_path):
    fixture(tmp_path)
    (tmp_path / "v1.0-mini/log.json").write_text('[{"token":"a","token":"b"}]')
    with pytest.raises(ValueError, match="Duplicate JSON"):
        load_nuscenes_reference(tmp_path, options=options())


def test_no_native_annotations_retains_every_negative_frame(tmp_path):
    tables = fixture(tmp_path)
    tables["sample_annotation"] = []
    tables["instance"] = []
    write_tables(tmp_path, tables)
    reference = load_nuscenes_reference(tmp_path, options=options())
    assert len(reference.frames) == 2
    assert not reference.boxes and not reference.rejections and not reference.native_nuscenes_annotations


def test_projection_policy_agreement_and_supported_taxonomy(tmp_path):
    fixture(tmp_path)
    baseline = options()
    bad = NuScenesReferenceOptions.model_validate(
        {**baseline.model_dump(), "projection": ProjectionPolicy(min_depth_m=0.2)}
    )
    with pytest.raises(ValueError, match="policies must agree"):
        load_nuscenes_reference(tmp_path, options=bad)
    profile = ReferenceProfile.model_validate(
        {
            **baseline.profile.model_dump(),
            "class_map": (ClassMapping(native_class="human.pedestrian.personal_mobility", class_name="person"),),
        }
    )
    bad = NuScenesReferenceOptions.model_validate({**baseline.model_dump(), "profile": profile})
    with pytest.raises(ValueError, match="Unsupported pedestrian"):
        load_nuscenes_reference(tmp_path, options=bad)


def test_near_plane_inclusive_boundary_and_invalid_geometry():
    camera = {**IDENTITY, "camera_intrinsic": INTRINSIC}
    box = {"translation": [0.0, 0.0, 2.0], "size": [2.0, 2.0, 2.0], "rotation": [1.0, 0.0, 0.0, 0.0]}
    assert (
        project_nuscenes_box(box, IDENTITY, camera, width=100, height=80, min_depth_m=1.0).rejection_reason
        == "near_plane"
    )
    assert (
        project_nuscenes_box({**box, "size": [-1.0, 2.0, 2.0]}, IDENTITY, camera, width=100, height=80).rejection_reason
        == "invalid_projection"
    )
    with pytest.raises(ValueError, match="finite"):
        project_camera_points(np.array([[np.inf, 0.0, 1.0]]), np.asarray(INTRINSIC))
    with pytest.raises(ValueError, match="finite"):
        global_points_to_camera(np.array([[0.0, 0.0, 1.0]]), {**IDENTITY, "translation": [np.nan, 0.0, 0.0]}, camera)
