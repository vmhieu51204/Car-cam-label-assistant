"""Selected SAM3 keyframes expand to a full source-frame timeline."""

import xml.etree.ElementTree as ET
from zipfile import ZipFile

from scripts.eval_autolabel import read_boxes
from src.services.video.export_2d import export_2d
from src.services.video.stitching import ChunkObservation, UnifiedTrack


def test_export_interpolates_between_processed_frames_and_marks_outside(tmp_path):
    track = UnifiedTrack(
        "auto:000001",
        "car",
        [
            ChunkObservation(0, (0, 0, 10, 10)),
            ChunkObservation(1, (10, 0, 20, 10)),
        ],
    )
    artifacts = export_2d(
        [track],
        source_frames=[0, 2, 4],
        image_names=[f"{frame:06d}.png" for frame in range(5)],
        width=40,
        height=30,
        output_dir=tmp_path,
    )
    coco = read_boxes(artifacts.coco)
    assert [(box.frame, box.xyxy) for box in coco] == [
        (0, (0.0, 0.0, 10.0, 10.0)),
        (1, (5.0, 0.0, 15.0, 10.0)),
        (2, (10.0, 0.0, 20.0, 10.0)),
    ]
    xml = artifacts.cvat_xml.read_text(encoding="utf-8")
    assert 'frame="4" outside="1"' in xml
    assert 'name="review_status">pending' in xml
    root = ET.fromstring(xml)
    assert root.findtext("./meta/task/mode") == "interpolation"
    assert root.findtext("./meta/task/original_size/width") == "40"
    assert root.findtext("./meta/task/original_size/height") == "30"
    assert root.findtext("./meta/task/labels/label/name") == "car"
    assert root.findtext("./track/box/attribute[@name='autolabel_track_id']") == "auto:000001"
    assert root.findtext("./track/box/attribute[@name='source']") == "auto"
    assert root.findtext("./track/box/attribute[@name='routing']") == "review"
    assert root.findtext("./track/box/attribute[@name='review_status']") == "pending"
    with ZipFile(artifacts.cvat_zip) as archive:
        assert archive.namelist() == ["annotations.xml"]


def test_export_does_not_interpolate_across_missed_sample(tmp_path):
    track = UnifiedTrack(
        "auto:000002",
        "person",
        [
            ChunkObservation(0, (0, 0, 10, 10)),
            ChunkObservation(2, (20, 0, 30, 10)),
        ],
    )
    artifacts = export_2d(
        [track],
        source_frames=[0, 2, 4],
        image_names=[f"{frame:06d}.png" for frame in range(5)],
        width=40,
        height=30,
        output_dir=tmp_path,
    )
    assert [box.frame for box in read_boxes(artifacts.coco)] == [0, 4]
    assert 'frame="2" outside="1"' in artifacts.cvat_xml.read_text(encoding="utf-8")
