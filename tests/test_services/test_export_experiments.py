import copy
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from scripts.eval_autolabel import Box
from scripts.prepare_tracking_evaluation import split_sequences
from scripts.profile_low_memory import collect, memory_bytes
from src.models.evaluation import FrameKey, ReferenceBox, SequenceKey
from src.models.video import BoundingBox2D, TrackObservation, VideoTrack
from src.services.evaluation.phase0 import canonical_predictions
from src.services.evaluation.tracking import _temporal
from src.services.evaluation.tracking_diagnosis import event_signal
from src.services.geometry.pointcloud_export import PointCloudExportConfig, display_points, split_cuboid_draft
from src.services.video.keyframe_sparsify import SparseConfig, box_iou, interpolate, sparsify_shapes
from src.services.video.sequence_review import SequenceReviewConfig
from src.services.video.track_filter import ClassFilter, TrackFilterConfig, classify_tracks
from src.services.video.track_triage import track_triage


def track(size=20, frames=3, source="auto"):
    return VideoTrack(
        track_id="t",
        label_id=1,
        label_name="car",
        source=source,
        observations=[
            TrackObservation(frame_index=i, bbox=BoundingBox2D(xtl=0, ytl=0, xbr=size, ybr=size)) for i in range(frames)
        ],
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_filter_is_reversible_and_non_mutating(enabled):
    tracks = [track(size=2, frames=1), track(source="human_seed").model_copy(update={"track_id": "human"})]
    before = copy.deepcopy(tracks)
    suspects = classify_tracks(tracks, TrackFilterConfig(enabled=enabled), width=100, height=100)
    assert suspects == ({"t": ["too_short", "tiny_box"]} if enabled else {})
    assert tracks == before


def test_filter_boundaries_classes_regions_and_missing_scores():
    t = track(size=12)
    c = TrackFilterConfig(
        enabled=True, defaults=ClassFilter(min_frames=3, min_area_px=144, min_height_px=12, min_score=0.9)
    )
    assert classify_tracks([t], c, width=100, height=100) == {}
    c.ignore_regions = [(0, 0, 0.12, 0.12)]
    assert classify_tracks([t], c, width=100, height=100) == {"t": ["in_ignore_region"]}
    c.classes["car"] = ClassFilter(min_frames=4, min_area_px=0, min_height_px=0)
    assert "too_short" in classify_tracks([t], c, width=100, height=100)["t"]
    with pytest.raises(ValidationError):
        TrackFilterConfig(ignore_regions=[(0, 0, 2, 1)])


def shapes(n=31):
    return [
        {"frame": i, "points": [i, 0, i + 20, 20], "outside": False, "occluded": False, "attributes": []}
        for i in range(n)
    ]


def test_sparse_disabled_exact_identity_and_linear_bound():
    rows = shapes()
    original = copy.deepcopy(rows)
    dense, sidecar = sparsify_shapes(rows, SparseConfig())
    assert dense is rows and sidecar == {}
    sparse, kept = sparsify_shapes(rows, SparseConfig(enabled=True))
    assert len(sparse) == 3
    assert [s["frame"] for s in sparse] == [0, 15, 30]
    assert rows == original
    assert kept[15] == ["max_gap"]
    for a, b in zip(sparse, sparse[1:]):
        for i in range(a["frame"], b["frame"] + 1):
            assert box_iou(rows[i]["points"], interpolate(a, b, i)) == 1


def test_sparse_nonlinear_mandatory_outside_gaps_and_attributes():
    rows = shapes(12)
    rows[4]["points"] = [60, 0, 80, 20]
    rows[7]["outside"] = True
    rows[8]["frame"] = 20
    rows = rows[:9]
    out, sidecar = sparsify_shapes(rows, SparseConfig(enabled=True), mandatory={3})
    assert {0, 3, 4, 6, 7, 20} <= {s["frame"] for s in out}
    assert sidecar[3] == ["review_or_boundary"]
    with pytest.raises(ValueError):
        sparsify_shapes(list(reversed(rows)), SparseConfig(enabled=True))


@pytest.mark.parametrize("value,expected", [("1GiB", 1024**3), ("2.5 MiB", 2621440), ("100MB", 100000000)])
def test_memory_units(value, expected):
    assert memory_bytes(value) == expected


def test_memory_allowlist_and_unavailable_measurements():
    def fake(args, **kwargs):
        assert args[:3] == ["docker", "stats", "--no-stream"]
        return SimpleNamespace(
            stdout='{"Name":"cvat_server","MemUsage":"1MiB / 2GiB","CPUPerc":"2%","Secret":"never output"}\n'
            '{"Name":"other","MemUsage":"9GiB / 10GiB","CPUPerc":"0%"}'
        )

    result = collect(fake)
    assert result["total_cvat_bytes"] == 1024**2
    assert result["wsl_vmmem_bytes"] is None
    assert "Secret" not in json.dumps(result)


def test_pointcloud_roi_voxel_stable_source_preservation():
    points = np.array([[1, 0, 0, 1], [1.01, 0, 0, 2], [-1, 0, 0, 3], [50, 0, 0, 4], [1, 10, 0, 5]], dtype=np.float32)
    original = points.copy()
    assert display_points(points, PointCloudExportConfig()) is points
    cfg = PointCloudExportConfig(enabled=True, max_range_m=5, voxel_size_m=0.1)
    out = display_points(points, cfg)
    np.testing.assert_array_equal(out, points[:1])
    np.testing.assert_array_equal(points, original)
    np.testing.assert_array_equal(out, display_points(points, cfg))
    assert len(display_points(points, cfg.model_copy(update={"voxel_size_m": 0}))) == 2


def test_split_tasks_source_mapping_and_cuboids_unchanged():
    draft = {
        "source_frame_count": 201,
        "tracks": [{"frame": 0, "shapes": [{"frame": i, "points": [i, 2, 3]} for i in range(201)]}],
    }
    original = copy.deepcopy(draft)
    tasks = split_cuboid_draft(draft, 100)
    assert [t["source_frame_count"] for t in tasks] == [100, 100, 1]
    assert tasks[1]["tracks"][0]["shapes"][0] == {"frame": 0, "points": [100, 2, 3]}
    assert tasks[2]["source_frames"] == [200]
    assert draft == original


def test_triage_stable_local_evidence_and_outside_only():
    manifest = SimpleNamespace(
        job_id="job",
        frames=[SimpleNamespace(frame_index=i, source_frame_index=i * 6, scene_cut_before=False) for i in range(3)],
    )
    stable = track().model_copy(update={"track_id": "stable"})
    ambiguous = track(size=2, frames=1).model_copy(
        update={"track_id": "ambiguous", "needs_review": True, "review_reasons": ["ambiguous_overlap"]}
    )
    invisible = track(frames=1).model_copy(
        update={
            "track_id": "outside",
            "observations": [track(frames=1).observations[0].model_copy(update={"outside": True})],
        }
    )
    tracks = SimpleNamespace(job_id="job", tracks=[stable, ambiguous, invisible])
    config = SequenceReviewConfig(class_review_policy={})
    rows = track_triage(tracks, manifest, config)
    assert [r["track_id"] for r in rows] == ["ambiguous", "stable"]
    assert rows[0]["flagged_frames"] == 1
    assert rows[0]["source_frame"] == 0
    assert rows[1]["reasons"] == []
    assert rows == track_triage(tracks, manifest, config)


def test_sequence_split_no_leakage_and_event_priority():
    names = [f"{i:04d}" for i in range(21)]
    a = split_sequences(names)
    assert a == split_sequences(list(reversed(names)))
    assert {"0013", "0020"} <= set(a["dev"])
    assert not set(a["dev"]) & set(a["held_out"])
    assert len(a["held_out"]) == 9
    assert (
        event_signal(near_cut=True, near_boundary=True, gap_inside_window=True, small=True, occluded=True)
        == "near_scene_cut"
    )
    assert (
        event_signal(near_cut=False, near_boundary=False, gap_inside_window=True, small=True, occluded=True)
        == "within_window_gap"
    )


def test_default_tracking_keeps_non_numeric_frame_uid_compatibility():
    key = FrameKey(sequence=SequenceKey(dataset_id="coco", split="test", sequence_uid="video"), frame_uid="cameraA")
    gt = ReferenceBox(
        annotation_id="gt",
        native_annotation_id="gt",
        frame=key,
        class_name="car",
        bbox_xyxy=(0, 0, 20, 20),
        native_track_id="t",
        source_ordinal=0,
        reference_kind="native_2d",
        native_category="car",
    )
    pred = canonical_predictions([Box(0, "car", (0, 0, 20, 20), "t")], {0: key}, "fixture")
    result = _temporal([gt], list(pred), (key,), 0.5)
    assert result["id_switches"] == 0 and result["fragmentations"] == 0
    assert "events" not in result


@pytest.mark.skipif(shutil.which("node") is None, reason="Node required")
def test_panel_transforms_encoding_and_dock_contract():
    p = Path("integrations/cvat-autolabel/files/cvat-ui/autolabel-panel.js")
    source = p.read_text(encoding="utf-8")
    assert not any(s in source for s in ["\u00c2", "\ufffd", "Ch?y"])
    assert "Chạy" in source and "Duyệt" in source
    assert "background.parentElement.appendChild(overlay)" in source
    assert "page.appendChild(panel)" in source
    assert "frame:proposal" not in source
    code = (
        "const fs=require('fs'),vm=require('vm'),assert=require('assert');const s=fs.readFileSync(process.argv[1],'utf8');"
        "const f=s.slice(s.indexOf('    function proposalRect('),s.indexOf('    function syncOverlay('));"
        "const c={};vm.createContext(c);vm.runInContext(f,c);const r=c.proposalRect([.1,.2,.5,.6],1000,500);"
        "assert.equal(r.x,100);assert.equal(r.y,100);assert.equal(r.width,400);assert(Math.abs(r.height-200)<1e-9);"
    )
    subprocess.run(["node", "-e", code, str(p)], check=True)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node required")
def test_canvas_overlay_tracks_display_transform_and_clears_on_other_frame():
    code = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const source=fs.readFileSync(process.argv[1],'utf8');
const helper=source.slice(source.indexOf('    function proposalRect('),source.indexOf('    function showProposal('));
const nodes=[];function node(){const n={style:{},dataset:{},attrs:{},children:[],classList:{add(){}},
setAttribute(k,v){this.attrs[k]=v;},appendChild(c){this.children.push(c);},replaceChildren(){this.children=[];},remove(){this.removed=true;}};nodes.push(n);return n;}
const parent=node(), background={width:1000,height:500,parentElement:parent};
const css={left:'30px',top:'-20px',width:'1000px',height:'500px',transform:'scale(2) rotate(0deg)',transformOrigin:'center'};
const c={selectedProposal:{frame:6,ready:false,proposal:{label:'car',bbox:[.1,.2,.5,.6],status:'pending'}},
sam3Capabilities:{canvas_overlay:true},panel:{},overlay:null,observedBackground:null,canvasObserver:null,
busy:false,language:'vi',frame:6,currentSourceFrame:()=>c.frame,tr:k=>k,getComputedStyle:()=>css,
document:{getElementById:()=>background,createElementNS:()=>node()},MutationObserver:class{observe(){}disconnect(){}}};
vm.createContext(c);vm.runInContext(helper,c);c.syncOverlay();
assert.equal(c.overlay.style.left,'30px');assert.equal(c.overlay.style.top,'-20px');
assert.equal(c.overlay.style.transform,'scale(2) rotate(0deg)');
assert.equal(c.overlay.children[0].attrs.x,'100');assert.equal(c.overlay.children[0].attrs.y,'100');
const old=c.overlay;c.frame=7;c.syncOverlay();assert.equal(c.overlay,null);assert(old.removed);
c.frame=6;background.width=500;background.height=250;css.transform='scale(0.5) rotate(30deg)';c.syncOverlay();
assert.equal(c.overlay.children[0].attrs.x,'50');assert.equal(c.overlay.children[0].attrs.y,'50');
assert.equal(c.overlay.style.transform,'scale(0.5) rotate(30deg)');
"""
    subprocess.run(["node", "-e", code, "integrations/cvat-autolabel/files/cvat-ui/autolabel-panel.js"], check=True)
