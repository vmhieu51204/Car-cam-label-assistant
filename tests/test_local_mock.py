from fastapi.testclient import TestClient
import pytest
from local_mock import create_mock_app


@pytest.fixture
def mock_client(tmp_path):
    with TestClient(create_mock_app(demo_root=tmp_path, autoload=False)) as client:
        yield client


def start(client):
    response = client.post("/mock/run")
    assert response.status_code == 200
    return response.json()


def test_original_panel_and_local_assets(mock_client):
    assert mock_client.get("/").status_code == 200
    for path in ("/mock.js", "/autolabel-panel.js"):
        assert mock_client.get(path).status_code == 200
    assert mock_client.get("/healthz").json()["weights_loaded"] is False
    assert mock_client.post("/mock/reevaluate", json={}).status_code == 404


def test_result_parser_review_queue_and_known_answer_metrics(mock_client):
    job = start(mock_client)
    state = mock_client.get("/mock/state").json()
    report = state["history"][-1]
    assert (report["tp"], report["fp"], report["fn"]) == (37, 1, 3)
    base = f"/autolabel/api/v1/video/jobs/{job['job_id']}"
    assert len(mock_client.get(base + "/review-queue").json()["units"]) == 24
    review = mock_client.get(base + "/review").json()
    assert any("box_jump" in t["reasons"] for t in review["tracks"])
    assert any("track_gap" in t["reasons"] for t in review["tracks"])


def test_correction_improves_metrics_and_export_uses_current_tracks(mock_client):
    start(mock_client)
    state = mock_client.get("/mock/state").json()
    for track in state["tracks"]["tracks"]:
        response = mock_client.post(f"/mock/tracks/{track['track_id']}/correct")
        assert response.status_code == 200
    assert response.json()["tp"] == 40
    assert response.json()["fp"] == response.json()["fn"] == 0
    exported = mock_client.get("/mock/export")
    assert "attachment" in exported.headers["content-disposition"]
    assert sum(len(t["shapes"]) for t in exported.json()["tracks"]) == 40


def test_edit_box_is_evaluated_and_rejects_invalid_geometry(mock_client):
    start(mock_client)
    track = mock_client.get("/mock/state").json()["tracks"]["tracks"][0]
    url = f"/mock/tracks/{track['track_id']}/box"
    assert mock_client.post(url, json={"frame": 9, "bbox": {"xtl": 208, "ytl": 270, "xbr": 363, "ybr": 370}}).json()["tp"] == 38
    assert mock_client.post(url, json={"frame": 9, "bbox": {"xtl": 5, "ytl": 5, "xbr": 4, "ybr": 10}}).status_code == 422
    assert mock_client.post(url, json={"frame": 9, "bbox": {"xtl": 1, "ytl": 1, "xbr": 961, "ybr": 20}}).status_code == 422
    assert mock_client.post("/mock/tracks/unknown/correct").status_code == 404


def test_review_state_is_separate_from_annotations(mock_client):
    job = start(mock_client)
    state = mock_client.get("/mock/state").json()
    track = state["tracks"]["tracks"][0]
    base = f"/autolabel/api/v1/video/jobs/{job['job_id']}"
    result = mock_client.post(base + "/review/state", json={"track_id": track["track_id"], "state": "resolved"})
    assert result.status_code == 200 and result.json()["seen"] == 1
    assert mock_client.get("/mock/state").json()["tracks"] == state["tracks"]


def test_re_evaluation_retains_history_and_validates_threshold(mock_client):
    start(mock_client)
    assert mock_client.post("/mock/reevaluate", json={"iou_threshold": .7}).json()["iteration"] == 2
    assert mock_client.post("/mock/reevaluate", json={"iou_threshold": 2}).status_code == 422
    assert len(mock_client.get("/mock/state").json()["history"]) == 2


def test_original_panel_import_contract_and_explicit_offline_limits(mock_client):
    base = "/autolabel/api/v1/video/jobs/cvat-jobs/1"
    assert mock_client.get(base).status_code == 404
    assert mock_client.post(base + "/import", json={"source_fps": 12, "auto_start_sequence": True}).status_code == 200
    assert mock_client.post(base + "/import", json={"source_fps": 12, "auto_start_sequence": True, "sequence_classes": ["boat"]}).status_code == 422
    assert mock_client.post(base + "/import", json={"source_fps": 12, "auto_start_sequence": True, "sam3_model_variant": "sam3.1"}).status_code == 422
    job = mock_client.get("/mock/state").json()["job"]
    assert mock_client.post(f"/autolabel/api/v1/video/jobs/{job['job_id']}/manual-claude").status_code == 503
