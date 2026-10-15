"""AutoLabel graph nodes call domain modules without network or CVAT writes."""

from src.agents.autolabel_state import AutoLabelState
from src.models.discrepancy import DetectorEvidence, FrameEvidence, TrackEvidence
from src.models.perception import Track
from src.services.discrepancy import DiscrepancyFilter
from src.services.pipeline import detect_tracks


def detect_node(state: AutoLabelState) -> dict:
    """Run the configured local mock detector and tracker."""
    metadata, tracks = detect_tracks(state["settings"], state["request"])
    return {"metadata": metadata, "tracks": tracks}


def mock_evidence(track: Track) -> TrackEvidence:
    """Expose only observed mock boxes; never create camera/LiDAR or calibrated evidence."""
    return TrackEvidence(
        track_id=track.track_id,
        label=track.label,
        data_source="mock",
        frames=[
            FrameEvidence(
                frame_index=item.frame_index,
                timestamp_s=item.timestamp_s,
                label=track.label,
                box=item.box,
                detector=DetectorEvidence(raw_score=item.confidence),
            )
            for item in track.observations
        ],
    )


def assess_node(state: AutoLabelState) -> dict:
    """Attach HCL evidence snapshots while retaining pending human review."""
    evaluator = DiscrepancyFilter(state["settings"].hcl_config)
    tracks = []
    for track in state["tracks"]:
        result = evaluator.evaluate(mock_evidence(track))
        tracks.append(track.model_copy(update={"discrepancy": result.model_dump(mode="json")}))
    metadata = state["metadata"] | {
        "hcl_config": state["settings"].hcl_config.model_dump(),
        "hcl_calibrated": False,
        "evidence_mode": "mock_boxes_only",
        "vlm_escalation": "not_integrated_without_image_pipeline",
    }
    return {"tracks": tracks, "metadata": metadata}
