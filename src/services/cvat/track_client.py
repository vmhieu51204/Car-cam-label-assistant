"""Add pending 2D tracks to CVAT without replacing human annotations."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import httpx
from pydantic import SecretStr

from src.models.video import BoundingBox2D, CvatPublishResult, FrameManifest, TrackSet, VideoTrack
from src.services.cvat.read_client import CvatReadClient, cvat_route_host, valid_job_id
from src.services.cvat.task_client import CvatWriteError

if TYPE_CHECKING:
    from src.config import Settings
    from src.services.video.keyframe_sparsify import SparseConfig


class CvatTrackClient:
    def __init__(
        self,
        *,
        base_url: str,
        token: SecretStr | str,
        read_client: CvatReadClient,
        timeout_s: float = 30,
        host_header: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        sparse_config: SparseConfig | None = None,
    ) -> None:
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("CVAT base URL must be an HTTP(S) URL without embedded credentials")
        if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("CVAT timeout must be a finite positive number")
        secret = token.get_secret_value() if isinstance(token, SecretStr) else token
        self.base_url = base_url.rstrip("/")
        self.authorization = secret if secret.lower().startswith(("token ", "bearer ")) else f"Token {secret}"
        self.read_client = read_client
        self.timeout_s = timeout_s
        self.host_header = host_header
        self.transport = transport
        from src.services.video.keyframe_sparsify import SparseConfig

        self.sparse_config = sparse_config or SparseConfig()

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        read_client: CvatReadClient,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> CvatTrackClient:
        from src.services.video.keyframe_sparsify import SparseConfig

        return cls(
            base_url=settings.cvat_url,
            token=settings.cvat_token,
            read_client=read_client,
            timeout_s=settings.cvat_timeout_s,
            host_header=cvat_route_host(settings),
            transport=transport,
            sparse_config=SparseConfig(
                enabled=settings.sparse_keyframes_enabled,
                min_iou=settings.sparse_keyframes_min_iou,
                max_gap=settings.sparse_keyframes_max_gap,
            ),
        )

    async def publish_tracks(
        self,
        *,
        job_id: str,
        cvat_job_id: int,
        manifest: FrameManifest,
        tracks: TrackSet,
        review_priority_by_track: dict[str, str] | None = None,
        review_score_by_track: dict[str, float] | None = None,
        expected_revision: int | None = None,
        sparse_sidecar: dict[str, Any] | None = None,
        mandatory_sparse_frames: set[int] | None = None,
    ) -> tuple[CvatPublishResult, str]:
        normalized_job_id = valid_job_id(cvat_job_id)
        existing, labels = await asyncio.gather(
            self.read_client.get_job_annotations(normalized_job_id),
            self.read_client.get_job_labels(normalized_job_id),
        )
        revision = _integer(existing.get("version"), "CVAT annotation revision", minimum=0)
        if expected_revision is not None and revision != expected_revision:
            raise CvatWriteError(
                f"stale CVAT annotation revision: expected {expected_revision}, CVAT returned {revision}",
                status_code=409,
            )
        if any(item.get("source") == "auto" for item in _object_list(existing.get("tracks", []), "tracks")):
            raise CvatWriteError(
                "CVAT job already contains automatic tracks; inspect before publishing again", status_code=409
            )
        combined = build_cvat_annotation_payload(
            existing=existing,
            manifest=manifest,
            tracks=tracks,
            label_attributes=_label_attribute_ids(labels),
            mutable_attributes=_label_mutable_names(labels),
            job_id=job_id,
            review_priority_by_track=review_priority_by_track or {},
            review_score_by_track=review_score_by_track or {},
        )
        sidecar = sparse_sidecar if sparse_sidecar is not None else {}
        sidecar.clear()
        if self.sparse_config.enabled and tracks.tracks:
            from src.services.video.keyframe_sparsify import sparsify_shapes

            cuts = {
                f.source_frame_index if f.source_frame_index is not None else f.frame_index
                for f in manifest.frames
                if f.scene_cut_before
            }
            eligible = [
                t
                for t in tracks.tracks
                if any(not (t.track_id.startswith(("shape:", "track:")) and o.frame_index == 0) for o in t.observations)
            ]
            exported = [t for t in combined["tracks"] if t.get("source") == "auto"]
            for item, original in zip(exported, eligible, strict=True):
                # Conservative: preserve every observation when only track-level review evidence exists.
                required = (
                    cuts
                    | (mandatory_sparse_frames or set())
                    | {s["frame"] for s in item["shapes"] if original.needs_review or original.review_reasons}
                )
                # Old artifacts do not persist overlap ordinals; reserve every window/overlap boundary.
                starts = [0] + [f.frame_index for f in manifest.frames if f.scene_cut_before and f.frame_index > 0]
                for start, stop in zip(starts, starts[1:] + [manifest.frame_count], strict=True):
                    required.update(
                        f.source_frame_index if f.source_frame_index is not None else f.frame_index
                        for f in manifest.frames[start:stop]
                        if (f.frame_index - start) % 45 in {0, 1, 2, 3, 4, 44} or f.frame_index == stop - 1
                    )
                dense = item["shapes"]
                item["shapes"], kept = sparsify_shapes(dense, self.sparse_config, mandatory=required)
                sidecar[original.track_id] = {"before": len(dense), "after": len(item["shapes"]), "kept": kept}
        generated_tracks = [item for item in _object_list(combined["tracks"], "tracks") if item.get("source") == "auto"]
        payload = {
            "version": revision,
            "tags": [],
            "shapes": [],
            "tracks": generated_tracks,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        payload_sha = hashlib.sha256(encoded.encode()).hexdigest()
        headers = self._headers()
        try:
            async with httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=self.timeout_s,
                transport=self.transport,
            ) as client:
                response = await client.patch(
                    f"/api/jobs/{normalized_job_id}/annotations",
                    params={"action": "create"},
                    json=payload,
                )
        except httpx.RequestError as error:
            raise CvatWriteError(
                "CVAT annotation create had an unknown outcome; read the job before retrying"
            ) from error
        if not response.is_success:
            detail = f": {response.text[:500]}" if response.status_code == 400 else ""
            raise CvatWriteError(
                f"CVAT annotation create failed with HTTP {response.status_code}{detail}",
                status_code=response.status_code,
            )
        try:
            response_payload = response.json()
        except ValueError as error:
            raise CvatWriteError("CVAT annotation create returned invalid JSON") from error
        if not isinstance(response_payload, dict):
            raise CvatWriteError("CVAT annotation create returned a non-object JSON payload")
        revision_after = _integer(response_payload.get("version"), "updated CVAT annotation revision", minimum=revision)
        published_count = len(generated_tracks)
        return (
            CvatPublishResult(
                job_id=job_id,
                cvat_job_id=normalized_job_id,
                revision_before=revision,
                revision_after=revision_after,
                published_track_count=published_count,
                payload_sha256=payload_sha,
            ),
            encoded,
        )

    def _headers(self) -> dict[str, str]:
        if self.authorization == "Token ":
            raise CvatWriteError("CVAT token is not configured")
        headers = {
            "Authorization": self.authorization,
            "Accept": "application/vnd.cvat+json",
            "Content-Type": "application/json",
        }
        if self.host_header:
            headers["Host"] = self.host_header
        return headers

    async def publish_seed_shapes(
        self,
        *,
        cvat_job_id: int,
        expected_revision: int,
        detections: list[dict[str, Any]],
    ) -> int:
        """Append editable SAM3 suggestions to frame 0, preserving existing CVAT data."""
        normalized_job_id = valid_job_id(cvat_job_id)
        existing = await self.read_client.get_job_annotations(normalized_job_id)
        revision = _integer(existing.get("version"), "CVAT annotation revision", minimum=0)
        if revision != expected_revision:
            raise CvatWriteError("CVAT annotations changed before SAM3 boxes could be added", status_code=409)
        if any(shape.get("frame") == 0 for shape in _object_list(existing.get("shapes", []), "shapes")):
            raise CvatWriteError("frame 0 already contains boxes; inspect them before retrying", status_code=409)
        shapes = []
        for detection in detections:
            box = detection["bbox"]
            shapes.append(
                {
                    "type": "rectangle",
                    "frame": 0,
                    "label_id": detection["label_id"],
                    "points": [box.xtl, box.ytl, box.xbr, box.ybr],
                    "occluded": False,
                    "outside": False,
                    "z_order": 0,
                    "rotation": 0,
                    "group": 0,
                    "source": "auto",
                    "attributes": [],
                }
            )
        payload = {"version": revision, "tags": [], "shapes": shapes, "tracks": []}
        try:
            async with httpx.AsyncClient(
                base_url=self.base_url,
                headers=self._headers(),
                timeout=self.timeout_s,
                transport=self.transport,
            ) as client:
                response = await client.patch(
                    f"/api/jobs/{normalized_job_id}/annotations",
                    params={"action": "create"},
                    json=payload,
                )
        except httpx.RequestError as error:
            raise CvatWriteError("SAM3 boxes may have been added to CVAT; refresh before retrying") from error
        if not response.is_success:
            raise CvatWriteError(
                f"CVAT rejected SAM3 frame-0 boxes (HTTP {response.status_code})", status_code=response.status_code
            )
        try:
            result = response.json()
        except ValueError as error:
            raise CvatWriteError("CVAT returned invalid JSON after adding SAM3 boxes") from error
        return _integer(result.get("version"), "updated CVAT annotation revision", minimum=revision)

    async def accept_rescue_rectangle(
        self,
        *,
        cvat_job_id: int,
        frame: int,
        label_id: int,
        bbox: BoundingBox2D,
        expected_revision: int,
    ) -> int:
        """Append one human-accepted proposal only against the visible CVAT revision.

        This is dormant until benchmark quality justifies proposal UI. A conflicting
        edit or uncertain network outcome must be resolved by refreshing the job.
        """
        normalized_job_id = valid_job_id(cvat_job_id)
        if frame < 0 or label_id < 1:
            raise ValueError("invalid proposal frame or label")
        existing, labels = await asyncio.gather(
            self.read_client.get_job_annotations(normalized_job_id),
            self.read_client.get_job_labels(normalized_job_id),
        )
        revision = _integer(existing.get("version"), "CVAT annotation revision", minimum=0)
        if revision != expected_revision:
            raise CvatWriteError(
                "CVAT annotations changed while rescue batch ran; refresh before accepting", status_code=409
            )
        if label_id not in _label_attribute_ids(labels):
            raise CvatWriteError("proposal class is no longer in this CVAT job", status_code=409)
        points = [bbox.xtl, bbox.ytl, bbox.xbr, bbox.ybr]
        for shape in _object_list(existing.get("shapes", []), "shapes"):
            if (
                shape.get("frame") == frame
                and shape.get("label_id") == label_id
                and shape.get("type") == "rectangle"
                and shape.get("points") == points
            ):
                raise CvatWriteError("proposal box already exists; refresh before accepting", status_code=409)
        payload = {
            "version": revision,
            "tags": [],
            "tracks": [],
            "shapes": [
                {
                    "type": "rectangle",
                    "frame": frame,
                    "label_id": label_id,
                    "points": points,
                    "occluded": False,
                    "outside": False,
                    "z_order": 0,
                    "rotation": 0,
                    "group": 0,
                    "source": "auto",
                    "attributes": [],
                }
            ],
        }
        try:
            async with httpx.AsyncClient(
                base_url=self.base_url, headers=self._headers(), timeout=self.timeout_s, transport=self.transport
            ) as client:
                response = await client.patch(
                    f"/api/jobs/{normalized_job_id}/annotations", params={"action": "create"}, json=payload
                )
        except httpx.RequestError as error:
            raise CvatWriteError("rescue create outcome unknown; refresh CVAT before retrying") from error
        if not response.is_success:
            raise CvatWriteError(
                f"CVAT rejected rescue box (HTTP {response.status_code})", status_code=response.status_code
            )
        try:
            returned = response.json()
        except ValueError as error:
            raise CvatWriteError("CVAT returned invalid JSON after rescue create") from error
        return _integer(returned.get("version"), "updated CVAT annotation revision", minimum=revision)

    async def accept_rescue_track(
        self,
        *,
        cvat_job_id: int,
        frame: int,
        label_id: int,
        bbox: BoundingBox2D,
        expected_revision: int,
        proposal_id: str | None = None,
        occluded: bool = False,
        box_extent: str = "visible",
    ) -> int:
        """Create one-frame editable CVAT track only after explicit user acceptance."""
        normalized_job_id = valid_job_id(cvat_job_id)
        if frame < 0 or label_id < 1:
            raise ValueError("invalid proposal frame or label")
        existing, labels = await asyncio.gather(
            self.read_client.get_guarded_job_annotations(normalized_job_id),
            self.read_client.get_job_labels(normalized_job_id),
        )
        revision = _integer(existing.get("autolabel_revision"), "CVAT annotation content revision", minimum=0)
        attr_ids = _label_attribute_ids(labels).get(label_id, {})
        identity = f"vlm-proposal:{proposal_id}" if proposal_id else None
        if not identity:
            raise CvatWriteError("proposal identity is required for atomic acceptance", status_code=409)
        if box_extent == "amodal_estimate" and "box_extent" not in attr_ids:
            raise CvatWriteError(
                "CVAT label lacks box_extent attribute; restore AutoLabel attributes before accepting", status_code=409
            )
        if identity and "autolabel_track_id" not in attr_ids:
            raise CvatWriteError(
                "CVAT label lacks proposal identity attribute; restore AutoLabel attributes before accepting",
                status_code=409,
            )
        if identity and "autolabel_track_id" in attr_ids:
            for track in _object_list(existing.get("tracks", []), "tracks"):
                if track.get("label_id") == label_id and any(
                    attr.get("spec_id") == attr_ids["autolabel_track_id"] and attr.get("value") == identity
                    for attr in track.get("attributes", [])
                ):
                    return revision
        if revision != expected_revision:
            raise CvatWriteError(
                "CVAT annotations changed while rescue batch ran; refresh before accepting", status_code=409
            )
        if label_id not in _label_attribute_ids(labels):
            raise CvatWriteError("proposal class is no longer in this CVAT job", status_code=409)
        points = [bbox.xtl, bbox.ytl, bbox.xbr, bbox.ybr]
        for track in _object_list(existing.get("tracks", []), "tracks"):
            if track.get("label_id") != label_id:
                continue
            if any(
                shape.get("frame") == frame and shape.get("points") == points and not shape.get("outside", False)
                for shape in _object_list(track.get("shapes", []), "track shapes")
            ):
                raise CvatWriteError("proposal track already exists; refresh before accepting", status_code=409)
        shapes = [
            {
                "frame": frame,
                "type": "rectangle",
                "occluded": occluded,
                "outside": False,
                "keyframe": True,
                "points": points,
                "z_order": 0,
                "attributes": [],
            }
        ]
        # Bound a single-frame suggestion so CVAT does not imply unobserved motion.
        task = await self.read_client.get_job(normalized_job_id)
        stop_frame = task.get("stop_frame")
        if isinstance(stop_frame, int) and frame < stop_frame:
            shapes.append(
                {
                    "frame": frame + 1,
                    "type": "rectangle",
                    "occluded": occluded,
                    "outside": True,
                    "keyframe": True,
                    "points": points,
                    "z_order": 0,
                    "attributes": [],
                }
            )
        payload: dict[str, Any] = {
            "version": _integer(existing.get("version"), "CVAT native version", minimum=0),
            "tags": [],
            "shapes": [],
            "tracks": [
                {
                    "frame": frame,
                    "label_id": label_id,
                    "group": 0,
                    "source": "auto",
                    "attributes": [
                        {"spec_id": attr_ids[name], "value": value}
                        for name, value in {
                            "autolabel_track_id": identity,
                            "review_status": "pending",
                            "review_priority": "high",
                            "source": "auto",
                            "routing": "review",
                            "box_extent": box_extent,
                        }.items()
                        if name in attr_ids and value is not None
                    ],
                    "shapes": shapes,
                }
            ],
        }
        mutable_ids = {attr_ids[name] for name in _label_mutable_names(labels).get(label_id, set())}
        attributes = payload["tracks"][0]["attributes"]
        payload["tracks"][0]["attributes"] = [a for a in attributes if a["spec_id"] not in mutable_ids]
        for shape in shapes:
            shape["attributes"] = [a for a in attributes if a["spec_id"] in mutable_ids]
        try:
            async with httpx.AsyncClient(
                base_url=self.base_url, headers=self._headers(), timeout=self.timeout_s, transport=self.transport
            ) as client:
                response = await client.patch(
                    f"/api/jobs/{normalized_job_id}/annotations",
                    params={"action": "create"},
                    json=payload,
                    headers={
                        "X-Autolabel-Expected-Revision": str(expected_revision),
                        "X-Autolabel-Proposal-Id": proposal_id or "",
                    },
                )
        except httpx.RequestError as error:
            raise CvatWriteError("rescue track create outcome unknown; refresh CVAT before retrying") from error
        if not response.is_success:
            raise CvatWriteError(
                f"CVAT rejected rescue track (HTTP {response.status_code})", status_code=response.status_code
            )
        try:
            result = response.json()
        except ValueError as error:
            raise CvatWriteError("CVAT returned invalid JSON after rescue track create") from error
        return _integer(result.get("autolabel_revision"), "updated CVAT annotation content revision", minimum=0)


def build_cvat_annotation_payload(
    *,
    existing: dict[str, Any],
    manifest: FrameManifest,
    tracks: TrackSet,
    label_attributes: dict[int, dict[str, int]] | None = None,
    mutable_attributes: dict[int, set[str]] | None = None,
    job_id: str | None = None,
    review_priority_by_track: dict[str, str] | None = None,
    review_score_by_track: dict[str, float] | None = None,
) -> dict[str, Any]:
    if tracks.job_id != manifest.job_id:
        raise ValueError("CVAT publish inputs belong to different video jobs")
    version = _integer(existing.get("version"), "CVAT annotation revision", minimum=0)
    tags = _object_list(existing.get("tags", []), "tags")
    shapes = _object_list(existing.get("shapes", []), "shapes")
    existing_tracks = _object_list(existing.get("tracks", []), "tracks")
    manual_tracks = [item for item in existing_tracks if item.get("source") != "auto"]
    generated = [
        _cvat_track(
            track,
            manifest,
            attribute_ids=(label_attributes or {}).get(track.label_id, {}),
            mutable_names=(mutable_attributes or {}).get(track.label_id, set()),
            job_id=job_id or tracks.job_id,
            review_priority=(review_priority_by_track or {}).get(track.track_id, "normal"),
            review_score=(review_score_by_track or {}).get(track.track_id),
        )
        for track in tracks.tracks
    ]
    generated = [item for item in generated if item is not None]
    return {
        "version": version,
        "tags": tags,
        "shapes": shapes,
        "tracks": manual_tracks + generated,
    }


def _cvat_track(
    track: VideoTrack,
    manifest: FrameManifest,
    *,
    attribute_ids: dict[str, int],
    mutable_names: set[str],
    job_id: str,
    review_priority: str,
    review_score: float | None,
) -> dict[str, Any] | None:
    observations = [
        item
        for item in track.observations
        if not (track.track_id.startswith(("shape:", "track:")) and item.frame_index == 0)
    ]
    if not observations:
        return None
    attributes = _track_attributes(track, attribute_ids, job_id, review_priority, review_score)
    mutable_ids = {attribute_ids[name] for name in mutable_names if name in attribute_ids}
    track_attributes = [item for item in attributes if item["spec_id"] not in mutable_ids]
    shape_attributes = [item for item in attributes if item["spec_id"] in mutable_ids]
    shapes: list[dict[str, object]] = []
    for observation in observations:
        if observation.frame_index >= manifest.frame_count:
            raise ValueError(f"track {track.track_id} references a frame outside the manifest")
        frame = manifest.frames[observation.frame_index]
        box = observation.bbox
        if box.xbr > frame.width or box.ybr > frame.height:
            raise ValueError(f"track {track.track_id} exceeds frame {observation.frame_index} bounds")
        shapes.append(
            {
                "type": "rectangle",
                "frame": frame.source_frame_index if frame.source_frame_index is not None else observation.frame_index,
                "points": [box.xtl, box.ytl, box.xbr, box.ybr],
                "outside": observation.outside,
                "occluded": observation.occluded,
                "keyframe": True,
                "z_order": 0,
                "rotation": 0.0,
                "attributes": shape_attributes,
                "source": "auto",
            }
        )
    return {
        "frame": (
            manifest.frames[observations[0].frame_index].source_frame_index
            if manifest.frames[observations[0].frame_index].source_frame_index is not None
            else observations[0].frame_index
        ),
        "label_id": track.label_id,
        "group": 0,
        "source": "auto",
        "attributes": track_attributes,
        "shapes": shapes,
    }


def _track_attributes(
    track: VideoTrack,
    attribute_ids: dict[str, int],
    job_id: str,
    review_priority: str,
    review_score: float | None,
) -> list[dict[str, object]]:
    required = {"confidence", "routing", "source", "review_status", "autolabel_track_id", "review_priority"}
    if attribute_ids and not required <= attribute_ids.keys():
        missing = ", ".join(sorted(required - attribute_ids.keys()))
        raise CvatWriteError(f"CVAT label {track.label_id} is missing AutoLabel attributes: {missing}")
    if not attribute_ids:
        return []
    scores = [item.score for item in track.observations if item.score is not None]
    confidence = sum(scores) / len(scores) if scores else 0.0
    values = {
        "confidence": f"{confidence:.3f}",
        "routing": "review",
        "source": "auto",
        "review_status": track.review_status,
        "autolabel_track_id": f"{job_id}:{track.track_id}",
        "review_priority": review_priority,
    }
    if "review_score" in attribute_ids and review_score is not None:
        values["review_score"] = f"{review_score:.3f}"
    return [{"spec_id": attribute_ids[name], "value": values[name]} for name in sorted(values)]


def _label_attribute_ids(labels: list[dict[str, Any]]) -> dict[int, dict[str, int]]:
    result: dict[int, dict[str, int]] = {}
    for label in labels:
        label_id = _integer(label.get("id"), "CVAT label ID", minimum=1)
        raw_attributes = label.get("attributes", [])
        if not isinstance(raw_attributes, list) or any(not isinstance(item, dict) for item in raw_attributes):
            raise CvatWriteError(f"CVAT label {label_id} has invalid attribute specifications")
        attributes: dict[str, int] = {}
        for item in raw_attributes:
            name = item.get("name")
            if not isinstance(name, str) or not name:
                raise CvatWriteError(f"CVAT label {label_id} has an invalid attribute name")
            attributes[name] = _integer(item.get("id"), "CVAT attribute specification ID", minimum=1)
        result[label_id] = attributes
    return result


def _label_mutable_names(labels: list[dict[str, Any]]) -> dict[int, set[str]]:
    return {
        _integer(label.get("id"), "CVAT label ID", minimum=1): {
            item["name"]
            for item in _object_list(label.get("attributes", []), "label attributes")
            if item.get("mutable") is True and isinstance(item.get("name"), str)
        }
        for label in labels
    }


def _object_list(value: object, name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise CvatWriteError(f"CVAT annotations contain an invalid {name} list")
    return [dict(item) for item in value]


def _integer(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CvatWriteError(f"{name} is missing or invalid")
    return value
