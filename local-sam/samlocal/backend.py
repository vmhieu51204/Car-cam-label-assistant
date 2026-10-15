"""SAM 3 and SAM 3.1 backends tuned for 16 GB-class GPUs.

``VideoBackend`` holds everything the two models share (the facebookresearch/sam3 predictor
API is the same for both: ``start_session`` / ``add_prompt`` / ``propagate_in_video``);
``Sam3Backend`` and ``Sam31Backend`` only differ in how the predictor is built and tuned.

Optimisations over the Kaggle T4 kernel (``kaggle/sam3_benchmark/main.py``):

* SAM 3.1 only: multiplex predictor (``build_sam3_predictor(version="sam3.1")``): objects
  are tracked jointly in buckets instead of one-by-one.
* Frames are decoded once per window; each extra prompt only does ``reset_session``
  (keeps ``input_batch``) instead of a fresh ``init_state`` that re-reads every JPEG.
* Forward-only propagation from the prompt frame (the API default "both" also runs a
  useless backward pass when the prompt is on frame 0).
* ``evict_cached_frame_outputs=True`` + immediate CPU reduction of masks.
* Hard VRAM cap via ``set_per_process_memory_fraction`` so Windows/WSL never silently
  spills into "shared GPU memory" (system RAM); OOM is raised and the scheduler backs off.
* Batched grounding chunk capped (``--grounding-batch``, default 2): the predictor runs the
  detector backbone on up to 16 frames at once and keeps every frame's FPN features, which
  costs ~0.7 GiB/frame and was what forced windows down to 12 frames on 16 GB cards.
* After an OOM the traceback frames are cleared before ``empty_cache`` so the retry starts
  with the failed window's tensors actually released.
* ``use_fa3`` only on Hopper (sm_90+): FlashAttention 3 does not exist for Ada (RTX 40xx).
* bf16 autocast + TF32 come from the official predictor (good for Ampere/Ada).
"""

from __future__ import annotations

import gc
import inspect
import logging
import time
import traceback
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from samlocal.compat import install_shared_conditioning_guard
from samlocal.models import SAM3, SAM31, ModelSpec
from samlocal.stitch import Detection

GIB = 1024**3
BOX_AUDIT_IOU = 0.5  # model box vs final-mask box below this IoU counts as a disagreement (0.9 flagged
                     # tiny boxes that only differ by sam3's exclusive max-pixel convention)
SPECK_MIN_PIXELS = 8  # mask components smaller than this, or than SPECK_FRACTION of the largest one, are dropped
SPECK_FRACTION = 0.02
SPARSE_MASK_FILL = 0.2  # mask pixels / own box area below this: several distant blobs, box likely inflated
BOX_AUDIT_EXAMPLES = 50


class WindowOOMError(RuntimeError):
    """CUDA ran out of memory while processing one window."""


def is_oom(exc: BaseException) -> bool:
    import torch

    oom_type = getattr(torch, "OutOfMemoryError", None) or torch.cuda.OutOfMemoryError
    if isinstance(exc, oom_type):
        return True
    text = str(exc).lower()
    return isinstance(exc, RuntimeError) and ("out of memory" in text or "cublas_status_alloc_failed" in text
                                              or "cudnn_status_alloc_failed" in text)


def mask_rle(mask: np.ndarray) -> dict:
    """Row-major run-length encoding, same format as the Kaggle kernel's ``_rle``."""
    height, width = mask.shape
    flat = mask.reshape(-1).astype(np.uint8)
    transitions = np.flatnonzero(np.diff(flat)) + 1
    counts = np.diff(np.concatenate(([0], transitions, [flat.size]))).tolist()
    if flat.size and flat[0]:
        counts.insert(0, 0)
    return {"height": int(height), "width": int(width), "counts": counts}


class VideoBackend:
    spec: ModelSpec

    def __init__(self, *, checkpoint: Path | None, compile_model: bool = False, offload_video: str = "auto",
                 async_frames: bool = True, vram_fraction: float = 0.90, prob_threshold: float = 0.5,
                 keep_masks: bool = False, logger: logging.Logger | None = None) -> None:
        self.checkpoint = checkpoint
        self.compile_model = compile_model
        self.offload_video_mode = offload_video
        self.async_frames = async_frames
        self.vram_fraction = vram_fraction
        self.prob_threshold = prob_threshold
        self.keep_masks = keep_masks
        self.log = logger or logging.getLogger("local-sam")
        self.predictor: Any = None
        self.torch: Any = None
        self.offload_video = False
        self.budget_bytes = 0
        self.total_bytes = 0
        self.session_fallback_used = False
        self.offload_state = False  # SAM 3 only; SAM 3.1's init_state has no such option
        # Accumulates over warm-up and every repeat; written into benchmark.json -> model.bbox_audit.
        self.bbox_audit: dict[str, Any] = {
            "bbox_source": "final_mask", "differs_if_iou_below": BOX_AUDIT_IOU,
            "sparse_if_fill_below": SPARSE_MASK_FILL, "speck_min_pixels": SPECK_MIN_PIXELS,
            "speck_fraction_of_largest": SPECK_FRACTION, "by_label": {}, "examples": []}
        # Also accumulates; written into benchmark.json -> model.object_cap. The upstream cap counts every masklet
        # added to a session (one prompt in one window), including objects that already left the frame.
        self.object_cap: dict[str, Any] = {
            "max_objects": None, "limit_warnings": 0, "prompt_windows_at_cap": 0,
            "max_ids_per_prompt_window": {}, "examples": []}

    @property
    def max_objects_cap(self) -> int | None:
        """Upstream ``max_num_objects`` per session; None = no limit (SAM 3)."""
        return None

    # Hooks implemented by the model-specific subclasses.
    def build_kwargs(self, props: Any) -> dict[str, Any]:
        raise NotImplementedError

    def configure(self) -> dict[str, Any]:
        """Tune the freshly built predictor; returns extra fields for ``model`` in the report."""
        return {}

    def describe(self) -> str:
        return ""

    # ------------------------------------------------------------------ load
    def load(self) -> dict[str, Any]:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available: check the NVIDIA driver / torch CUDA build")
        self.torch = torch
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        self.total_bytes = props.total_memory
        if 0 < self.vram_fraction < 1:
            # Allocator raises OOM before the WDDM driver can page into shared system RAM.
            torch.cuda.set_per_process_memory_fraction(self.vram_fraction, device)
        self.budget_bytes = int(self.total_bytes * min(self.vram_fraction, 1.0))
        if self.offload_video_mode == "auto":
            self.offload_video = self.total_bytes <= 24 * GIB
        else:
            self.offload_video = self.offload_video_mode == "on"

        from sam3.model_builder import build_sam3_predictor

        kwargs = self.build_kwargs(props)
        accepted = inspect.signature(build_sam3_predictor).parameters
        if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in accepted.values()):
            kwargs = {k: v for k, v in kwargs.items() if k in accepted}
        self.log.info("building %s predictor (%s, %.1f GiB VRAM, budget %.1f GiB, offload_video=%s%s)",
                      self.spec.label, props.name, self.total_bytes / GIB, self.budget_bytes / GIB,
                      self.offload_video, self.describe())
        self.object_cap["max_objects"] = self.max_objects_cap
        _install_cap_counter(self.object_cap)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        self.predictor = build_sam3_predictor(**kwargs)
        extra = self.configure()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        info = {"model_version": self.spec.key, "model_load_seconds": seconds,
                "weights_allocated_gib": torch.cuda.memory_allocated() / GIB,
                "load_peak_reserved_gib": torch.cuda.max_memory_reserved() / GIB,
                "offload_video_to_cpu": self.offload_video,
                "vram_budget_gib": self.budget_bytes / GIB, "total_vram_gib": self.total_bytes / GIB,
                "bbox_audit": self.bbox_audit, "object_cap": self.object_cap, **extra}
        self.log.info("model loaded in %.1fs, weights use %.2f GiB", seconds, info["weights_allocated_gib"])
        return info

    # --------------------------------------------------------------- session
    def _start_session(self, frame_dir: Path) -> str:
        request = {"type": "start_session", "resource_path": str(frame_dir),
                   "offload_video_to_cpu": self.offload_video, "offload_state_to_cpu": self.offload_state}
        if not self.session_fallback_used:
            try:
                return self.predictor.handle_request(request)["session_id"]
            except TypeError as exc:
                # Sam3BasePredictor.start_session forwards ``offload_state_to_cpu``, which the
                # multiplex model's init_state does not accept on some commits.
                if "unexpected keyword" not in str(exc):
                    raise
                self.log.warning("start_session incompatible (%s); using signature-filtered init_state", exc)
                self.session_fallback_used = True
        model = self.predictor.model
        kwargs = {"resource_path": str(frame_dir), "offload_video_to_cpu": self.offload_video,
                  "offload_state_to_cpu": self.offload_state, "async_loading_frames": self.async_frames}
        accepted = inspect.signature(model.init_state).parameters
        with self.torch.inference_mode():
            state = model.init_state(**{k: v for k, v in kwargs.items() if k in accepted})
        session_id = str(uuid.uuid4())
        now = time.time()
        self.predictor._all_inference_states[session_id] = {
            "state": state, "session_id": session_id, "start_time": now, "last_use_time": now}
        return session_id

    def _close_session(self, session_id: str | None) -> None:
        if session_id is None:
            return
        try:
            self.predictor.handle_request({"type": "close_session", "session_id": session_id,
                                           "run_gc_collect": True})
        except Exception as exc:  # never mask the original error
            self.log.debug("close_session failed: %s", exc)

    def cleanup(self) -> None:
        gc.collect()
        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    # ---------------------------------------------------------------- window
    def run_window(self, frame_dir: Path, num_frames: int, prompts: list[str], *, frame_offset: int,
                   on_prompt_done: Callable[[str, float], None] | None = None) -> dict[str, list[Detection]]:
        """Run every prompt over one staged window. Raises ``WindowOOMError`` on CUDA OOM."""
        torch = self.torch
        session_id: str | None = None
        results: dict[str, list[Detection]] = {}
        active_prompt = None
        last_frame = None
        oom_message: str | None = None
        try:
            session_id = self._start_session(frame_dir)
            for position, prompt in enumerate(prompts):
                active_prompt = prompt
                last_frame = 0
                started = time.perf_counter()
                if position:
                    self.predictor.handle_request({"type": "reset_session", "session_id": session_id})
                by_frame: dict[int, list[Detection]] = {}
                response = self.predictor.handle_request({
                    "type": "add_prompt", "session_id": session_id, "frame_index": 0, "text": prompt,
                    "output_prob_thresh": self.prob_threshold})
                by_frame[0] = self._detections(response.get("outputs"), 0, frame_offset, prompt)
                for response in self.predictor.handle_stream_request({
                        "type": "propagate_in_video", "session_id": session_id,
                        "propagation_direction": "forward", "start_frame_index": 0,
                        "max_frame_num_to_track": num_frames, "output_prob_thresh": self.prob_threshold,
                        "evict_cached_frame_outputs": True}):
                    local = int(response["frame_index"])
                    last_frame = local
                    if 0 <= local < num_frames:
                        by_frame[local] = self._detections(response.get("outputs"), local, frame_offset, prompt)
                torch.cuda.synchronize()
                results[prompt] = [det for local in sorted(by_frame) for det in by_frame[local]]
                self._count_objects(prompt, results[prompt], frame_offset, num_frames)
                if on_prompt_done:
                    on_prompt_done(prompt, time.perf_counter() - started)
            return results
        except Exception as exc:
            if not is_oom(exc):
                self.log.error("window failed: prompt=%r, last yielded local frame=%s, global offset=%d: %s",
                               active_prompt, last_frame, frame_offset, exc)
                raise
            # The traceback keeps the propagation generators' frames (and their GPU tensors)
            # alive; drop them so the cleanup below can actually return the memory.
            oom_message = str(exc).splitlines()[0]
            _release_traceback(exc)
        finally:
            self._close_session(session_id)
            if oom_message is None:
                self.cleanup()
        # Only reached after an OOM: the except block has exited, so nothing pins the tensors.
        self.cleanup()
        raise WindowOOMError(oom_message)

    def _detections(self, outputs: dict | None, local: int, frame_offset: int, label: str) -> list[Detection]:
        if not isinstance(outputs, dict) or not outputs:
            return []
        obj_ids = np.asarray(outputs.get("out_obj_ids", []))
        if obj_ids.size == 0:
            return []
        probs = np.asarray(outputs.get("out_probs", np.ones(len(obj_ids))), dtype=np.float32)
        masks = outputs.get("out_binary_masks")
        boxes = outputs.get("out_boxes_xywh")
        height = width = None
        if masks is not None:
            masks = np.asarray(masks)
            height, width = masks.shape[-2:]
        detections = []
        for row, obj_id in enumerate(obj_ids.tolist()):
            mask = masks[row].astype(bool) if masks is not None else None
            model_box = None
            if boxes is not None and width:
                x, y, w, h = (float(v) for v in np.asarray(boxes[row]).reshape(-1)[:4])
                if w > 0 and h > 0:
                    model_box = (x * width, y * height, (x + w) * width, (y + h) * height)
            # sam3 computes out_boxes_xywh from the masks BEFORE the non-overlapping constraint, but returns
            # out_binary_masks AFTER it. With SAM 3.1 multiplex a pre-constraint mask can still hold pixels
            # that belong to other objects, inflating the box to the frame edge. Use the final mask.
            if mask is not None:
                mask, specks = drop_mask_specks(mask)
                bbox = _mask_bbox(mask)
                self._audit_box(label, frame_offset + local, int(obj_id), model_box, bbox, mask, specks)
            else:
                bbox = model_box
            if bbox is None:
                continue
            detections.append(Detection(
                frame=frame_offset + local, local_id=int(obj_id), label=label, bbox=bbox,
                score=float(probs[row]) if row < len(probs) else 1.0,
                mask_rle=mask_rle(mask) if self.keep_masks and mask is not None else None))
        return detections

    def _count_objects(self, prompt: str, detections: list[Detection], frame_offset: int, num_frames: int) -> None:
        """Distinct object ids one prompt produced in one window; reaching the cap means objects may be missing."""
        ids = len({det.local_id for det in detections})
        cap = self.object_cap
        cap["max_ids_per_prompt_window"][prompt] = max(ids, cap["max_ids_per_prompt_window"].get(prompt, 0))
        if cap["max_objects"] is not None and ids >= cap["max_objects"]:
            cap["prompt_windows_at_cap"] += 1
            if len(cap["examples"]) < BOX_AUDIT_EXAMPLES:
                cap["examples"].append({"prompt": prompt, "first_frame": frame_offset, "frames": num_frames,
                                        "ids": ids})
            self.log.warning("prompt %r reached the object cap (%d ids, max_objects=%d) in window starting at %d: "
                             "raise --max-objects", prompt, ids, cap["max_objects"], frame_offset)

    def _audit_box(self, label: str, frame: int, obj_id: int, model_box: tuple | None, mask_box: tuple | None,
                   mask: np.ndarray, specks: int = 0) -> None:
        """Count how often the model's box disagrees with the cleaned final mask, masks that carried stray
        specks, and masks still mostly empty inside their own box, with a few examples to look up."""
        audit = self.bbox_audit
        row = audit["by_label"].setdefault(label, {"compared": 0, "model_box_differs": 0, "empty_final_mask": 0,
                                                   "sparse_mask": 0, "masks_with_specks": 0, "specks_removed": 0})
        row["compared"] += 1
        row["masks_with_specks"] += specks > 0
        row["specks_removed"] += specks
        if mask_box is None:
            row["empty_final_mask"] += 1
            return
        area = (mask_box[2] - mask_box[0]) * (mask_box[3] - mask_box[1])
        fill = float(mask.sum()) / area if area else 0.0
        differs = model_box is not None and _box_iou(model_box, mask_box) < BOX_AUDIT_IOU
        row["model_box_differs"] += differs
        row["sparse_mask"] += fill < SPARSE_MASK_FILL
        if (differs or fill < SPARSE_MASK_FILL) and len(audit["examples"]) < BOX_AUDIT_EXAMPLES:
            audit["examples"].append({"frame": frame, "label": label, "obj_id": obj_id, "mask_fill": round(fill, 3),
                                      "model_box": [round(v, 1) for v in model_box] if model_box else None,
                                      "mask_box": [round(v, 1) for v in mask_box]})

    # --------------------------------------------------------------- memory
    def reset_peak(self) -> None:
        self.torch.cuda.reset_peak_memory_stats()

    def peak(self) -> dict[str, float]:
        torch = self.torch
        allocated = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
        return {"peak_allocated_gib": allocated / GIB, "peak_reserved_gib": reserved / GIB,
                "peak_fraction_of_budget": reserved / self.budget_bytes if self.budget_bytes else None}

    def close(self) -> None:
        if self.predictor is not None:
            try:
                self.predictor.shutdown()
            except Exception:
                pass
        self.predictor = None
        self.cleanup()


class Sam3Backend(VideoBackend):
    """SAM 3: detector + per-object tracker, one frame at a time (no multiplex buckets)."""

    spec = SAM3

    def __init__(self, *, checkpoint: Path | None, offload_state: bool = False, **kwargs: Any) -> None:
        super().__init__(checkpoint=checkpoint, **kwargs)
        self.offload_state = offload_state

    def build_kwargs(self, props: Any) -> dict[str, Any]:
        # build_sam3_predictor(version="sam3") forwards only these to Sam3VideoPredictor;
        # multiplex / FA3 / rope / warm-up options belong to SAM 3.1 and are not accepted.
        return {"checkpoint_path": self.checkpoint.as_posix() if self.checkpoint else None, "version": "sam3",
                "compile": False, "async_loading_frames": self.async_frames}

    def configure(self) -> dict[str, Any]:
        return {"offload_state_to_cpu": self.offload_state}

    def describe(self) -> str:
        return f", offload_state={self.offload_state}"


class Sam31Backend(VideoBackend):
    """SAM 3.1 Object Multiplex: objects are tracked jointly in buckets of ``multiplex_count``."""

    spec = SAM31

    def __init__(self, *, checkpoint: Path | None, max_objects: int = 16, multiplex_count: int = 16,
                 grounding_batch: int = 2, **kwargs: Any) -> None:
        super().__init__(checkpoint=checkpoint, **kwargs)
        if grounding_batch < 1:
            raise ValueError("grounding_batch must be >= 1")
        self.max_objects = max_objects
        self.multiplex_count = multiplex_count
        self.grounding_batch = grounding_batch
        self.use_fa3 = False

    def build_kwargs(self, props: Any) -> dict[str, Any]:
        self.use_fa3 = props.major >= 9 and _has_fa3()
        return {
            "checkpoint_path": self.checkpoint.as_posix() if self.checkpoint else None,
            "version": "sam3.1",
            "compile": self.compile_model,
            # The predictor's own warm-up runs inside its constructor, before configure() can cap the grounding
            # batch, so it used upstream's 16-frame chunk and ran out of memory on 16 GB (RTX 4080 run
            # 20261013-191926). configure() runs the same warm-up after the cap instead.
            "warm_up": False,
            "max_num_objects": self.max_objects,
            "multiplex_count": self.multiplex_count,
            "use_fa3": self.use_fa3,
            "use_rope_real": True,
            "async_loading_frames": self.async_frames,
            "default_output_prob_thresh": self.prob_threshold,
        }

    def configure(self) -> dict[str, Any]:
        compatibility = install_shared_conditioning_guard(self.predictor.model.tracker, self.log)
        self.log.info("SAM3 shared-conditioning guard: %s", compatibility)
        grounding = configure_grounding_batch(self.predictor.model, self.grounding_batch)
        self.log.info("SAM3 batched grounding: %s", grounding)
        warm_up_seconds = None
        if self.compile_model:
            # same steps as Sam3MultiplexVideoPredictor.__init__(warm_up=True); counted in model_load_seconds
            model = self.predictor.model
            started = time.perf_counter()
            self.log.info("torch.compile warm-up (30 dummy frames, grounding batch %d) ...", self.grounding_batch)
            model._warm_up_complete = False
            model.warm_up_compilation()
            model._warm_up_complete = True
            self.torch.cuda.synchronize()
            warm_up_seconds = time.perf_counter() - started
            self.log.info("torch.compile warm-up done in %.1fs", warm_up_seconds)
        return {"use_fa3": self.use_fa3, "shared_conditioning_guard": compatibility,
                "batched_grounding": grounding, "compile_warm_up_seconds": warm_up_seconds}

    def describe(self) -> str:
        return (f", fa3={self.use_fa3}, compile={self.compile_model}, max_objects={self.max_objects}")

    @property
    def max_objects_cap(self) -> int | None:
        return self.max_objects


def build_backend_for(spec: ModelSpec, **kwargs: Any) -> VideoBackend:
    return {SAM3.key: Sam3Backend, SAM31.key: Sam31Backend}[spec.key](**kwargs)


def configure_grounding_batch(model: Any, batch_size: int) -> dict[str, Any]:
    """Cap the detector's batched-grounding chunk; ``build_sam3_predictor`` hard-codes 16."""
    info: dict[str, Any] = {"requested_batch_size": batch_size,
                            "use_batched_grounding": getattr(model, "use_batched_grounding", None),
                            "upstream_batch_size": getattr(model, "batched_grounding_batch_size", None)}
    if info["upstream_batch_size"] is None:
        info["applied"] = False
        info["reason"] = "model has no batched_grounding_batch_size attribute"
        return info
    model.batched_grounding_batch_size = batch_size
    info["applied"] = True
    info["batch_size"] = batch_size
    return info


class _CapWarningCounter(logging.Handler):
    """Counts upstream "hitting max_num_objects" / "cannot add a new object" warnings (detections dropped)."""

    def __init__(self, target: dict[str, Any]) -> None:
        super().__init__(logging.WARNING)
        self.target = target

    def emit(self, record: logging.LogRecord) -> None:
        if "max_num_objects" in record.getMessage():
            self.target["limit_warnings"] += 1


def _install_cap_counter(target: dict[str, Any]) -> None:
    from samlocal.logs import UPSTREAM_LOGGERS

    for name in UPSTREAM_LOGGERS:
        upstream = logging.getLogger(name)
        upstream.handlers = [h for h in upstream.handlers if not isinstance(h, _CapWarningCounter)]
        upstream.addHandler(_CapWarningCounter(target))


def _release_traceback(exc: BaseException) -> None:
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if exc.__traceback__ is not None:
            traceback.clear_frames(exc.__traceback__)
            exc.__traceback__ = None
        exc = exc.__cause__ or exc.__context__


def _box_iou(a: tuple, b: tuple) -> float:
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def drop_mask_specks(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Remove tiny disconnected components; returns (mask, number of components removed).

    SAM 3.1 multiplex masks (RTX 4080 run 20261013-104910) carry 1-2 pixel specks on a fixed grid
    (x = 4 or 846, y = 32, 52, 112, ... at 854x480) next to the real object. One speck at the frame edge
    stretches the box across the image. Real parts of an object split by an occluder stay: only components
    under max(SPECK_MIN_PIXELS, SPECK_FRACTION x largest) go.
    """
    rows = np.flatnonzero(mask.any(axis=1))
    if not rows.size:
        return mask, 0
    cols = np.flatnonzero(mask.any(axis=0))
    y0, y1, x0, x1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    crop = mask[y0:y1, x0:x1]
    from scipy import ndimage

    labels, count = ndimage.label(crop)
    if count <= 1:
        return mask, 0
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    keep = sizes >= max(SPECK_MIN_PIXELS, SPECK_FRACTION * sizes.max())
    keep[0] = False
    removed = int(count - (keep.sum()))
    if not removed:
        return mask, 0
    cleaned = np.zeros_like(mask)
    cleaned[y0:y1, x0:x1] = keep[labels]
    return cleaned, removed


def _mask_bbox(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if not rows.size or not cols.size:
        return None
    return float(cols[0]), float(rows[0]), float(cols[-1] + 1), float(rows[-1] + 1)


def _has_fa3() -> bool:
    try:
        import flash_attn_interface  # noqa: F401
    except ImportError:
        return False
    return True
