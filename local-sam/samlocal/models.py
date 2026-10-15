"""Per-model settings. ``sam3/`` and ``sam3.1/`` entry scripts only pick one of these specs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str  # ``--model``-style id, also the ``version`` passed to ``build_sam3_predictor``
    slug: str  # file-system friendly id used in script and log-folder names
    label: str
    hf_repo: str
    hf_checkpoint: str
    precision_policy: str
    multiplex: bool  # SAM 3.1 Object Multiplex (objects tracked jointly in buckets)
    supports_compile: bool
    supports_offload_state: bool


SAM3 = ModelSpec(
    key="sam3", slug="sam3", label="SAM 3", hf_repo="facebook/sam3", hf_checkpoint="sam3.pt",
    precision_policy="official SAM3 predictor autocast (bf16) in pinned upstream source",
    multiplex=False, supports_compile=False, supports_offload_state=True)

SAM31 = ModelSpec(
    key="sam3.1", slug="sam31", label="SAM 3.1", hf_repo="facebook/sam3.1", hf_checkpoint="sam3.1_multiplex.pt",
    precision_policy="official SAM3 predictor autocast; bf16/TF32 policy in pinned upstream source",
    multiplex=True, supports_compile=True, supports_offload_state=False)

SPECS = {spec.key: spec for spec in (SAM3, SAM31)}


def get_spec(key: str) -> ModelSpec:
    try:
        return SPECS[key]
    except KeyError:
        raise ValueError(f"unknown model {key!r}; choose from {sorted(SPECS)}") from None
