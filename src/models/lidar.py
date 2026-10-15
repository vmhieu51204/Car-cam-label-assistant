"""Lightweight metric detector interchange, shared by isolated CUDA runtimes."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

Finite = Annotated[float, Field(allow_inf_nan=False)]
Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class LidarPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sample_token: str = Field(min_length=1)
    frame_index: int = Field(ge=0)
    label: str = Field(min_length=1)
    score: Probability
    center_lidar_xyz_m: tuple[Finite, Finite, Finite]
    size_lwh_m: tuple[Positive, Positive, Positive]
    yaw_lidar_rad: Finite
