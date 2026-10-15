from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.models.discrepancy import HCLConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # Optional measurement only; never changes inference/annotation behavior.
    profile_2d_enabled: bool = False

    # Export/review experiments; uncalibrated and disabled by default.
    track_filter_enabled: bool = False
    stitch_decision_log_enabled: bool = True
    track_filter_rules: str = "{}"
    sparse_keyframes_enabled: bool = False
    sparse_keyframes_min_iou: float = Field(default=0.9, gt=0, le=1, allow_inf_nan=False)
    sparse_keyframes_max_gap: int = Field(default=15, ge=1)
    track_review_enabled: bool = False
    track_review_weights: str = "{}"
    claude_canvas_overlay_enabled: bool = False
    docked_panel_enabled: bool = False

    # App
    app_name: str = "AutoLabel 3D"
    app_env: Literal["development", "production", "test"] = "development"
    app_port: int = Field(default=8000, ge=1, le=65535)
    app_host: str = "127.0.0.1"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    cors_origins: str = "http://localhost:3000"

    # LLM
    openai_api_key: str = ""
    model_name: str = "gpt-4o-mini"
    llm_temperature: float = Field(default=0.7, ge=0.0, le=2.0)

    # Database
    database_url: str = "sqlite:///./data/app.db"

    # Vector Store
    chroma_persist_dir: str = "./data/chroma"

    data_root: Path = Field(
        default=PROJECT_ROOT.parent / "data/nuscenes", validation_alias=AliasChoices("DATA_DIR", "RELABEL_DATA_ROOT")
    )
    data_version: str = Field(
        default="v1.0-mini", validation_alias=AliasChoices("DATA_VERSION", "RELABEL_DATA_VERSION")
    )
    db_path: Path = Field(
        default=PROJECT_ROOT / "state/autolabel.sqlite3", validation_alias=AliasChoices("DB_PATH", "RELABEL_DB_PATH")
    )
    video_work_root: Path = Field(
        default=PROJECT_ROOT / "state/video-jobs",
        validation_alias=AliasChoices("VIDEO_WORK_DIR", "RELABEL_VIDEO_WORK_DIR"),
    )
    video_max_upload_bytes: int = Field(
        default=2_147_483_648,
        gt=0,
        validation_alias=AliasChoices("VIDEO_MAX_UPLOAD_BYTES", "RELABEL_VIDEO_MAX_UPLOAD_BYTES"),
    )
    video_ffmpeg_timeout_s: float = Field(
        default=3600,
        gt=0,
        le=86400,
        validation_alias=AliasChoices("VIDEO_FFMPEG_TIMEOUT_S", "RELABEL_VIDEO_FFMPEG_TIMEOUT_S"),
    )
    ffmpeg_path: str = Field(default="ffmpeg", validation_alias=AliasChoices("FFMPEG_PATH", "RELABEL_FFMPEG_PATH"))
    ffprobe_path: str = Field(default="ffprobe", validation_alias=AliasChoices("FFPROBE_PATH", "RELABEL_FFPROBE_PATH"))
    propagation_provider: Literal["kaggle_sam3"] = "kaggle_sam3"
    kaggle_username: str = ""
    kaggle_api_token: SecretStr = SecretStr("")
    kaggle_machine_shape: str = "NvidiaTeslaT4"
    kaggle_timeout_s: float = Field(default=21600, gt=0, le=86400)
    kaggle_poll_seconds: float = Field(default=20, gt=0, le=300)
    sam3_model_id: str = "facebook/sam3"
    kaggle_sam3_checkpoint_dataset: str = ""
    local_sam_enabled: bool = False
    # Ubuntu workflow is opt-in; old jobs retain their provider and contracts.
    vm_workflow_enabled: bool = False
    hybrid_enabled: bool = False
    lidar3d_enabled: bool = False
    bevfusion_enabled: bool = False
    lidar3d_dataset_root: Path = Path("/opt/autolabel/datasets/nuscenes")
    lidar3d_manifest_sha256: str = ""
    lidar3d_service_url: str = "http://127.0.0.1:8013"
    lidar3d_max_upload_bytes: int = Field(default=1_073_741_824, gt=0)
    lidar3d_max_expanded_bytes: int = Field(default=2_147_483_648, gt=0)
    hybrid_yolo_checkpoint: Path | None = None
    hybrid_yolo_sha256: str = ""
    hybrid_classes_manifest: Path | None = None
    hybrid_classes_sha256: str = ""
    hybrid_class_mapping: str = '{"car":2,"person":0}'
    vm_work_root: Path = PROJECT_ROOT / "state/vm-jobs"
    local_frame_cache_max_bytes: int = Field(default=536870912, ge=1048576)
    privacy_python: Path = Path(".venv-privacy/bin/python")
    privacy_face_checkpoint: Path | None = None
    privacy_plate_checkpoint: Path | None = None
    privacy_face_sha256: str = ""
    privacy_plate_sha256: str = ""
    privacy_source_commit: str = ""
    privacy_timeout_s: float = Field(default=21600, gt=0, le=86400)
    vlm_settings_encryption_key: SecretStr = SecretStr("")
    vlm_allowed_internal_hosts: str = ""
    # Opt-in shared key for the default VLM (Claude Sonnet 5.5 via OpenRouter) used by accounts
    # without their own key. Read at request time only; never sealed into profiles or job snapshots.
    vlm_default_api_key: SecretStr = SecretStr("")
    local_sam_commit: str = ""
    local_sam_python: Path = Path(".venv-sam/bin/python")
    local_sam3_checkpoint: Path | None = None
    local_sam31_checkpoint: Path | None = None
    local_sam3_sha256: str = ""
    local_sam31_sha256: str = ""
    local_sam_timeout_s: float = Field(default=21600, gt=0, le=86400)
    vlm_box_rescue_enabled: bool = True
    vlm_box_rescue_model: Literal["anthropic/claude-sonnet-5.5"] = "anthropic/claude-sonnet-5.5"
    openrouter_api_key: SecretStr = Field(
        default=SecretStr(""), validation_alias=AliasChoices("OPENROUTER_API_KEY", "VLM_API_KEY")
    )
    vlm_box_rescue_fewshot_manifest: Path = Field(
        default=PROJECT_ROOT / "state/claude-fewshot/manifest.json",
        validation_alias=AliasChoices("VLM_BOX_RESCUE_FEWSHOT_MANIFEST"),
    )
    vlm_box_rescue_retry_attempts: int = Field(default=3, ge=1, le=3)
    vlm_box_rescue_retry_base_delay_s: float = Field(default=1.0, ge=0, le=60)
    vlm_box_rescue_retry_max_delay_s: float = Field(default=60.0, gt=0, le=3600)
    b2_endpoint_url: str = ""
    b2_bucket: str = ""
    b2_access_key_id: str = ""
    b2_secret_access_key: SecretStr = SecretStr("")
    video_propagation_lease_s: float = Field(
        default=120,
        ge=3,
        le=3600,
        validation_alias=AliasChoices("VIDEO_PROPAGATION_LEASE_S", "RELABEL_VIDEO_PROPAGATION_LEASE_S"),
    )
    detector_url: str = Field(
        default="http://127.0.0.1:8020", validation_alias=AliasChoices("DETECTOR_URL", "RELABEL_DETECTOR_URL")
    )
    detector_model_id: str = Field(
        default="PekingU/rtdetr_r18vd",
        validation_alias=AliasChoices("DETECTOR_MODEL_ID", "RELABEL_DETECTOR_MODEL_ID"),
    )
    detector_score_threshold: float = Field(
        default=0.35,
        ge=0,
        le=1,
        validation_alias=AliasChoices("DETECTOR_SCORE_THRESHOLD", "RELABEL_DETECTOR_SCORE_THRESHOLD"),
    )
    detector_timeout_s: float = Field(
        default=3600,
        gt=0,
        le=86400,
        validation_alias=AliasChoices("DETECTOR_TIMEOUT_S", "RELABEL_DETECTOR_TIMEOUT_S"),
    )
    video_discovery_lease_s: float = Field(
        default=120,
        ge=3,
        le=3600,
        validation_alias=AliasChoices("VIDEO_DISCOVERY_LEASE_S", "RELABEL_VIDEO_DISCOVERY_LEASE_S"),
    )
    video_audit_lease_s: float = Field(
        default=120,
        ge=3,
        le=3600,
        validation_alias=AliasChoices("VIDEO_AUDIT_LEASE_S", "RELABEL_VIDEO_AUDIT_LEASE_S"),
    )
    video_audit_periodic_stride: int = Field(
        default=30,
        ge=1,
        le=10000,
        validation_alias=AliasChoices("VIDEO_AUDIT_PERIODIC_STRIDE", "RELABEL_VIDEO_AUDIT_PERIODIC_STRIDE"),
    )
    video_audit_max_frames: int = Field(
        default=20,
        ge=1,
        le=50,
        validation_alias=AliasChoices("VIDEO_AUDIT_MAX_FRAMES", "RELABEL_VIDEO_AUDIT_MAX_FRAMES"),
    )
    video_audit_max_regions: int = Field(
        default=20,
        ge=0,
        le=20,
        validation_alias=AliasChoices("VIDEO_AUDIT_MAX_REGIONS", "RELABEL_VIDEO_AUDIT_MAX_REGIONS"),
    )
    cvat_url: str = Field(
        default="http://localhost:8080", validation_alias=AliasChoices("CVAT_URL", "RELABEL_CVAT_URL")
    )
    cvat_public_url: str = Field(
        default="http://localhost:8080",
        validation_alias=AliasChoices("CVAT_PUBLIC_URL", "RELABEL_CVAT_PUBLIC_URL"),
    )
    cvat_token: SecretStr = Field(
        default=SecretStr(""), validation_alias=AliasChoices("CVAT_TOKEN", "RELABEL_CVAT_TOKEN")
    )
    cvat_timeout_s: float = Field(
        default=30, gt=0, le=300, validation_alias=AliasChoices("CVAT_TIMEOUT_S", "RELABEL_CVAT_TIMEOUT_S")
    )
    cvat_retry_attempts: int = Field(
        default=3, ge=1, le=10, validation_alias=AliasChoices("CVAT_RETRY_ATTEMPTS", "RELABEL_CVAT_RETRY_ATTEMPTS")
    )
    cvat_retry_delay_s: float = Field(
        default=0.25,
        ge=0,
        le=10,
        validation_alias=AliasChoices("CVAT_RETRY_DELAY_S", "RELABEL_CVAT_RETRY_DELAY_S"),
    )
    cvat_task_ready_timeout_s: float = Field(
        default=120,
        gt=0,
        le=1800,
        validation_alias=AliasChoices("CVAT_TASK_READY_TIMEOUT_S", "RELABEL_CVAT_TASK_READY_TIMEOUT_S"),
    )
    mock_seed: int = Field(default=42, ge=0, validation_alias=AliasChoices("MOCK_SEED", "RELABEL_MOCK_SEED"))
    mock_position_std: float = Field(
        default=0.15,
        ge=0,
        allow_inf_nan=False,
        validation_alias=AliasChoices("MOCK_POSITION_STD", "RELABEL_MOCK_POSITION_STD"),
    )
    mock_miss_rate: float = Field(
        default=0.20, ge=0, le=1, validation_alias=AliasChoices("MOCK_MISS_RATE", "RELABEL_MOCK_MISS_RATE")
    )
    mock_false_rate: float = Field(
        default=0.10, ge=0, le=1, validation_alias=AliasChoices("MOCK_FALSE_RATE", "RELABEL_MOCK_FALSE_RATE")
    )
    mock_class_error_rate: float = Field(
        default=0.05,
        ge=0,
        le=1,
        validation_alias=AliasChoices("MOCK_CLASS_ERROR_RATE", "RELABEL_MOCK_CLASS_ERROR_RATE"),
    )
    mock_yaw_std: float = Field(
        default=0.08, ge=0, allow_inf_nan=False, validation_alias=AliasChoices("MOCK_YAW_STD", "RELABEL_MOCK_YAW_STD")
    )
    match_giou: float = Field(
        default=0.0, ge=-1, le=1, validation_alias=AliasChoices("MATCH_GIOU", "RELABEL_MATCH_GIOU")
    )
    max_missed: int = Field(default=2, ge=0, validation_alias=AliasChoices("MAX_MISSED", "RELABEL_MAX_MISSED"))
    publish_requires_review: bool = True
    vlm_timeout_s: float = Field(
        default=45, gt=0, le=300, validation_alias=AliasChoices("VLM_TIMEOUT_S", "RELABEL_VLM_TIMEOUT_S")
    )
    vlm_cache_path: Path = Field(
        default=PROJECT_ROOT / "state/vlm-cache.sqlite3",
        validation_alias=AliasChoices("VLM_CACHE_PATH", "RELABEL_VLM_CACHE_PATH"),
    )
    hcl_config: HCLConfig = Field(default_factory=HCLConfig)

    @model_validator(mode="after")
    def require_review_in_production(self) -> "Settings":
        if self.app_env == "production" and not self.publish_requires_review:
            raise ValueError("production must require human review before training-label publication")
        return self

    @field_validator("data_root", "db_path", "vlm_cache_path", "video_work_root", "vlm_box_rescue_fewshot_manifest")
    @classmethod
    def resolve_project_path(cls, value: Path) -> Path:
        """Interpret relative runtime paths from the repository, never the caller's cwd."""
        return (PROJECT_ROOT / value).resolve() if not value.is_absolute() else value


@lru_cache
def get_settings() -> Settings:
    """Return one cached environment-derived settings object."""
    return Settings()
