"""Independent Claude worker; never runs SAM3 or writes CVAT annotations."""

from __future__ import annotations

import argparse
import logging
import time

from src.config import Settings
from src.services.storage.repository import Store
from src.services.video.box_rescue_worker import BoxRescueWorker
from src.services.video.claude_box_rescue import ClaudeBoxRescueAuditor
from src.services.video.workspace import VideoWorkspace


def build_worker(settings: Settings) -> BoxRescueWorker:
    return BoxRescueWorker(
        profile_enabled=settings.profile_2d_enabled,
        store=Store(settings.db_path),
        workspace=VideoWorkspace(settings.video_work_root),
        auditor=ClaudeBoxRescueAuditor(
            api_key=settings.openrouter_api_key.get_secret_value(),
            cache_path=settings.vlm_cache_path,
            model=settings.vlm_box_rescue_model,
            fewshot_manifest=settings.vlm_box_rescue_fewshot_manifest,
            timeout_s=settings.vlm_timeout_s,
            retry_attempts=settings.vlm_box_rescue_retry_attempts,
            retry_base_delay_s=settings.vlm_box_rescue_retry_base_delay_s,
            retry_max_delay_s=settings.vlm_box_rescue_retry_max_delay_s,
        ),
        lease_seconds=settings.video_audit_lease_s,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    settings = Settings()
    if not settings.vlm_box_rescue_enabled:
        return 0
    worker = build_worker(settings)
    while True:
        processed = worker.process_next()
        if args.once:
            return 0
        if processed is None:
            time.sleep(2)


if __name__ == "__main__":
    raise SystemExit(main())
