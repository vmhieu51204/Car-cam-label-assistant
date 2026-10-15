"""Poll and process persistent video frame-extraction jobs."""

from __future__ import annotations

import argparse
import logging
import time

from src.config import Settings
from src.services.cvat import CvatReadClient
from src.services.storage.repository import Store
from src.services.video import FFmpegVideoExtractor, VideoWorker, VideoWorkspace


def build_worker(settings: Settings) -> VideoWorker:
    store = Store(settings.db_path)
    recovered = store.requeue_expired_video_extractions()
    if recovered:
        logging.warning("requeued %d expired video extraction job(s)", recovered)
    workspace = VideoWorkspace(settings.video_work_root)
    extractor = FFmpegVideoExtractor(
        workspace,
        ffmpeg_path=settings.ffmpeg_path,
        ffprobe_path=settings.ffprobe_path,
        timeout_s=settings.video_ffmpeg_timeout_s,
    )
    return VideoWorker(
        profile_enabled=settings.profile_2d_enabled,
        store=store,
        workspace=workspace,
        extractor=extractor,
        cvat_client=CvatReadClient.from_settings(settings),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Process queued video extraction jobs")
    parser.add_argument("--once", action="store_true", help="Process at most one job, then exit")
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    args = parser.parse_args()
    if args.poll_seconds <= 0 or args.poll_seconds > 60:
        parser.error("--poll-seconds must be greater than 0 and at most 60")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    worker = build_worker(Settings())
    while True:
        job = worker.process_next()
        if job is not None:
            logging.info("video job %s -> %s", job.job_id, job.status.value)
        if args.once:
            return 0
        if job is None:
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
