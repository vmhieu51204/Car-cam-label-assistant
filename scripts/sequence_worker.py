"""Run queued seed-free SAM3 text-prompt video jobs through private Kaggle kernels."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from src.config import Settings
from src.services.cvat import CvatTaskClient, CvatTrackClient
from src.services.cvat.read_client import CvatReadClient
from src.services.storage.repository import Store
from src.services.video.kaggle_batch import KaggleBatchRunner
from src.services.video.local_sam import LocalSamRunner
from src.services.video.seeds import CvatSeedReader
from src.services.video.sequence_worker import SequenceWorker
from src.services.video.service import VideoJobService
from src.services.video.workspace import VideoWorkspace


def build_worker(settings: Settings) -> SequenceWorker:
    if not settings.local_sam_enabled and (not settings.kaggle_api_token.get_secret_value() or not settings.kaggle_sam3_checkpoint_dataset):
        raise ValueError("Kaggle token and SAM3 checkpoint dataset are required")
    workspace = VideoWorkspace(settings.video_work_root)
    store = Store(settings.db_path)
    cvat = CvatReadClient.from_settings(settings)
    return SequenceWorker(
        profile_enabled=settings.profile_2d_enabled,
        stitch_decision_log_enabled=settings.stitch_decision_log_enabled,
        local_runner=LocalSamRunner(settings, workspace.root) if settings.local_sam_enabled else None,
        store=store,
        workspace=workspace,
        cvat=cvat,
        runner=KaggleBatchRunner(
            username=settings.kaggle_username,
            work_root=workspace.root,
            machine_shape=settings.kaggle_machine_shape,
            timeout_s=settings.kaggle_timeout_s,
            poll_seconds=settings.kaggle_poll_seconds,
            sam3_checkpoint_dataset=settings.kaggle_sam3_checkpoint_dataset,
            api_token=settings.kaggle_api_token.get_secret_value(),
            command=str(Path(sys.executable).parent / ("kaggle.exe" if os.name == "nt" else "kaggle")),
        ) if settings.kaggle_username else None,
        notebook=Path(__file__).resolve().parents[1] / "kaggle/notebooks/autolabel_sam3_benchmark.ipynb",
        lease_seconds=settings.video_propagation_lease_s,
        draft_service=VideoJobService(
            store=store,
            workspace=workspace,
            seed_reader=CvatSeedReader(cvat),
            task_client=CvatTaskClient.from_settings(settings),
            track_client=CvatTrackClient.from_settings(settings, read_client=cvat),
            max_upload_bytes=settings.video_max_upload_bytes,
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=1)
    args = parser.parse_args()
    if not 0 < args.poll_seconds <= 60:
        parser.error("--poll-seconds must be in (0, 60]")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = Settings()
    worker = build_worker(settings)
    while True:
        job = worker.process_next()
        if job is not None:
            logging.info("sequence job %s -> %s", job.job_id, job.status.value)
        if args.once:
            return 0
        if job is None:
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
