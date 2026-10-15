"""Run the opted-in privacy-first pipeline on one Ubuntu GPU."""

import argparse
import logging
import time

from src.config import Settings
from src.services.vm.worker import VmWorker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    worker = VmWorker(Settings())
    while True:
        job = worker.process_next()
        if job:
            logging.info("VM job %s: %s", job["job_id"], job["status"])
        if args.once:
            return
        if not job:
            time.sleep(1)


if __name__ == "__main__":
    main()
