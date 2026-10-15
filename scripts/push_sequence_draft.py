"""Append an offline sequence draft to a CVAT job as pending editable tracks."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import dotenv_values

from src.services.video.publish_draft import SequenceDraftConflictError, push_sequence_draft


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--cvat-job-id", type=int, required=True)
    parser.add_argument("--cvat-url", default="http://localhost:8080")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()
    token = os.getenv("CVAT_TOKEN") or dotenv_values(args.env_file).get("CVAT_TOKEN")
    if not token:
        parser.error("CVAT_TOKEN is missing; set it in the environment or --env-file")
    draft = json.loads(args.draft.read_text(encoding="utf-8"))
    try:
        receipt = asyncio.run(
            push_sequence_draft(base_url=args.cvat_url, token=token, cvat_job_id=args.cvat_job_id, draft=draft)
        )
    except SequenceDraftConflictError as error:
        print(f"CVAT draft rejected: {error}", file=sys.stderr)
        raise SystemExit(2) from None
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
