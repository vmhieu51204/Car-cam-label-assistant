"""SAM 3.1: Run once and save predictions (shared code lives in ../samlocal).

    python run_sam31.py --payload ~/data/payload.zip --checkpoint ~/models/sam3.1_multiplex.pt --visualize 10
"""

import sys
from pathlib import Path

# Appended (not inserted first) so the facebookresearch `sam3` package, not a folder of this repo, wins imports.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from samlocal import configure_env  # noqa: E402

configure_env()

from samlocal.runner import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main("sam3.1"))
