"""Where the tutorial's frozen inputs live: tutorial/assets/, or the directory in SBI4F_TUTORIAL_ASSETS."""

import os
from pathlib import Path

ASSETS = Path(os.environ.get("SBI4F_TUTORIAL_ASSETS", Path(__file__).resolve().parents[3] / "assets"))
RESULTS = ASSETS / "results" / "calibration"  # accepted study results
CHECKPOINTS = ASSETS / "checkpoints"  # trained networks
QUICK_CHECKS = ASSETS / "quick-checks"  # outputs of the lab's quick checks, not accepted studies
DATA = ASSETS / "data"  # the exact Heston posterior on SPY
TASKS = ASSETS / "tasks"  # task files the lab stored
IMAGES = ASSETS / "images"
