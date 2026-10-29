"""Export the audited final500 Heston candidate or prepare its small offline cache."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbi4finance.calibration.full_surface_teaching import main

if __name__ == "__main__":
    main()
