"""Prepare a fixed return window for the participant lessons, without downloading data.

Run from lab/, then publish the new output through studies/accept.py --run.
The source is an existing dated simple-return CSV; its exact bytes are hashed.
"""

import argparse
import csv
from datetime import date
import hashlib
import json
from pathlib import Path

import numpy as np


SOURCE_REVISION = "069d5dc5943dbe85505849f2fa8bfb563617ff97"


def prepare(source, destination, *, ticker="VOO", end="2021-01-07", count=256):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError("Use a new output directory; preserved examples are immutable")
    end = date.fromisoformat(end).isoformat()
    if not isinstance(count, int) or isinstance(count, bool) or count < 2:
        raise ValueError("count must be an integer of at least two returns")
    rows = []
    with source.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"date", ticker} <= set(reader.fieldnames or ()):
            raise ValueError("source needs date and selected ticker columns")
        previous = None
        for row in reader:
            current = date.fromisoformat(row["date"]).isoformat()
            if previous is not None and current <= previous:
                raise ValueError("source dates must be strictly increasing and unique")
            previous = current
            if current <= end:
                rows.append((current, row[ticker]))
    if len(rows) < count:
        raise ValueError("source does not cover the requested return window")
    rows = rows[-count:]
    dates = np.asarray([row[0] for row in rows], dtype="U10")
    simple = np.asarray([float(row[1]) for row in rows], dtype=np.float64)
    if not np.isfinite(simple).all() or np.any(simple <= -1):
        raise ValueError("every selected simple return must be finite and above -100%")
    logs = np.log1p(simple)
    destination.mkdir(parents=True)
    np.savez_compressed(destination / "returns.npz", dates=dates,
                        simple_returns=simple, log_returns=logs)
    metadata = {
        "format": "fixed_market_return_example_v1", "ticker": ticker, "n": count,
        "start_date": str(dates[0]), "end_date": str(dates[-1]), "requested_end": end,
        "source": "Adjusted-close simple returns cached by amortized-fractals; its loader uses Yahoo Finance. No fresh download.",
        "source_repository": "https://github.com/stefanradev93/amortized-fractals",
        "source_revision": SOURCE_REVISION, "source_file": source.name,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_observation": "decimal simple returns from the preserved upstream cache",
        "log_conversion": "numpy.log1p(simple_returns), float64; no recentering, scaling or filtering",
        "selection": f"last {count} recorded observations on or before {end}",
        "scope": "selected empirical motivation and model-adequacy example; not a forecasting benchmark",
        "arrays_sha256": hashlib.sha256((destination / "returns.npz").read_bytes()).hexdigest(),
    }
    (destination / "market.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ticker", default="VOO")
    parser.add_argument("--end", default="2021-01-07")
    parser.add_argument("--count", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output, ticker=args.ticker,
                             end=args.end, count=args.count), indent=2))


if __name__ == "__main__":
    main()
