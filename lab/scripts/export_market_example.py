"""Prepare a fixed return window for the participant lessons, without downloading data.

Run from lab/, then publish the new output through studies/accept.py --run.
The source is a dated simple-return CSV, or a downloaded Yahoo chart response
with a SHA-256 download receipt (--source-kind yahoo-chart --fetch-record FILE).
Exact source bytes are hashed; this exporter itself performs no network access.
"""

import argparse
import csv
from datetime import date
import hashlib
import json
from pathlib import Path

import numpy as np


SOURCE_REVISION = "069d5dc5943dbe85505849f2fa8bfb563617ff97"


def prepare_chart(source, destination, *, fetch_record):
    """Preserve every adjusted-price return in one dated public chart response.

    Downloading is an explicit lab step. The participant notebook only reads the
    accepted snapshot. Missing prices are errors, never silently removed rows.
    """
    source, destination, fetch_record = map(Path, (source, destination, fetch_record))
    if destination.exists():
        raise FileExistsError("Use a new output directory; preserved examples are immutable")
    receipt = json.loads(fetch_record.read_text())
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if receipt.get("raw_sha256") != source_hash:
        raise ValueError("download receipt and source response differ")
    response = json.loads(source.read_text())["chart"]
    if response.get("error") is not None or len(response.get("result") or []) != 1:
        raise ValueError("expected one successful market chart result")
    result = response["result"][0]
    if result["meta"]["symbol"] != "VOO":
        raise ValueError("this lesson snapshot requires VOO")
    timestamps = np.asarray(result["timestamp"], dtype=np.int64)
    prices = np.asarray(result["indicators"]["adjclose"][0]["adjclose"], dtype=np.float64)
    if (len(timestamps) < 3 or prices.shape != timestamps.shape
            or not np.all(np.diff(timestamps) > 0)
            or not np.isfinite(prices).all() or np.any(prices <= 0)):
        raise ValueError("every adjusted close must be positive, finite and uniquely ordered")
    if (receipt.get("count") != len(timestamps)
            or receipt.get("first_timestamp") != int(timestamps[0])
            or receipt.get("last_timestamp") != int(timestamps[-1])):
        raise ValueError("download receipt observation axes differ")
    price_dates = timestamps.astype("datetime64[s]").astype("datetime64[D]").astype("U10")
    if not np.all(price_dates[1:] > price_dates[:-1]):
        raise ValueError("expected one close per trading date")
    simple = prices[1:] / prices[:-1] - 1
    logs = np.log1p(simple)
    if not np.isfinite(logs).all():
        raise ValueError("adjusted prices produced a nonfinite return")
    destination.mkdir(parents=True)
    np.savez_compressed(destination / "returns.npz", dates=price_dates[1:],
                        simple_returns=simple, log_returns=logs,
                        price_dates=price_dates, adjusted_close=prices)
    metadata = {
        "format": "fixed_market_return_example_v2", "ticker": "VOO", "n": len(logs),
        "start_date": str(price_dates[1]), "end_date": str(price_dates[-1]),
        "price_start_date": str(price_dates[0]), "n_prices": len(prices),
        "source": "Yahoo Finance daily chart; adjusted close including corporate-action adjustments",
        "source_url": receipt["url"], "fetched_at_utc": receipt["fetched_at_utc"],
        "source_file": source.name, "source_sha256": source_hash,
        "fetch_record_sha256": hashlib.sha256(fetch_record.read_bytes()).hexdigest(),
        "source_observation": "daily adjusted closing prices; every returned row retained",
        "log_conversion": "simple=P[t]/P[t-1]-1; log=numpy.log1p(simple), float64",
        "selection": "all dated closes in the recorded response; no filtering, recentering or scaling",
        "scope": "dated empirical motivation; no neural market inference or forecasting claim",
        "arrays_sha256": hashlib.sha256((destination / "returns.npz").read_bytes()).hexdigest(),
    }
    (destination / "market.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return metadata


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
    parser.add_argument("--source-kind", choices=("csv", "yahoo-chart"), default="csv")
    parser.add_argument("--fetch-record", type=Path)
    args = parser.parse_args()
    if args.source_kind == "yahoo-chart":
        if args.fetch_record is None:
            parser.error("--fetch-record is required for a downloaded chart")
        result = prepare_chart(args.source, args.output, fetch_record=args.fetch_record)
    else:
        result = prepare(args.source, args.output, ticker=args.ticker, end=args.end, count=args.count)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
