"""Scalar validation and canonical JSON, copied from the lab."""
import json
import numpy as np


def _json(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _integer(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)
