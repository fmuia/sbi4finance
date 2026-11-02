"""Compatibility API for tutorial-owned lesson helpers and archived model loading.

Numerical and presentation implementations are edited in the application
packages under tutorial/src/sbi4f_tutorial/. The historical BF2.0 loader below
remains a lab-only API for recorded studies.
"""

import json
from pathlib import Path
import warnings

# Shared implementations are owned by the tutorial runtime.
from sbi4f_tutorial.shared.teaching import (
    LABELS,
    TENORS,
    PRODUCT_NAMES,
    SPX_SKEW_EXPONENT,
    load_json,
    load_npz,
    existing,
    design_arrays,
    smile_rows,
    flat_versus_smile,
    skew_term_structures,
    pairs,
    reserve_rows,
    price_histograms,
    agreement_rows,
    cost_rows,
    implied_vol_grid,
)
from sbi4f_tutorial.rough_options.lesson import (
    price_rough_smile,
    live_smile_figure,
)
from sbi4f_tutorial.heston_options.lesson import (
    heston_smile,
    heston_variance_swap_vol,
    _heston_teaching_inputs,
    heston_illustrative_paths,
    heston_posterior_predictive,
)


def load_network(checkpoint):
    """(task, workflow) from a run directory with model.keras, task.json and training.json."""
    from . import inference

    checkpoint = Path(checkpoint)
    task = json.loads((checkpoint / "task.json").read_text())
    record = json.loads((checkpoint / "training.json").read_text())
    workflow = inference.build_workflow(task, subnet_widths=tuple(record["training"]["subnet_widths"]))
    with warnings.catch_warnings():
        # accepted checkpoints carry no optimizer state (studies/accept.py); sampling never needs it
        warnings.filterwarnings("ignore", message="Skipping variable loading for optimizer")
        workflow.load_approximator(checkpoint / "model.keras")
    return task, workflow
