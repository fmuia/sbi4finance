"""Compatibility imports for the former flat option-lesson helpers.

Edit heston_options.lesson, rough_options.lesson or shared.teaching.
"""

from .shared.teaching import (
    LABELS,
    PRODUCT_NAMES,
    Path,
    SPX_SKEW_EXPONENT,
    TENORS,
    agreement_rows,
    cost_rows,
    design_arrays,
    existing,
    flat_versus_smile,
    implied_vol_grid,
    json,
    load_json,
    load_network,
    load_npz,
    np,
    pairs,
    price_histograms,
    reserve_rows,
    skew_term_structures,
    smile_rows,
    surface,
    warnings,
)
from .heston_options.lesson import (
    _heston_teaching_inputs,
    heston_illustrative_paths,
    heston_posterior_predictive,
    heston_smile,
    heston_variance_swap_vol,
)
from .rough_options.lesson import (
    live_smile_figure,
    price_rough_smile,
)
