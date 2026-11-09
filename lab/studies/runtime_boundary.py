"""Check that lab APIs reuse the authoritative tutorial runtime.

This is read-only. Edit shared implementations under tutorial/src/sbi4f_tutorial/;
there is no code-copy or synchronization step. Historical bs_hierarchy and
long_end studies retain their original input globals and are checked numerically
by test_tutorial_parity instead of joining this reuse contract.
"""

import argparse
import ast
import importlib
from pathlib import Path


# Lab compatibility API -> authoritative runtime module -> imported symbols.
REUSED_SYMBOLS = {
    "sbi4finance.calibration.correction": {
        "sbi4f_tutorial.shared.parameter_transforms": (
            "bounds_of", "to_unbounded", "to_bounded", "log_jacobian",
        ),
    },
    "sbi4finance.calibration.pricer_net": {
        "sbi4f_tutorial.rough_options.surrogate": (
            "features", "_log_known_curve", "quote_noise_mse", "targets", "Pricer", "train", "errors",
        ),
    },
    "sbi4finance.bergomi": {
        "sbi4f_tutorial.rough_options.model": (
            "_positive_integer", "_scalar", "curve_values", "_model_parameters", "_normals",
            "_q_inputs", "_asset_paths", "conditional_put_payoffs",
        ),
    },
    "sbi4finance.gbm": {
        "sbi4f_tutorial.shared.gbm": (
            "_vector", "_scalar", "_normal_cdf", "european_price",
        ),
    },
    "sbi4finance.gbm_bank": {
        "sbi4f_tutorial.shared.bank_helpers": (
            "_json", "_integer",
        ),
    },
    "sbi4finance.calibration.full_surface": {
        "sbi4f_tutorial.shared.full_surface": (
            "FAMILIES", "ENCODERS", "validate_task", "uses_known_curve", "observations",
            "conditioning_data", "training_data", "fresh_noise", "configuration", "build_workflow",
            "sample", "load_checkpoint",
        ),
    },
    "sbi4finance.calibration.full_surface_teaching": {
        "sbi4f_tutorial.shared.full_surface_teaching": (
            "NAMES", "CACHE_FILES", "ROUGH_NAMES", "NEURAL_FILES", "digest", "fingerprint", "require",
            "heston_task", "variance_products", "load_cache", "rough_task", "load_neural_replay",
            "load_tutorial_artifact", "load_tutorial_checkpoint", "rough_variance_products",
        ),
    },
    "sbi4finance.hkde": {
        "sbi4f_tutorial.heston_options.model": (
            "PARAMETERS", "validate_parameters", "_compensation", "characteristic_function",
            "_quadrature", "european_price",
        ),
    },
    "sbi4finance.calibration.inference": {
        "sbi4f_tutorial.shared.inference": (
            "EULER_STEPS", "SAMPLER_STEPS", "SAMPLER_DESCRIPTION", "sampler_times", "use_sampler",
            "conditioning_data", "build_workflow", "sample", "_checked_task",
        ),
    },
    "sbi4finance.svj.plots": {
        "sbi4f_tutorial.shared.plots": (
            "SURFACE", "INK", "INK_2", "MUTED", "GRID", "BASELINE", "SERIES", "CRITICAL", "style",
        ),
    },
    "sbi4finance.calibration.pricing": {
        "sbi4f_tutorial.shared.pricing": (
            "_PAIR_CHUNK", "step_curve", "_block_contracts", "rough_block", "rough_surface",
            "heston_surface", "heston_variance_values", "heston_integrated_variance", "heston_variance_curve",
        ),
    },
    "sbi4finance.rough_bergomi": {
        "sbi4f_tutorial.rough_options.simulator": (
            "hybrid_weights", "_reference_factors", "rough_covariance", "rough_driver",
            "simulate_rough",
        ),
    },
    "sbi4finance.calibration.surface": {
        "sbi4f_tutorial.shared.surface": (
            "TRADING_DAYS", "DESIGN_DAYS", "DESIGN_DELTAS", "SIGMA_REF", "NOISE_ATM", "NOISE_WING",
            "_d1", "bs_call", "bs_put", "bs_price", "bs_vega", "implied_vol", "design_grid",
            "contract_arrays", "standardized_moneyness", "implied_vol_noise", "quote_noise",
            "smile_statistics", "skew_power_law",
        ),
    },
    "sbi4finance.calibration.tasks": {
        "sbi4f_tutorial.shared.tasks": (
            "knots", "context_size", "sigma", "model_parameters", "_heston_split", "heston_prices",
            "curve", "FAMILY", "JOINT_ROUGH_ID", "JOINT_ROUGH_PARAMETERS", "HKDE_PARAMETERS",
            "UNIVERSE", "TASK_KEYS", "CURVE_KINDS", "is_calibration_task", "is_joint_curve_task",
            "_positive_int", "_interval", "validate_context", "validate_task", "_validate_joint_task",
            "curve_kind", "validate_curve_prior", "MAX_CURVE_DRAWS", "interval_midpoints", "draw_context",
            "_joint_coefficients", "derived_context", "joint_curve",
        ),
    },
    "sbi4finance.calibration.teaching": {
        "sbi4f_tutorial.shared.teaching": (
            "LABELS", "TENORS", "PRODUCT_NAMES", "SPX_SKEW_EXPONENT", "load_json", "load_npz",
            "existing", "design_arrays", "smile_rows", "flat_versus_smile", "skew_term_structures",
            "pairs", "reserve_rows", "price_histograms", "agreement_rows", "cost_rows",
            "implied_vol_grid",
        ),
        "sbi4f_tutorial.rough_options.lesson": (
            "price_rough_smile", "live_smile_figure",
        ),
        "sbi4f_tutorial.heston_options.lesson": (
            "heston_smile", "heston_variance_swap_vol", "_heston_teaching_inputs",
            "heston_illustrative_paths", "heston_posterior_predictive",
        ),
    },
    "sbi4finance.path_inference.inference": {
        "sbi4f_tutorial.trajectories.inference": (
            "FORMAT", "DEFAULT_NETWORK", "LOGISTIC_SD", "PREFIX_POLICIES", "BACKEND_SEED_MAPPING",
            "fingerprint", "sha256", "integer", "backend_seed", "validate_task", "_bounds",
            "inverse_parameters", "conditioning_data", "build_workflow", "sample", "load_checkpoint",
            "transform_parameters", "training_data", "draw_prefix_lengths", "random_prefixes",
        ),
    },
    "sbi4finance.heston": {
        "sbi4f_tutorial.heston_options.simulator": (
            "_PARAMETERS", "_dt", "_batch", "_parameters", "_unsigned", "draw_innovations",
            "_numpy_steps", "_numba_steps", "simulate",
        ),
    },
    "sbi4finance.path_inference.bank": {
        "sbi4f_tutorial.trajectories.simulation": (
            "MODELS", "SPLITS", "TEACHING_LANDMARKS", "TEACHING_TASK_IDS", "_integer", "_json", "_rng",
            "default_task", "generate", "task_fingerprint", "teaching_task", "validate_task",
        ),
    },
}


def differences():
    """Return every missing, duplicated or replaced shared lab definition."""
    failures = []
    for lab_name, sources in REUSED_SYMBOLS.items():
        lab = importlib.import_module(lab_name)
        source_tree = ast.parse(Path(lab.__file__).read_text())
        local_definitions = {node.name for node in source_tree.body
                             if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        local_definitions |= {target.id for node in source_tree.body if isinstance(node, ast.Assign)
                              for target in node.targets if isinstance(target, ast.Name)}
        for runtime_name, names in sources.items():
            runtime = importlib.import_module(runtime_name)
            for name in names:
                if name in local_definitions:
                    failures.append(f"{lab_name}.{name}: duplicate implementation; import {runtime_name}.{name}")
                elif not hasattr(runtime, name) or not hasattr(lab, name):
                    failures.append(f"{lab_name}.{name}: missing runtime compatibility symbol")
                elif getattr(lab, name) is not getattr(runtime, name):
                    failures.append(f"{lab_name}.{name}: does not reuse {runtime_name}.{name}")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="read-only check (also the default)")
    parser.parse_args()
    failures = differences()
    if failures:
        raise SystemExit("\n".join(failures))
    count = sum(len(names) for sources in REUSED_SYMBOLS.values() for names in sources.values())
    print(f"tutorial-owned runtime: {count} shared symbols, no duplicated lab implementations")


if __name__ == "__main__":
    main()
