"""Compact Bergomi results bind a plotting-only lesson to one immutable task."""

from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


LAB = Path(__file__).resolve().parents[1]
DATA = LAB / "evidence/pilot-results/2026-11-14/bergomi-capstone"
HELPER = LAB / "notebooks/_bergomi_capstone_results.py"
HOST_DESIGN = LAB / "evidence/pilot-results/2026-11-09/heston-spread-baseline/revised-grid/cv-preparation/design.json"
SPEC = importlib.util.spec_from_file_location("compact_bergomi_capstone", HELPER)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BergomiCapstoneResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = MODULE.load_results(DATA)
        cls.design = json.loads(HOST_DESIGN.read_text())

    def assert_rejected(self, mutate):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = deepcopy(self.record)
            mutate(record)
            (root / "results.json").write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                MODULE.load_results(root)

    def test_shipped_compact_data_binds_to_host_quotes_without_mutation(self):
        before = sha256((DATA / "results.json").read_bytes()).hexdigest()
        quotes = self.design["selected_quotes"]
        record = MODULE.load_results(DATA, expected_contracts=quotes,
                                     expected_spreads=[quote["half_spread"] for quote in quotes],
                                     expected_observations=[quote["mid"] for quote in quotes])
        self.assertEqual(record["parameter_names"], list(MODULE.PARAMETER_NAMES))
        self.assertEqual(record["training"]["rows"], 200000)
        self.assertFalse(record["scientific_acceptance"])
        self.assertFalse(record["numerical_qualification_passed"])
        self.assertEqual(before, sha256((DATA / "results.json").read_bytes()).hexdigest())
        self.assertEqual(MODULE.load_results(DATA, expected_design=self.design), record)

    def test_changed_or_reordered_contracts_are_rejected_even_with_updated_hash(self):
        def stale_hash(record):
            record["geometry"]["observed_raw"][0] += .1
        self.assert_rejected(stale_hash)
        reversed_quotes = self.design["selected_quotes"][::-1]
        with self.assertRaises(ValueError):
            MODULE.load_results(DATA, expected_contracts=reversed_quotes)
        wrong_spreads = np.array(self.record["geometry"]["half_spread_raw"])
        wrong_spreads[0] *= 2
        with self.assertRaises(ValueError):
            MODULE.load_results(DATA, expected_spreads=wrong_spreads)
        wrong_design = deepcopy(self.record["geometry"])
        wrong_design["discount"][0] *= .99
        with self.assertRaises(ValueError):
            MODULE.load_results(DATA, expected_design=wrong_design)

    def test_financial_units_and_prior_identity_cannot_be_relabelled(self):
        mutations = (
            lambda r: r["observation_model"].update(h_units="Black volatility"),
            lambda r: r["observation_model"].update(c_bounds=[.5, 2.]),
            lambda r: r["observation_model"].update(independent=False),
            lambda r: r.update(host_heston_c_bounds=[.5, 50.]),
            lambda r: r["parameter_bounds"][1].__setitem__(1, 3.),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.assert_rejected(mutation)

    def test_seven_target_frozen_task_and_primary_training_budget_are_required(self):
        mutations = (
            lambda r: r["task"].update(id="older_supplied_curve_tutorial"),
            lambda r: r["task"].update(conditioning="whitened_Y_over_h_and_true_c"),
            lambda r: r["parameter_names"].reverse(),
            lambda r: r["training"].update(rows=300000),
            lambda r: r["training"].update(checkpoint_epoch=258),
            lambda r: r["training"].update(checkpoint_role="validation_minimum"),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.assert_rejected(mutation)

    def test_plot_summaries_require_finite_ordered_quantiles_and_normalized_density(self):
        def inverted(record):
            quantiles = record["posteriors"]["amortizer"]["q05_q50_q95"]
            quantiles[0][6] = quantiles[2][6] + 1
        def density(record):
            record["posteriors"]["mcmc"]["histograms"]["c"]["density"][0] += 1
        def nonfinite(record):
            record["predictives"]["mcmc"]["clean_mean_raw"][0] = float("nan")
        def wrong_units(record):
            record["fit_comparison"]["bergomi"]["price_mean_raw"][0] /= 1000
        for mutation in (inverted, density, nonfinite, wrong_units):
            with self.subTest(mutation=mutation.__name__):
                self.assert_rejected(mutation)

    def test_original_paired_draws_and_descriptive_predictive_counts_are_preserved(self):
        mutations = (
            lambda r: r["predictives"]["amortizer"].update(conditional_c_redraw=True),
            lambda r: r["predictives"]["mcmc"].update(paired_parameter_and_c_draws=False),
            lambda r: r["predictives"]["amortizer"].update(support_exclusions=-1),
            lambda r: r["predictives"]["mcmc"].update(draw_count=4096),
            lambda r: r["predictives"]["mcmc"].update(noisy_in_band_count=-1),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.assert_rejected(mutation)

    def test_approximate_reference_requires_measured_diagnostics_and_same_mean_function(self):
        mutations = (
            lambda r: r["reference"].update(exact_bergomi=True),
            lambda r: r["reference"].update(mean_function_sha256="0" * 64),
            lambda r: r["reference"].update(max_r_hat=1.02),
            lambda r: r["reference"].update(min_ess_bulk=399),
            lambda r: r["reference"].update(divergences=1),
            lambda r: r["reference"].update(balanced_draws_per_chain=1024),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.assert_rejected(mutation)

    def test_source_hashes_duplicate_keys_and_private_payloads_are_rejected(self):
        source = next(iter(self.record["source_bindings"]))
        mutations = (
            lambda r: r["source_bindings"].update({source: "g" * 64}),
            lambda r: r.update(private_path="/Users/example/private/chain.npz"),
            lambda r: r.update(theta=[[1, 2, 3]]),
            lambda r: r.update(source_file="../outside/results.json"),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.assert_rejected(mutation)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = json.dumps(self.record)
            (root / "results.json").write_text('{"format":"duplicate",' + payload[1:])
            with self.assertRaisesRegex(ValueError, "duplicate result key"):
                MODULE.load_results(root)


if __name__ == "__main__":
    unittest.main()
