"""The thin-chain posterior (S2-T): layout, curve prior density, training data, the audit, products.

The training test needs BayesFlow (the pixi environment); it is skipped without it.
"""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from sbi4finance.calibration import bank, tasks, thin, thin_products

ROOT = Path(__file__).resolve().parents[1]
PRIOR = {"kind": "power_law_walk", "level_bounds": [0.15, 0.30], "exponent_bounds": [-0.5, 0.5], "walk_sd": 0.15,
         "vol_bounds": [0.10, 0.45]}


def small_task():
    return tasks.rough_task(days=(5, 10, 21), deltas=(-1.0, 0.0, 1.0), blocks=[{"horizon": 21 / 252, "n_steps": 42}],
                            paths_ladder=(256, 512), budget=10.0, last_rung_cap=10.0, curve_prior=PRIOR)


def small_layout(task, config):
    idx = [3, 4, 5, 6, 7, 8]  # the 10- and 21-day contracts
    sigma = tasks.sigma(task)
    return {"format": thin.LAYOUT_FORMAT, "ticker": "TEST", "asof": "2026-10-22", "source": "unit test",
            "contract_indices": idx, "noise_sd": [float(3 * sigma[i]) for i in idx],
            "half_width_vol": [0.01] * len(idx), "maturity_days": [10, 10, 10, 21, 21, 21],
            "delta": [-1.0, 0.0, 1.0] * 2, "expiry_map": {}, "forward": "carry",
            "task_fingerprint": bank.task_fingerprint(config)}


class ThinTests(unittest.TestCase):
    def test_layouts_validate_and_the_full_bank_layout_is_consistent(self):
        task = small_task()
        config = {"task": task, "root_seed": 1, "splits": {"train": 2, "validation": 1, "test": 1}}
        layout = small_layout(task, config)
        thin.validate_layout(layout, task)
        bad = dict(layout, noise_sd=[x / 10 for x in layout["noise_sd"]])
        with self.assertRaisesRegex(ValueError, "below the design noise"):
            thin.validate_layout(bad, task)
        full = json.loads((ROOT / "configs" / "tasks" / "calibration_rough_bank.json").read_text())
        ewg = thin.load_layout(ROOT / "configs" / "thin" / "ewg_layout.json", full["task"])
        self.assertEqual(ewg["task_fingerprint"], bank.task_fingerprint(full))

    def test_curve_prior_density_matches_the_sampler(self):
        task = small_task()
        rng = np.random.default_rng(0)
        draws = np.log(np.array([tasks.draw_context(task, rng)[0] for _ in range(20000)]))
        # Importance weights of prior draws under the prior density, against a broad Gaussian reference:
        # E_prior[q(x) / p(x)] = 1 for any normalized q; here we check that the density is finite on every draw
        # and that its normalization is constant across two independent halves.
        log_p = thin.curve_log_prior(draws, task)
        self.assertTrue(np.all(np.isfinite(log_p)))
        mean, cov = draws.mean(axis=0), np.cov(draws.T)
        from scipy.stats import multivariate_normal
        log_q = multivariate_normal(mean, cov).logpdf(draws)
        ratio = np.exp(log_q - log_p)
        a, b = ratio[:10000].mean(), ratio[10000:].mean()
        self.assertAlmostEqual(a / b, 1.0, delta=0.05)
        # truncation: outside vol_bounds the density is zero
        outside = draws[:1].copy()
        outside[0, -1] = 2 * np.log(0.5)
        self.assertEqual(thin.curve_log_prior(outside, task)[0], -np.inf)

    def test_training_data_noise_and_conditions(self):
        task = small_task()
        config = {"task": task, "root_seed": 1, "splits": {"train": 4, "validation": 2, "test": 1}}
        layout = small_layout(task, config)
        with tempfile.TemporaryDirectory() as directory:
            bank.write_bank(Path(directory) / "bank", config)
            _, splits = bank.load_bank(Path(directory) / "bank")
        data = thin.thin_clean_data(splits["train"], task, layout)
        self.assertEqual(data["clean_scaled"].shape, (4, 6))
        self.assertEqual(sorted(k for k in data if k != "clean_scaled"), sorted(thin.target_names(task)))
        np.testing.assert_allclose(data["log_xi_2"][:, 0], np.log(splits["train"]["context_values"][:, 1]))
        augment = thin.thin_noise(0)
        batch = augment(dict(data))
        self.assertNotIn("clean_scaled", batch)
        s = np.exp(batch["log_scale"])
        self.assertTrue(np.all((s >= thin.SCALE_BOUNDS[0]) & (s <= thin.SCALE_BOUNDS[1])))
        residual = batch["observations"] - data["clean_scaled"] / s
        self.assertLess(abs(residual.mean()), 1.0)
        idx, noise = thin.layout_arrays(layout)
        quotes = splits["train"]["reference_prices"][:, idx]
        cond = thin.thin_conditions(quotes, layout, 2.0)
        np.testing.assert_allclose(cond["observations"], quotes / (2.0 * noise))
        np.testing.assert_allclose(cond["log_scale"], np.log(2.0))

    def test_audit_recovers_a_known_posterior(self):
        """Proposal and weights: a Gaussian likelihood on the first parameter, prior elsewhere."""
        task = small_task()
        rng = np.random.default_rng(5)
        draws = thin.draw_prior(task, rng, 400)
        draws[:, 0] = np.clip(0.15 + 0.02 * rng.standard_normal(400), 0.03, 0.34)
        targets, log_q, _ = thin.propose(draws, task, 4000, seed=1)
        log_lik = -0.5 * ((targets[:, 0] - 0.15) / 0.02) ** 2
        result = thin.correct(log_lik, log_q, targets, task)
        mean = np.sum(result["weights"] * targets[:, 0])
        self.assertAlmostEqual(mean, 0.15, delta=0.004)
        self.assertLess(result["khat"], 0.7)

    def test_power_law_extension_and_curve_map(self):
        task = json.loads((ROOT / "configs" / "tasks" / "calibration_rough_bank.json").read_text())["task"]
        days = [21, 40, 58, 120]
        np.testing.assert_array_equal(thin.classical_curve_map(task, days), [0, 0, 0, 1, 2, 3, 3])
        mid = tasks.interval_midpoints(task)
        edges = np.array([0] + days) / 252
        quoted_mid = 0.5 * (edges[:-1] + edges[1:])
        levels = 2 * np.log(0.2) + 0.3 * np.log(quoted_mid / quoted_mid[0])  # an exact power law
        row = np.concatenate([[0.1, 1.5, -0.7], levels[thin.classical_curve_map(task, days)]])
        out = thin.power_law_extension(task, row, days)
        np.testing.assert_allclose(out[:-1], row[:-1])
        self.assertAlmostEqual(out[-1], 2 * np.log(0.2) + 0.3 * np.log(mid[-1] / quoted_mid[0]), places=10)

    def test_thin_products_variance_swap_is_the_curve_average(self):
        task = json.loads((ROOT / "configs" / "tasks" / "calibration_rough_bank.json").read_text())["task"]
        levels = np.full(7, 0.04)
        prices, se = thin_products.price(task, [0.1, 1.5, -0.7], np.log(levels), 2048, 0)
        self.assertAlmostEqual(prices[0], 0.04, places=12)
        self.assertTrue(np.all(se[1:] > 0))

    @unittest.skipUnless(subprocess.run([sys.executable, "-c", "import bayesflow"], capture_output=True).returncode == 0,
                         "BayesFlow not installed")
    def test_training_runs_and_reloads(self):
        task = small_task()
        config = {"task": task, "root_seed": 1, "splits": {"train": 16, "validation": 4, "test": 1}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bank.write_bank(root / "bank", config)
            layout_path = root / "layout.json"
            layout_path.write_text(json.dumps(small_layout(task, config)))
            run = subprocess.run([sys.executable, str(ROOT / "tutorial_final" / "studies" / "train_calibration.py"),
                                  "--bank", str(root / "bank"), "--out", str(root / "run"), "--epochs", "2",
                                  "--batch-size", "8", "--thin-layout", str(layout_path), "--widths", "16,16"],
                                 capture_output=True, text=True, env={**__import__("os").environ,
                                                                       "KERAS_BACKEND": "jax"})
            self.assertEqual(run.returncode, 0, run.stderr[-2000:])
            report = json.loads((root / "run" / "training.json").read_text())
            self.assertTrue(report["same_seed_reload_exact"])
            self.assertEqual(report["training"]["thin_layout"]["contracts"], 6)


if __name__ == "__main__":
    unittest.main()
