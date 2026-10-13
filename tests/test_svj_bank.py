"""Bank identity, native model replay, strict storage validation and CLI checks."""

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal

from sbi4finance.svj import bank, model

ROOT = Path(__file__).resolve().parents[1]


def config():
    return {"task": bank.default_task(), "root_seed": 43021,
            "splits": {"train": 7, "validation": 3, "test": 2}}


def legacy_simulate(n, seed, n_days):
    """Original contributed draw/transition order, frozen before extraction."""
    rng = np.random.default_rng(seed)
    coordinates = model.draw_coordinates(n, rng)
    params = model.model_params(coordinates)
    v = model.initial_variance(coordinates, rng)
    mu, theta, kappa, sigma_v, rho, lam, mu_j, sigma_j = (params[:, i] for i in range(8))
    eps1 = rng.standard_normal((n_days, n))
    eps2 = rng.standard_normal((n_days, n))
    jumps = rng.random((n_days, n)) < lam
    sizes = mu_j + sigma_j * rng.standard_normal((n_days, n))
    returns = np.empty((n_days, n))
    variance = np.empty((n_days, n))
    for t in range(n_days):
        sd = np.sqrt(v)
        returns[t] = mu + sd * eps1[t] + jumps[t] * sizes[t]
        v = np.maximum(v + kappa * (theta - v) + sigma_v * sd
                       * (rho * eps1[t] + np.sqrt(1 - rho**2) * eps2[t]), model.V_MIN)
        variance[t] = v
    targets = np.concatenate([coordinates, model.log_spot_vol(v)[:, None]], axis=1)
    return {"returns": returns.T, "variance": variance.T, "jumps": jumps.T, "targets": targets}


class SVJBankTests(unittest.TestCase):
    def test_historical_seed_and_rng_call_sequence_are_unchanged(self):
        for n, n_days in ((1, 13), (7, model.WINDOW)):
            actual = model.simulate(n, seed=83127, n_days=n_days)
            expected = legacy_simulate(n, 83127, n_days)
            for key in expected:
                assert_array_equal(actual[key], expected[key])

    def test_supplied_inputs_match_scalar_daily_equations_and_lean_output(self):
        coords = model.draw_coordinates(3, np.random.default_rng(51))
        params = model.model_params(coords)
        initial = model.initial_variance(coords, np.random.default_rng(19))
        rng = np.random.default_rng(9)
        e1, e2, z = [rng.standard_normal((5, 3)) for _ in range(3)]
        u = rng.random((5, 3))
        lean = model.daily_model_from_innovations(params, initial, e1, e2, u, z)
        full = model.daily_model_from_innovations(params, initial, e1, e2, u, z, retain_auxiliary=True)
        self.assertEqual(set(lean), {"returns", "terminal_state"})
        assert_array_equal(lean["returns"], full["returns"])
        assert_array_equal(lean["terminal_state"], full["variance"][:, -1:])
        expected_returns, expected_state = np.empty((3, 5)), initial.copy()
        for row, (mu, theta, kappa, sigma_v, rho, lam, mu_j, sigma_j) in enumerate(params):
            for day in range(5):
                v = expected_state[row]
                expected_returns[row, day] = mu + np.sqrt(v) * e1[day, row] + (u[day, row] < lam) * (mu_j + sigma_j * z[day, row])
                expected_state[row] = max(v + kappa * (theta - v) + sigma_v * np.sqrt(v)
                                          * (rho * e1[day, row] + np.sqrt(1 - rho**2) * e2[day, row]), model.V_MIN)
        assert_array_equal(lean["returns"], expected_returns)
        assert_array_equal(lean["terminal_state"][:, 0], expected_state)
        assert_array_equal(initial, model.initial_variance(coords, np.random.default_rng(19)))
        with self.assertRaisesRegex(ValueError, "uniforms"):
            model.daily_model_from_innovations(params, initial, e1, e2, np.ones_like(u), z)
        with self.assertRaisesRegex(ValueError, "finite"):
            model.daily_model_from_innovations(params, np.full(3, -model.V_MIN), e1, e2, u, z)

    def test_legacy_forward_replay_keeps_positive_inferred_start_below_floor(self):
        coordinates = model.draw_coordinates(3, np.random.default_rng(11))
        log_states = np.array([
            model.log_spot_vol(model.V_MIN),
            np.float32(model.log_spot_vol(model.V_MIN)),
            model.log_spot_vol(model.V_MIN / 100),
        ])
        targets = np.column_stack([coordinates, log_states])
        actual = model.forward_returns(targets, horizon=5, n_paths=2, rng=np.random.default_rng(93))
        params = np.repeat(model.model_params(targets), 2, axis=0)
        state = np.repeat(model.variance_from_log_spot_vol(log_states), 2)
        self.assertTrue(np.all(state < model.V_MIN))
        rng = np.random.default_rng(93)
        e1, e2 = rng.standard_normal((5, 6)), rng.standard_normal((5, 6))
        jumps = rng.random((5, 6)) < params[:, 5]
        sizes = params[:, 6] + params[:, 7] * rng.standard_normal((5, 6))
        expected = np.empty((5, 6))
        mu, theta, kappa, sigma_v, rho = (params[:, j] for j in range(5))
        for day in range(5):
            sd = np.sqrt(state)
            expected[day] = mu + sd * e1[day] + jumps[day] * sizes[day]
            state = np.maximum(state + kappa * (theta - state) + sigma_v * sd
                               * (rho * e1[day] + np.sqrt(1 - rho**2) * e2[day]), model.V_MIN)
        assert_array_equal(actual, expected.T.reshape(3, 2, 5))

    def test_example_identity_survives_reorder_chunk_and_split_count_changes(self):
        task = config()
        original = bank.generate_examples(task, "train", range(7))
        requested = [6, 0, 3, 1]
        reordered = bank.generate_examples(task, "train", requested)
        with patch.object(bank, "GENERATION_BATCH_SIZE", 2):
            rechunked = bank.generate_examples(task, "train", range(7))
        enlarged = deepcopy(task)
        enlarged["splits"]["train"] = 1024
        resized = bank.generate_examples(enlarged, "train", range(7))
        for key, values in original.items():
            assert_array_equal(reordered[key], values if key == "dt_years" else values[requested])
            assert_array_equal(rechunked[key], values)
            assert_array_equal(resized[key], values)
        for split in ("validation", "test"):
            other = bank.generate_examples(task, split, range(7))
            for key in ("parameters", "observations", "terminal_state", "example_id"):
                self.assertFalse(np.array_equal(other[key], original[key]))
        # Full uint64 IDs use two words; high bits are not lost in seeding.
        high = bank.generate_examples(task, "train", [0, 2**32, 2**64 - 1])
        self.assertFalse(np.array_equal(high["parameters"][0], high["parameters"][1]))

    def test_prior_and_initial_diffusion_jump_streams_are_independent(self):
        streams = bank._rngs(config(), "train", 4)
        samples = [rng.random(8) for rng in streams.values()]
        for a in range(len(samples)):
            for b in range(a):
                self.assertFalse(np.array_equal(samples[a], samples[b]))
        perturbed = bank._rngs(config(), "train", 4)
        perturbed["prior"].normal(size=99)
        for name in ("initialization", "diffusion", "jump_indicator", "jump_size"):
            assert_array_equal(perturbed[name].random(8), samples[list(streams).index(name)])

    def test_full_task_identity_and_bounds_are_strict(self):
        original = config()
        bank.validate_config(original)
        altered = deepcopy(original)
        altered["task"]["prior"]["drift_loc"] += 1
        bank.validate_config(altered)
        self.assertNotEqual(bank.task_fingerprint(original), bank.task_fingerprint(altered))
        self.assertFalse(np.array_equal(bank.generate_examples(original, "train", [0])["parameters"],
                                       bank.generate_examples(altered, "train", [0])["parameters"]))
        invalid = []
        for key, value in (("measure", "Q"), ("n_observations", 999), ("dt_years", 1 / 365),
                           ("substeps", 2), ("observation_unit", "percent_log_return")):
            copy = deepcopy(original)
            copy["task"][key] = value
            invalid.append(copy)
        for location, key, value in (("prior", "rho_a", 0), ("prior", "drift_sd", np.nan),
                                     ("constants", "variance_floor", 0),
                                     ("features", "squared_return_offset", 1),
                                     ("reference_sampler", "steps", 100)):
            copy = deepcopy(original)
            copy["task"][location][key] = value
            invalid.append(copy)
        for value in (-1, 2**64, True, 1.2):
            invalid.append({**original, "root_seed": value})
        for bad in invalid:
            with self.subTest(config=bad), self.assertRaises(ValueError):
                bank.validate_config(bad)
        for ids in ([], [1, 1], [-1], [2**64], [True], [0.2]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                bank.generate_examples(original, "train", ids)
        with self.assertRaises(ValueError):
            bank.generate_examples(original, "other", [0])

    def test_bank_roundtrip_hashes_roles_units_ids_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bank"
            written = bank.write_bank(path, config())
            manifest, splits = bank.load_bank(path)
            self.assertEqual(written, manifest)
            self.assertEqual(manifest["rng"]["version"], bank.RNG_VERSION)
            self.assertFalse(manifest["future_outcomes_in_conditioning"])
            identities = set()
            for split, arrays in splits.items():
                expected = bank.generate_examples(config(), split, range(len(arrays["example_id"])))
                for key in expected:
                    assert_array_equal(arrays[key], expected[key])
                self.assertFalse(identities.intersection(arrays["example_id"]))
                identities.update(arrays["example_id"])
                self.assertTrue(np.all(arrays["terminal_state"] >= model.V_MIN))
            with self.assertRaises(FileExistsError):
                bank.write_bank(path, config())
            data = bytearray((path / "train.npz").read_bytes())
            data[-1] ^= 1
            (path / "train.npz").write_bytes(data)
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                bank.load_bank(path)

    def test_valid_hash_does_not_excuse_invalid_arrays_or_false_manifest_roles(self):
        changes = {
            "bad_state": lambda a: a["terminal_state"].fill(0),
            "bad_parameters": lambda a: a["parameters"].fill(np.inf),
            "overflow_parameters": lambda a: a["parameters"].fill(1e4),
            "bad_grid": lambda a: a["dt_years"].fill(1 / 365),
            "bad_observation": lambda a: a["observations"].fill(np.nan),
            "bad_ids": lambda a: a.update(example_id=np.array(["test:0"] * len(a["example_id"]))),
            "extra_truth": lambda a: a.update(future_returns=np.zeros((len(a["example_id"]), 1))),
            "bad_dtype": lambda a: a.update(observations=a["observations"].astype(np.float32)),
        }
        for name, mutate in changes.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "bank"
                manifest = bank.write_bank(path, config())
                source = path / "train.npz"
                with np.load(source, allow_pickle=False) as saved:
                    arrays = dict(saved)
                mutate(arrays)
                np.savez(source, **arrays)
                manifest["splits"]["train"]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
                (path / "manifest.json").write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    bank.load_bank(path)
        for key, value in (("future_outcomes_in_conditioning", True), ("roles", {}),
                           ("task_fingerprint", "wrong"), ("rng", {})):
            with self.subTest(field=key), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "bank"
                manifest = bank.write_bank(path, config())
                manifest[key] = value
                (path / "manifest.json").write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, "manifest field"):
                    bank.load_bank(path)

    def test_failure_is_manifest_last_and_names_example_without_redraw(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bank"
            with patch.object(model, "initial_variance", side_effect=ValueError("bad Gamma draw")):
                with self.assertRaisesRegex(ValueError, "failed example train:0"):
                    bank.write_bank(path, config())
            self.assertFalse((path / "manifest.json").exists())
            failure = json.loads((path / "failure.json").read_text())
            self.assertIn("train:0", failure["error"])
            with self.assertRaisesRegex(ValueError, "incomplete bank"):
                bank.load_bank(path)

    def test_cli_checker_exact_portable_source_and_environment_contract(self):
        spec = importlib.util.spec_from_file_location("check_svj_bank", ROOT / "scripts/check_svj_bank.py")
        checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checker)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bank"
            manifest = bank.write_bank(path, config())
            self.assertTrue(checker.check_bank(path)["passed"])
            portable = checker.check_bank(path, portable=True)
            self.assertEqual(portable["tolerance"], bank.PORTABLE_TOLERANCE)
            manifest["versions"]["python"] = "different"
            (path / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "environment versions"):
                checker.check_bank(path)
            self.assertTrue(checker.check_bank(path, portable=True)["passed"])
            manifest["source_sha256"]["svj/model.py"] = "0" * 64
            (path / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "generator source"):
                checker.check_bank(path, portable=True)


if __name__ == "__main__":
    unittest.main()
