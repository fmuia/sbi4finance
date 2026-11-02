"""Forward-emulator reporting and protected holdout use."""

from argparse import Namespace
import json
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("KERAS_BACKEND", "jax")
import numpy as np

from scripts import train_full_surface_pricer as cli
from sbi4finance.calibration import bank, pricer_net, tasks


class FullSurfacePricerCliTests(unittest.TestCase):
    def test_reports_surface_accumulation_in_addition_to_quote_errors(self):
        report = cli.error_summary(np.full((3, 63), .1))
        self.assertAlmostEqual(report["rmse"], .1)
        self.assertAlmostEqual(report["surface_l2_error_quantiles"]["0.5"], np.sqrt(63) * .1)
        self.assertAlmostEqual(report["surface_rms_error_quantiles"]["1.0"], .1)
        with self.assertRaisesRegex(ValueError, "finite"):
            cli.error_summary([[np.nan]])

    def test_actual_small_fit_binds_selected_pricer_without_opening_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {"task": tasks.heston_task(), "root_seed": 716,
                      "splits": {"train": 16, "validation": 4, "test": 2}}
            bank.write_bank(root / "bank", config)
            # A missing reserved payload makes any accidental test read fail.
            (root / "bank/test.npz").unlink()
            args = Namespace(bank=root / "bank", out=root / "fit", widths="8,8",
                             epochs=2, batch_size=8, patience=2, learning_rate=.001, seed=0)
            report = cli.run_training(args)
            checkpoint = args.out / "checkpoint"
            saved = json.loads((checkpoint / "pricer.json").read_text())
            self.assertEqual(saved, report)
            self.assertEqual(report["status"], "completed")
            self.assertIn("tutorial/src/sbi4f_tutorial/shared/full_surface.py",
                          report["shared_runtime_source_sha256"])
            self.assertEqual(report["pricer_sha256"], cli.digest(checkpoint / "pricer.npz"))
            self.assertEqual(json.loads((checkpoint / "task.json").read_text()), config["task"])
            self.assertFalse(report["reserved_test_evaluated"])
            self.assertFalse(report["scientific_acceptance"])
            self.assertTrue(report["same_reload_exact"])
            self.assertEqual(report["validation_label_reported_mcse"]["max"], 0.)
            self.assertEqual(pricer_net.Pricer.load(checkpoint).activation, "gelu_erf")
            with self.assertRaises(FileExistsError):
                cli.run_training(args)


if __name__ == "__main__":
    unittest.main()
