"""A training seed must not depend on whether BayesFlow was already imported."""

from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
try:
    CORE_ENVIRONMENT = metadata.version("bayesflow") == "2.1.0"
except metadata.PackageNotFoundError:
    CORE_ENVIRONMENT = False

PROBE = textwrap.dedent("""
    from argparse import Namespace
    import hashlib, json, os, tempfile
    from pathlib import Path
    from unittest.mock import patch
    import numpy as np
    from scripts import train_full_surface as trainer
    from sbi4finance.calibration import bank, full_surface, tasks
    if os.environ['AUDIT_PREIMPORT_BAYESFLOW'] == '1':
        import bayesflow
    task = tasks.heston_task()
    config = {'task': task, 'root_seed': 12,
              'splits': {'train': 16, 'validation': 8, 'test': 4}}
    manifest = {'configuration': config, 'task_fingerprint': bank.task_fingerprint(config)}
    bounds = np.asarray(task['prior_bounds'])
    def split(n):
        parameters = np.broadcast_to(bounds.mean(1), (n, 4)).copy()
        return {'parameters': parameters, 'reference_prices': np.full((n, 63), .02),
                'observations': np.full((n, 63, 1), .02),
                'context_values': np.empty((n, 0))}
    original = full_surface.build_workflow
    class BuildOnlyComplete(Exception):
        pass
    def capture(*args, **kwargs):
        workflow = original(*args, **kwargs)
        network = workflow.inference_network
        network.build((1, 4), (1, 63))
        hashes = [hashlib.sha256(np.asarray(w).tobytes()).hexdigest()
                  for w in network.trainable_variables]
        print('INITIAL_WEIGHTS=' + json.dumps(hashes))
        raise BuildOnlyComplete
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root/'bank').mkdir()
        (root/'bank/manifest.json').write_text(json.dumps(manifest))
        args = Namespace(bank=root/'bank', out=root/'out', family='flow_matching',
                         encoder='vector', widths='256,256,256,256', summary_dim=32,
                         flow_steps=None, mean_steps=4, epochs=1, batch_size=8,
                         chunk_epochs=1, seed=0)
        with patch.object(trainer.bank, 'load_bank', return_value=(manifest,
                {'train': split(16), 'validation': split(8)})), \
             patch.object(full_surface, 'build_workflow', side_effect=capture):
            try:
                trainer.run_training(args)
            except BuildOnlyComplete:
                pass
            else:
                raise AssertionError('Probe must stop before training')
""")


@unittest.skipUnless(CORE_ENVIRONMENT, "native training requires the BayesFlow 2.1 core environment")
class TrainingSeedOrderTests(unittest.TestCase):
    def test_cold_and_preimported_bayesflow_give_identical_initial_weights(self):
        hashes = []
        for preimport in ("0", "1"):
            environment = {**os.environ, "AUDIT_PREIMPORT_BAYESFLOW": preimport,
                           "KERAS_BACKEND": "jax", "JAX_PLATFORMS": "cpu",
                           "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT)]),
                           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                           "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
            result = subprocess.run([sys.executable, "-c", PROBE], cwd=ROOT,
                                    env=environment, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            line = next(line for line in result.stdout.splitlines()
                        if line.startswith("INITIAL_WEIGHTS="))
            hashes.append(json.loads(line.split("=", 1)[1]))
        self.assertEqual(len(hashes[0]), 41)
        self.assertEqual(hashes[0], hashes[1])


if __name__ == "__main__":
    unittest.main()
