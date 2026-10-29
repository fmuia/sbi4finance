"""Read only two fixed validation cases and their already-saved posterior draws."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--bank', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    a = p.parse_args(); sys.path.insert(0, str(a.repo/'src'))
    from sbi4finance.calibration import bank_collection, tasks
    def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    manifest = json.loads((a.bank/'manifest.json').read_text()); task = manifest['configuration']['task']
    rows = next(bank_collection.iter_snapshot(a.bank, 'validation', roles=(
        'parameters', 'context_values', 'observations', 'reference_prices', 'example_id')))
    with np.load(a.run/'validation/draws.npz', allow_pickle=False) as saved:
        draws = saved['draws'][[0,2]]; truth = saved['truth'][[0,2]]
    if not np.array_equal(truth, rows['parameters'][[0,2]]):
        raise ValueError('Saved draws do not match the fixed bank validation identities')
    points = []
    for j, case in enumerate((0,2)):
        if rows['example_id'][case] != f'validation:{case}': raise ValueError('Validation IDs changed')
        for role, theta in [('truth', truth[j]), ('neural_mean', draws[j].mean(0))]:
            curve = (tasks.derived_context(task, theta)[0] if tasks.is_joint_curve_task(task)
                     else rows['context_values'][case])
            points.append({'case': case, 'role': role, 'theta': theta.tolist(),
                'curve_levels': curve.tolist(), 'observed_prices': rows['observations'][case].reshape(-1).tolist(),
                'stored_bank_prices': rows['reference_prices'][case].tolist()})
    training = json.loads((a.run/'posterior/training.json').read_text())
    paths = [a.bank/'manifest.json', a.run/'validation/draws.npz', a.run/'posterior/training.json']
    print(json.dumps({'format': 'rough_current_validation_representatives_v1', 'task': task, 'points': points,
        'selection': 'Predeclared validation:0 and validation:2; truth and mean of all1000 already-saved final500 draws; no case search.',
        'input_sha256': {str(path): sha(path) for path in paths},
        'checkpoint_sha256': training['checkpoint_sha256'], 'reserved_test_evaluated': False}, indent=2))


if __name__ == '__main__': main()
