"""Reuse retained historical declarations for the append-only seed audit; no prices."""
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path('/Users/afengler/Projects/proj_sbi4finance/sbi4finance-full-surface-core')
sys.path.insert(0, str(REPO / 'src'))
from sbi4finance.calibration import bank


def digest(x):
    return hashlib.sha256(bank._json(x).encode()).hexdigest()


def main(out):
    prior_path = Path('/private/tmp/rough-bank-seed-audit-20261027.json')
    prior = json.loads(prior_path.read_text())
    assert prior['passed'] and np.__version__ == '2.5.3'
    configurations, states, seeds, inputs = {}, set(), set(), []

    def add_config(x):
        if not isinstance(x.get('task'), dict) or 'root_seed' not in x or 'splits' not in x:
            return
        key = (int(x['root_seed']), digest(x['task']))
        if key not in configurations:
            configurations[key] = json.loads(json.dumps(x))
        else:
            for split, n in x['splits'].items():
                configurations[key]['splits'][split] = max(n, configurations[key]['splits'].get(split, 0))

    def scalars(x):
        if type(x) is int and x >= 0:
            seeds.add(x)
        elif isinstance(x, (list, tuple)):
            for v in x:
                scalars(v)
        elif isinstance(x, dict):
            for v in x.values():
                scalars(v)

    def walk(x):
        if isinstance(x, dict):
            add_config(x)
            for k, v in x.items():
                if k == 'pcg64_initial_state_sha256' and isinstance(v, str):
                    states.add(v)
                elif k == 'seed_sequence_entropy' and isinstance(v, list) and all(type(z) is int for z in v):
                    states.add(digest(np.random.PCG64(np.random.SeedSequence(v)).state))
                elif k == 'seed_sequence_state_uint64' and isinstance(v, list) and v:
                    seeds.add(int(v[0]))
                elif k in ('seed', 'seeds', 'pricing_seed', 'pricing_seeds', 'parameter_seed', 'streams'):
                    scalars(v)
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    for row in prior['input_files']:
        p = Path(row['path'])
        if not p.is_absolute():
            p = REPO / p
        if not p.exists() and Path(str(p) + '.gz').exists():
            p = Path(str(p) + '.gz')
        raw = gzip.decompress(p.read_bytes()) if p.suffix == '.gz' else p.read_bytes()
        x = json.loads(raw)
        inputs.append({'path': str(p), 'uncompressed_sha256': hashlib.sha256(raw).hexdigest()})
        if p.name == 'rough-bank-historical-manifests.json':
            for item in x:
                walk(item['record'])
        else:
            walk(x)
    for name in ('calibration_heston_backbone_100k.json', 'calibration_rough_joint_development_20k.json',
                 'calibration_rough_known_backbone_20k.json'):
        p = REPO / 'configs' / name
        walk(json.loads(p.read_text()))
        inputs.append({'path': str(p), 'uncompressed_sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
    pricing_plan = REPO / 'docs/pilot-results/2026-10-28/known-pricing-validation/known-rough-pricing-plan-20261028.json'
    p = json.loads(pricing_plan.read_text())
    for job in p['jobs']:
        for entropy in job['stream_entropy_by_block']:
            states.add(digest(np.random.PCG64(np.random.SeedSequence(entropy)).state))
    result = {'format': 'bank_delivery_retained_history_v1', 'configurations': list(configurations.values()),
              'explicit_seeds': sorted(seeds), 'explicit_pcg64_states': sorted(states), 'inputs': inputs,
              'numpy': np.__version__, 'prior_audit_sha256': hashlib.sha256(prior_path.read_bytes()).hexdigest(),
              'pricing_diagnostic_plan_sha256': hashlib.sha256(pricing_plan.read_bytes()).hexdigest(),
              'scope': 'Recorded configurations include planned ungenerated ranges; original raw course banks are unavailable.'}
    with Path(out).open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'configurations': len(configurations), 'explicit_seeds': len(seeds), 'explicit_states': len(states)}))


if __name__ == '__main__':
    main(sys.argv[1])
