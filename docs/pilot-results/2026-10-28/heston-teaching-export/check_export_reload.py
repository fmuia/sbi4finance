import hashlib
from importlib import metadata
import json
from pathlib import Path
import sys
import time

import numpy as np

root = Path(sys.argv[1])
checkpoint = Path('/private/tmp/sbi4finance-heston-verification-path.txt').read_text().strip()
checkpoint = Path(checkpoint) / 'checkpoint'
sys.path.insert(0, str(root / 'src'))
from sbi4finance.calibration import full_surface, full_surface_teaching as teaching

started = time.perf_counter()
before = {p.name: teaching.digest(p) for p in checkpoint.iterdir() if p.is_file()}
publication = json.loads((checkpoint / 'export.json').read_text())
for name, sha in publication['files'].items():
    assert teaching.digest(checkpoint / name) == sha
assert teaching.digest(checkpoint / 'model.keras') == '8d4a243ba099524cf4cda40e4f0616e895879f053829a7f60dadbe65b98301db'
task, restored, record = full_surface.load_checkpoint(checkpoint)
assert record['exported_candidate_epoch'] == 500 and record['original_validation_selected_epoch'] == 402
assert not record['scientific_validation'] and not record['same_seed_reload_exact']
_, cache, values = teaching.load_cache(root / 'data/synthetic/full-surface-heston-final500/validation-0',
    expected_task=task, expected_checkpoint_sha256=record['checkpoint_sha256'])
direct = full_surface.build_workflow(task, **record['network'])
direct.load_estimator(checkpoint / 'model.keras')
left = full_surface.sample(restored, task, values['observations'][None, :], 10, seed=280500)
right = full_surface.sample(direct, task, values['observations'][None, :], 10, seed=280500)
assert left.shape == right.shape == (1, 10, 4) and np.array_equal(left, right)
assert before == {p.name: teaching.digest(p) for p in checkpoint.iterdir() if p.is_file()}
report = {'format': 'heston_export_reload_verification_v1', 'status': 'passed',
    'candidate_sha256': record['checkpoint_sha256'], 'exported_candidate_epoch': 500,
    'original_validation_selected_epoch': 402, 'normal_core_loader_succeeded': True,
    'seeded_direct_vs_export_reload_exact': True, 'draw_shape': list(left.shape), 'seed': 280500,
    'reference': 'Direct build+candidate load versus normal full_surface.load_checkpoint(export)',
    'draw_bytes_sha256': hashlib.sha256(left.tobytes()).hexdigest(),
    'model_and_metadata_unchanged': True, 'offline_cache_replayed': cache['case_id'],
    'source_sha256': {str(p.relative_to(root)): teaching.digest(p) for p in [
        root/'src/sbi4finance/calibration/full_surface_teaching.py', root/'scripts/prepare_full_surface_teaching.py',
        root/'tests/test_full_surface_teaching.py']},
    'runtime': {name: metadata.version(name) for name in ('bayesflow','keras','jax','numpy')},
    'seconds': time.perf_counter()-started,
    'scope': 'Read-only local CPU software/reload check on existing development observation; no training, pricing, MCMC or notebook execution.'}
out = root / 'docs/pilot-results/2026-10-28/heston-teaching-export'
out.mkdir(parents=True, exist_ok=True)
(out / 'verification.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report))
