from pathlib import Path
import hashlib,json,subprocess
root=Path('/oscar/scratch/afengler/sbi4finance-heston-audit-20261027')
repo=Path('/oscar/scratch/afengler/sbi4finance-rough-known-reference-20261027/repo')
checkpoints={'50k':'/oscar/scratch/afengler/sbi4finance-backbone-fit-20261027/runs/heston-50k-native/checkpoint','100k':'/oscar/scratch/afengler/sbi4finance-heston-100k-fit-20261027/runs/heston-100k-native/checkpoint'}
refs=Path('/oscar/scratch/afengler/sbi4finance-reviewed-unit-20261027-9b495cf/runs/heston')
archive=repo/'tutorial_final/checkpoints/heston';overlay=Path('/oscar/scratch/afengler/sbi4finance-backbone-fit-20261027/legacy-bayesflow')
files=[root/'audit_heston_saved_readouts_20261027.py',repo/'pyproject.toml',repo/'pixi.lock',*[repo/'src/sbi4finance/calibration'/n for n in ['full_surface.py','inference.py','validation.py']],*[archive/n for n in ['model.keras','task.json','training.json','ACCEPTED.json']]]
for cp in checkpoints.values():
 cp=Path(cp);r=json.loads((cp/'resume.json').read_text());files.extend([cp/n for n in ['model.keras','task.json','training.json','resume.json']]);files.append(cp/r['checkpoint_files']['last'])
for i in range(3):files.extend(refs/f'case-{i}'/'reference'/n for n in ['reference.json','diagnostics.json','chains.npz'])
files.extend(overlay.rglob('*.py'))
d={'root':str(root),'repo':str(repo),'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'overlay':str(overlay),'archive':str(archive),'references':str(refs),'new_checkpoints':checkpoints,'frozen_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'no_training_pricing_mcmc_or_test':True}
(root/'inputs.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps({'source':d['source_revision'],'frozen_files':len(files),'sha256':hashlib.sha256((root/'inputs.json').read_bytes()).hexdigest()}))
