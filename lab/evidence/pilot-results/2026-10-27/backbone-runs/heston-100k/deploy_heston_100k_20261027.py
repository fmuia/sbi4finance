from pathlib import Path
import hashlib,json,subprocess
root=Path('/oscar/scratch/afengler/sbi4finance-heston-100k-fit-20261027');base=Path('/oscar/scratch/afengler/sbi4finance-backbone-fit-20261027/repo');bundle=root/'source.bundle'
assert hashlib.sha256(bundle.read_bytes()).hexdigest()=='239e975141ff0f5fe5deb16041e66c66d0b9e15c3ebf05b55e2851b27c83f661'
assert not (root/'repo').exists()
for args in [['git','-C',str(base),'bundle','verify',str(bundle)],['git','-C',str(base),'fetch',str(bundle),'HEAD:refs/heads/oscar-heston-100k-20261027'],['git','-C',str(base),'worktree','add','--quiet','--detach',str(root/'repo'),'eb53cea9b26283c5d7094b022af3c9c28f44a596']]:subprocess.run(args,check=True)
for name in ('pyproject.toml','pixi.lock'):assert (root/'repo'/name).read_bytes()==(base/name).read_bytes()
record={'source_revision':'eb53cea9b26283c5d7094b022af3c9c28f44a596','bundle_sha256':hashlib.sha256(bundle.read_bytes()).hexdigest(),'locked_environment_files_match_50k_run':True,'no_existing_producer_modified':True}
(root/'manifests/deployment.json').write_text(json.dumps(record,indent=2)+chr(10));print(json.dumps(record))
