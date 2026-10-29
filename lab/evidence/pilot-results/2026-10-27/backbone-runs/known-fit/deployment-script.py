from pathlib import Path
import hashlib,json,subprocess
root=Path('/oscar/scratch/afengler/sbi4finance-rough-known-fit-20261027');base=Path('/oscar/scratch/afengler/sbi4finance-backbone-fit-20261027/repo');bundle=root/'source.bundle'
assert hashlib.sha256(bundle.read_bytes()).hexdigest()=='93d620f835cc7ccbbffae24b926d3ed1cf340c53ea49824f29dd7ec09eeeac99'
assert not (root/'repo').exists()
for args in [['git','-C',str(base),'bundle','verify',str(bundle)],['git','-C',str(base),'fetch',str(bundle),'HEAD:refs/heads/oscar-rough-known-fit-20261027'],['git','-C',str(base),'worktree','add','--quiet','--detach',str(root/'repo'),'845c458b849c472271527cccf815ea6714f02206']]:subprocess.run(args,check=True)
for name in ('pyproject.toml','pixi.lock'):assert (root/'repo'/name).read_bytes()==(base/name).read_bytes()
record={'source_revision':'845c458b849c472271527cccf815ea6714f02206','bundle_sha256':hashlib.sha256(bundle.read_bytes()).hexdigest(),'locked_environment_files_match_50k_run':True,'no_existing_producer_modified':True}
(root/'manifests/deployment.json').write_text(json.dumps(record,indent=2)+chr(10));print(json.dumps(record))
