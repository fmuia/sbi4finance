import json
from pathlib import Path
roots={
 'heston100k':'/oscar/scratch/afengler/sbi4finance-heston-100k-fit-20261027/runs/heston-100k-native/checkpoint',
 'heston50k':'/oscar/scratch/afengler/sbi4finance-backbone-fit-20261027/runs/heston-50k-native/checkpoint',
 'known20k':'/oscar/scratch/afengler/sbi4finance-rough-known-fit-20261027/runs/known-20k-native/checkpoint',
 'heston_export':'/oscar/scratch/afengler/sbi4finance-heston-teaching-20261028/checkpoint-final500'}
result={}
for name,root in roots.items():
 p=Path(root);public=p/'model.keras';files=list((p/'.resume').glob('epoch-*/*.keras'))
 result[name]={'root':root,'public_model_bytes':public.stat().st_size if public.exists() else None,
               'recovery_model_files':len(files),'recovery_model_bytes':sum(f.stat().st_size for f in files)}
print(json.dumps(result,indent=2))
