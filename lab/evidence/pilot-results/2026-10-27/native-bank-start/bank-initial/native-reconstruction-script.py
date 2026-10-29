"""Read saved initial training shards; reconstruct only prior and quote RNGs."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
root=Path('/oscar/scratch/afengler/sbi4finance-native-bank-20261027')
results=[]
for path in sorted((root/'collection'/'parts').glob('*/manifest.json')):
    manifest=json.loads(path.read_text());part=path.parent
    if manifest['split']!='train' or manifest['stop']>100:continue
    c=manifest['configuration'];t=c['task'];fp=manifest['task_fingerprint']
    words=[int(fp[i:i+8],16) for i in range(0,64,8)];bounds=np.asarray(t['prior_bounds']);sigma=np.asarray(t['context']['quote_noise_std'])
    with np.load(part/'arrays.npz',allow_pickle=False) as saved:arrays=dict(saved)
    for row,i in enumerate(range(manifest['start'],manifest['stop'])):
        identity=[0x4F5054,c['root_seed'],*words,0,i]
        p=np.random.Generator(np.random.PCG64(np.random.SeedSequence([*identity,0])))
        q=np.random.Generator(np.random.PCG64(np.random.SeedSequence([*identity,2])))
        assert np.array_equal(arrays['parameters'][row],p.uniform(bounds[:,0],bounds[:,1])),f'prior mismatch train:{i}'
        expected=(arrays['reference_prices'][row]+sigma*q.standard_normal(sigma.shape))[:,None]
        assert np.array_equal(arrays['observations'][row],expected),f'quote noise mismatch train:{i}'
    results.append({'passed':True,'split':'train','start':manifest['start'],'stop':manifest['stop'],'exact_linux_prior_and_quote_noise_reconstruction':True,'array_sha256':hashlib.sha256((part/'arrays.npz').read_bytes()).hexdigest(),'python':sys.version,'numpy':np.__version__,'model_calls':0})
print(json.dumps(results))
