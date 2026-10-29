import sys,json,hashlib
import numpy as np
import keras
import bayesflow as bf
keras.utils.set_random_seed(0)
if '--extra-seed' in sys.argv:keras.random.SeedGenerator()
f=bf.networks.FlowMatching(subnet_kwargs={'widths':[256]*4},time_power_law_alpha=-.5)
f.build((1,4),(1,63))
rows=[]
for w in f.trainable_variables:
 a=np.asarray(w);rows.append({'name':w.path,'shape':list(a.shape),'sha256':hashlib.sha256(a.tobytes()).hexdigest(),'first':a.reshape(-1)[:4].tolist()})
print(json.dumps({'version':bf.__version__,'extra_seed':'--extra-seed' in sys.argv,'params':sum(np.asarray(w).size for w in f.trainable_variables),'trainable':rows,'network_seed':np.asarray(f.seed_generator.state).tolist()}))
