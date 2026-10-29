import sys,json
import numpy as np
import keras
import bayesflow as bf
keras.utils.set_random_seed(0)
if '--extra-seed' in sys.argv:keras.random.SeedGenerator()
f=bf.networks.FlowMatching(subnet_kwargs={'widths':[256]*4},time_power_law_alpha=-.5)
f.build((3,4),(3,63))
r=np.random.default_rng(37);x=r.normal(size=(3,4)).astype('float32');c=r.normal(size=(3,63)).astype('float32');t=np.array([[.05],[.5],[.95]],dtype='float32')
v=np.asarray(f.velocity(x,time=t,conditions=c,training=False))
result={'version':bf.__version__,'velocity':v.tolist(),'output_shape':list(v.shape),'input_dim':int(f.subnet.x_proj.kernel.shape[0]),'condition_dim':int(f.subnet.c_proj.kernel.shape[0]),'time_features':int(f.subnet.time_emb.compute_output_shape((3,1))[-1])}
if hasattr(f,'mask_handler'):
 masks=f.mask_handler.resolve({},targets=x,conditions=c,stage='training',subnet=f.subnet)
 result['masks']={n:None if v is None else np.asarray(v).tolist() for n,v in vars(masks).items()}
print(json.dumps(result))
