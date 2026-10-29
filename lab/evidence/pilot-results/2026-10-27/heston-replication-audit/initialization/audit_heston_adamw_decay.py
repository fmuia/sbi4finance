"""Nominal zero-gradient AdamW decay from the exact installed Keras schedule."""
import json,hashlib
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from sbi4finance.calibration.inference import offline_optimizer
weight_decay=.005
rows=[]
for name,total_steps,selected_epoch,batches in [('archive',98000,487,196),('new50k',98000,445,196),('new100k',195500,402,391)]:
 optimizer=offline_optimizer(total_steps)
 rates=np.asarray(jax.jit(jax.vmap(optimizer._learning_rate))(jnp.arange(total_steps)),dtype=np.float64)
 for stage,n in [('selected',selected_epoch*batches),('full500',total_steps)]:
  selected=rates[:n]
  log_factor=float(np.log1p(-weight_decay*selected).sum())
  rows.append({'run':name,'stage':stage,'epochs':selected_epoch if stage=='selected' else 500,'updates':n,'total_schedule_updates':total_steps,'learning_rate_sum':float(selected.sum()),'log_nominal_decay_factor':log_factor,'nominal_decay_factor':float(np.exp(log_factor)),'nominal_percent_reduction':float(100*(-np.expm1(log_factor)))})
record={'scope':'Zero-gradient nominal decoupled shrinkage product(1-lr(step)*weight_decay). Not observed weight norms; ignores gradient updates and finite-precision in-place rounding.','weight_decay':weight_decay,'rows':rows,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
Path('/private/tmp/audit_heston_adamw_decay.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
