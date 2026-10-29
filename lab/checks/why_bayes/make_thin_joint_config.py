"""Write a thin-layout variant of PR #48's joint-curve bank configuration.

The task keeps the six targets, the prior box and the pricing numerics of the joint pilot; the contracts become the
layout's (EWG's 27 of the 63 design contracts), the quote noise the layout's, the simulation blocks those of the
thin studies (thin.blocks_for) and the curve knots the layout's maturities. The ladder starts at 4,096 paths,
which meets the bank's precision budget on this layout. The bank builder accepts only task, root_seed and splits,
so the layout's provenance goes to a sidecar file beside the configuration.

Needs the joint task of PR #48. Run from lab/:
python checks/why_bayes/make_thin_joint_config.py configs/tasks/calibration_rough_joint_pilot.json \
    configs/thin/ewg_layout.json OUT.json TRAIN VALIDATION TEST SEED
then: python scripts/calibration_bank.py local --config OUT.json --chunk 10 --workers 4 --out BANK_DIR
"""

import json
import sys
from pathlib import Path

from sbi4finance.calibration import tasks, thin

pilot = json.loads(Path(sys.argv[1]).read_text())
layout = json.loads(Path(sys.argv[2]).read_text())
out = Path(sys.argv[3])
train, validation, test, seed = (int(v) for v in sys.argv[4:8])
task = pilot["task"]
design = task["context"]["contracts"]
idx = layout["contract_indices"]
days = sorted(set(layout["maturity_days"]))
task["context"]["contracts"] = [design[i] for i in idx]
task["context"]["quote_noise_std"] = list(layout["noise_sd"])
task["numerics"]["blocks"] = thin.blocks_for(days)
task["numerics"]["paths_ladder"] = [4096, 8192, 16384, 32768, 65536, 131072]
task["model_context"]["forward_variance_knots"] = [0.0] + [d / 252 for d in days]
tasks.validate_task(task)
config = dict(pilot, task=task, root_seed=seed)
config["splits"] = dict(config["splits"], train=train, validation=validation, test=test)
out.write_text(json.dumps(config, indent=1) + "\n")
note = {"ticker": layout["ticker"], "asof": layout["asof"], "layout_task_fingerprint": layout["task_fingerprint"],
        "contract_indices": idx, "maturity_days": days}
out.with_name(out.stem + "-layout-note.json").write_text(json.dumps(note, indent=1) + "\n")
print(f"{out}: {len(task['context']['contracts'])} contracts, blocks {task['numerics']['blocks']}, "
      f"splits {config['splits']}")
