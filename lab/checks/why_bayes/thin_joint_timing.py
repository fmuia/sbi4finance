"""Time one EWG-layout surface under the joint-curve task's pricer, at the path counts a bank would use.

Needs the joint task of PR #48. Run from lab/: python checks/why_bayes/thin_joint_timing.py configs/thin/ewg_layout.json
"""

import json
import sys
import time

import numpy as np

from sbi4finance.calibration import pricing, tasks, thin

layout = json.load(open(sys.argv[1]))
task = tasks.joint_rough_task()
contracts = [task["context"]["contracts"][i] for i in layout["contract_indices"]]
noise = np.asarray(layout["noise_sd"], dtype=float)
days = sorted(set(layout["maturity_days"]))
blocks = thin.blocks_for(days)
theta = np.array([0.09, 2.6, -0.73, 0.02, 0.06, 0.5])
curve = tasks.joint_curve(task, theta)
print("contracts", len(contracts), "days", days, "blocks", blocks)
for paths in (4096, 8192, 16384, 32768):
    t0 = time.process_time()
    p, s, c, used = pricing.rough_surface(tasks.model_parameters(task, theta), curve, contracts, blocks, [paths],
                                          noise, 0.25, 7, cap=np.inf)
    print(f"{paths} paths: {time.process_time() - t0:.2f} CPU-s, largest SE/noise {np.max(s / noise):.3f}")
