import pathlib
import sys, time, json
sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[6] / "tutorial" / "src"), str(pathlib.Path(__file__).resolve().parents[6] / "lab" / "src")]
import numpy as np
from sbi4f_tutorial.shared import pricing, surface, tasks
from sbi4finance.calibration import tasks as lab_tasks
task = json.load(open(str(pathlib.Path(__file__).resolve().parents[6]) + '/tutorial/assets/backbones/rough-joint100k-validation0/task.json'))
b = np.array(task['prior_bounds']); theta = b.mean(1)
params = tasks.model_parameters(task, theta)
curve = lab_tasks.joint_curve(task, theta)
contracts = task['context']['contracts']
for n in [int(sys.argv[1])]:
    t0 = time.time()
    p, s, c, used = pricing.rough_surface(params, curve, contracts, task['numerics']['blocks'], [n], tasks.sigma(task), np.inf, 1)
    print(n, time.time() - t0, np.max(s / tasks.sigma(task)), np.median(s/tasks.sigma(task)))
