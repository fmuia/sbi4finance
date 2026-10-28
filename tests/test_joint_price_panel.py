"""Check that the paired diagnostic preserves the actual analytic curve."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
from scripts.validate_rough_price_panel import fine_curve
from sbi4finance.bergomi import curve_values
from sbi4finance.calibration import pricing, tasks


class JointPricePanelTests(unittest.TestCase):
    def test_analytic_curve_is_evaluated_on_refined_nodes(self):
        task=tasks.joint_rough_task();theta=np.array([.12,1.8,-.7,.09,.02,6.])
        steps=[2*b['n_steps'] for b in task['numerics']['blocks']]
        curve=fine_curve(task,theta,steps)
        for block,n in zip(task['numerics']['blocks'],steps):
            times=block['horizon']/n*np.arange(n)
            expected=theta[4]+(theta[3]-theta[4])*np.exp(-theta[5]*times)
            np.testing.assert_allclose(curve_values(curve,times),expected,rtol=0,atol=3e-17)

    def test_reused_finest_marginal_matches_native_bank_estimator(self):
        path=Path(__file__).resolve().parents[1]/'docs/pilot-results/2026-10-28/known-pricing-validation/known_rough_pricing_check_20261028.py'
        spec=importlib.util.spec_from_file_location('original_pricing_helper',path)
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        task=tasks.joint_rough_task();theta=np.array([.12,1.8,-.7,.09,.02,6.])
        steps=[2*b['n_steps'] for b in task['numerics']['blocks']]
        curve=fine_curve(task,theta,steps);block=task['numerics']['blocks'][0]
        contracts=task['context']['contracts'][:2];parameters=tasks.model_parameters(task,theta)
        kwargs=dict(horizon=block['horizon'],n_pairs=32,seed=91)
        paired=helper.coupled_controlled_block(parameters,curve,contracts,
            steps=(block['n_steps'],steps[0]),**kwargs)
        native=pricing.rough_block(parameters,curve,contracts,n_steps=steps[0],**kwargs)
        np.testing.assert_allclose(paired['controlled_prices'][1],native[0],rtol=0,atol=1e-14)
        self.assertTrue(np.isfinite(paired['joint_covariance']).all())


if __name__=='__main__':unittest.main()
