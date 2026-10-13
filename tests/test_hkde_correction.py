"""Proposal-law, raw-density Jacobian, weighted summaries and joint SIR checks."""

import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import norm

from sbi4finance import hkde_correction as correction, hkde_inference as inference


ROOT = Path(__file__).resolve().parents[1]
TASK = json.loads((ROOT / "configs/tasks/hkde_inference_bank.json").read_text())["task"]
BOUNDS = np.asarray(TASK["prior_bounds"])
WIDTHS = np.diff(BOUNDS, axis=1)[:, 0]
OBSERVATION = np.zeros((9, 1))


class AnalyticWorkflow:
    """Known q: scaled Beta(2,1) for theta, independent uniform intensity."""
    def sample(self, *, num_samples, conditions, seed):
        assert conditions["observations"].shape == (1, 9)
        unit = np.random.default_rng(seed).uniform(size=(num_samples, 2))
        unit[:, 0] = np.sqrt(unit[:, 0])
        values = BOUNDS[:, 0] + WIDTHS*unit
        return {name: values[None, :, d:d+1] for d, name in enumerate(inference.PARAMETER_NAMES)}

    def log_prob(self, data):
        count = len(data["theta"])
        assert data["theta"].shape == data["jump_intensity"].shape == (count, 1)
        assert data["observations"].shape == (count, 9)
        u = (data["theta"][:, 0]-BOUNDS[0, 0])/WIDTHS[0]
        return np.log(2*u)-np.log(WIDTHS).sum()


class HKDECorrectionMathTests(unittest.TestCase):
    def test_defensive_denominator_is_whole_mixture_for_both_components(self):
        proposal = correction.propose(AnalyticWorkflow(), TASK, OBSERVATION, 4096, seed=661)
        u = (proposal["parameters"][:, 0]-BOUNDS[0, 0])/WIDTHS[0]
        expected = np.log(.9*2*u+.1)-np.log(WIDTHS).sum()
        assert_allclose(proposal["log_proposal"], expected, atol=1e-14)
        self.assertTrue(np.any(proposal["is_prior"]))
        self.assertTrue(np.any(~proposal["is_prior"]))
        self.assertLess(abs(proposal["is_prior"].mean()-.1), .02)
        self.assertLess(abs(u.mean()-(.9*2/3+.1/2)), .02)
        self.assertTrue(np.all(np.exp(proposal["log_proposal"]) >= .1/np.prod(WIDTHS)))
        replay = correction.propose(AnalyticWorkflow(), TASK, OBSERVATION, 4096, seed=661)
        for key in ("parameters", "log_proposal", "is_prior"):
            assert_array_equal(proposal[key], replay[key])

    def test_weight_correction_recovers_uniform_target_from_nonuniform_proposal(self):
        proposal = correction.propose(AnalyticWorkflow(), TASK, OBSERVATION, 32768, seed=662)
        # A constant quote likelihood leaves the uniform prior as the target.
        result = correction.correct(proposal["parameters"], proposal["log_proposal"], OBSERVATION,
                                    TASK, prices=np.zeros((32768, 9)))
        weights = result["weights"]
        expected = np.exp(-proposal["log_proposal"])
        expected /= expected.sum()
        assert_allclose(weights, expected, rtol=1e-13)
        unit = (proposal["parameters"]-BOUNDS[:, 0])/WIDTHS
        assert_allclose(weights @ unit, [.5, .5], atol=.007)
        self.assertAlmostEqual(weights.sum(), 1.)
        self.assertAlmostEqual(result["ess"], 1/np.sum(weights**2))
        self.assertAlmostEqual(result["ess_fraction"], result["ess"]/32768)
        sigma = TASK["context"]["quote_noise_std"]
        assert_allclose(result["log_likelihood"], -9*np.log(sigma*np.sqrt(2*np.pi)))

    def test_boundary_mixture_density_component_law_and_overlap(self):
        prior_fraction, boundary_fraction = .15, .25
        proposal = correction.propose(AnalyticWorkflow(), TASK, OBSERVATION, 65536, seed=667,
                                      prior_fraction=prior_fraction, boundary_fraction=boundary_fraction)
        unit = (proposal["parameters"]-BOUNDS[:, 0])/WIDTHS
        # Independent analytic density on normalized coordinates, including the
        # two scale Jacobians. Every generated component uses this denominator.
        boundary = 1/(np.pi**2*np.sqrt(np.prod(unit*(1-unit), axis=1)))
        expected = np.log(.6*2*unit[:, 0]+prior_fraction+boundary_fraction*boundary)-np.log(WIDTHS).sum()
        assert_allclose(proposal["log_proposal"], expected, rtol=1e-13, atol=1e-13)
        masks = [proposal["is_prior"], proposal["is_boundary"],
                 ~(proposal["is_prior"] | proposal["is_boundary"])]
        self.assertFalse(np.any(masks[0] & masks[1]))
        for mask, fraction in zip(masks, [.15, .25, .6], strict=True):
            self.assertGreater(mask.sum(), 1000)
            self.assertLess(abs(mask.mean()-fraction), .007)
        assert_allclose(unit[masks[1]].mean(axis=0), [.5,.5], atol=.012)
        assert_allclose((unit[masks[1]]**2).mean(axis=0), [.375,.375], atol=.012)
        self.assertLess(abs(np.corrcoef(unit[masks[1]].T)[0,1]), .03)
        self.assertTrue(np.all(np.exp(proposal["log_proposal"]) >= prior_fraction/np.prod(WIDTHS)))
        # Importance integration against the normalized three-component law
        # recovers both the uniform normalizer and uniform target moments.
        inverse_density = np.exp(-proposal["log_proposal"])/np.prod(WIDTHS)
        self.assertLess(abs(inverse_density.mean()-1), .012)
        assert_allclose(np.mean(inverse_density[:,None]*unit,axis=0), [.5,.5], atol=.01)
        fixed = correction.correct(proposal["parameters"], proposal["log_proposal"], OBSERVATION,
                                   TASK, prices=np.zeros((len(unit),9)))
        assert_allclose(fixed["weights"] @ unit, [.5,.5], atol=.007)

    def test_zero_boundary_component_preserves_v1_and_positive_component_replays(self):
        baseline = correction.propose(AnalyticWorkflow(), TASK, OBSERVATION, 4096, seed=668)
        zero = correction.propose(AnalyticWorkflow(), TASK, OBSERVATION, 4096, seed=668, boundary_fraction=0.)
        positive = correction.propose(AnalyticWorkflow(), TASK, OBSERVATION, 4096, seed=668, boundary_fraction=.1)
        replay = correction.propose(AnalyticWorkflow(), TASK, OBSERVATION, 4096, seed=668, boundary_fraction=.1)
        for key in ("parameters", "log_proposal", "is_prior", "is_boundary"):
            assert_array_equal(baseline[key],zero[key])
            assert_array_equal(positive[key],replay[key])
        self.assertFalse(zero["is_boundary"].any())
        assert_array_equal(positive["is_prior"],baseline["is_prior"])
        retained = ~positive["is_boundary"]
        assert_array_equal(positive["parameters"][retained],baseline["parameters"][retained])
        # At every retained row the new density includes at least .8/.9 times
        # the original density, including rows generated by the uniform prior.
        self.assertTrue(np.all(positive["log_proposal"][retained] >=
                               baseline["log_proposal"][retained]+np.log(.8/.9)-1e-14))

    def test_boundary_fraction_and_endpoint_failures_are_explicit(self):
        for fraction in (-.1, .9, 1., np.nan, np.inf, True):
            with self.assertRaises(ValueError):
                correction.propose(AnalyticWorkflow(),TASK,OBSERVATION,32,seed=669,
                                   boundary_fraction=fraction)
        # No clipping of an unrepresentable endpoint back into the domain.
        with patch.object(correction.np.random, "default_rng") as factory:
            factory.return_value.random.return_value = np.full(32,.15)
            factory.return_value.uniform.return_value = np.tile(BOUNDS.mean(axis=1),(32,1))
            factory.return_value.beta.return_value = np.zeros((32,2))
            workflow = AnalyticWorkflow()
            center = {name: np.full((1,32,1),BOUNDS[d].mean())
                      for d,name in enumerate(inference.PARAMETER_NAMES)}
            with patch.object(workflow,"sample",return_value=center), self.assertRaises(ValueError):
                correction.propose(workflow,TASK,OBSERVATION,32,seed=669,boundary_fraction=.1)

    def test_exact_likelihood_path_and_extreme_log_weights(self):
        parameters = BOUNDS.mean(axis=1)[None]+np.array([[-.001,-.01],[.001,.01]])
        log_proposal = np.full(2, -np.log(WIDTHS).sum())
        prices = np.ones((2, 9))*1000.
        prices[1, 0] += .00001
        with patch.object(correction.reference, "price_parameters", return_value=prices) as price:
            result = correction.correct(parameters, log_proposal, OBSERVATION, TASK)
        price.assert_called_once_with(TASK, parameters)
        self.assertTrue(np.isfinite(result["weights"]).all())
        self.assertGreater(result["weights"][0], result["weights"][1])
        self.assertAlmostEqual(result["weights"].sum(), 1.)

    def test_neural_negative_infinity_density_still_has_prior_support(self):
        workflow = AnalyticWorkflow()
        with patch.object(workflow, "log_prob", side_effect=lambda data: np.full(len(data["theta"]), -np.inf)):
            result = correction.propose(workflow, TASK, OBSERVATION, 32, seed=663)
        assert_allclose(result["log_proposal"], np.log(.1)-np.log(WIDTHS).sum())
        with patch.object(workflow, "log_prob", side_effect=lambda data: np.full(len(data["theta"]), np.nan)):
            with self.assertRaises(ValueError):
                correction.propose(workflow, TASK, OBSERVATION, 32, seed=663)

    def test_weighted_joint_summary_reference_mass_and_block_uncertainty(self):
        rng = np.random.default_rng(664)
        parameters = rng.multivariate_normal([0,0], [[1,.7],[.7,1]], size=32768)
        targets = parameters*np.array([2.,1.])+[3.,4.]
        axis = np.linspace(-8, 8, 2049)
        ref = {"mean": [0.,0.], "std": [1.,1.], "correlation": .7,
               "probabilities": [.05,.5,.95], "marginal_axes": [axis,axis],
               "marginal_cdfs": [norm.cdf(axis),norm.cdf(axis)],
               "target_quantiles": np.array([3.,4.])[:,None]+np.array([2.,1.])[:,None]*norm.ppf([.05,.5,.95])}
        result = correction.summarize(parameters, np.ones(len(parameters)), targets, ref)
        assert_allclose(result["mean_error_in_ref_sd"], 0, atol=.02)
        assert_allclose(result["relative_std_error"], 0, atol=.02)
        self.assertLess(abs(result["correlation_error"]), .02)
        assert_allclose(result["reference_mass_of_90_interval"], [.9,.9], atol=.012)
        assert_allclose(result["target_quantile_error_currency"], 0, atol=.06)
        assert_allclose(result["uncertainty"]["mean_snis_mcse"],
                        parameters.std(axis=0, ddof=1)/np.sqrt(len(parameters)), rtol=1e-13)
        assert_allclose(result["uncertainty"]["target_mean_snis_mcse"],
                        targets.std(axis=0, ddof=1)/np.sqrt(len(parameters)), rtol=1e-13)
        blockmeans = np.array(result["uncertainty"]["block_parameter_means"])
        assert_allclose(result["uncertainty"]["parameter_mean_block_mcse"], blockmeans.std(axis=0,ddof=1)/np.sqrt(8))
        self.assertEqual(np.shape(result["uncertainty"]["target_quantile_block_mcse"]), (2,3))
        shuffled=parameters.copy()
        rng.shuffle(shuffled[:,1])
        changed=correction.summarize(shuffled,np.ones(len(parameters)),targets,ref)
        self.assertGreater(abs(changed["correlation_error"]), .65)
        assert_allclose(changed["parameter_quantiles"],result["parameter_quantiles"])
        json.dumps(result, allow_nan=False)

    def test_resampling_preserves_joint_pairs_and_is_repeatable(self):
        parameters = np.array([[.03,.5],[.04,.7],[.05,.9]])
        weights = np.array([0.,1.,3.])
        draws = correction.resample(parameters, weights, 4096, seed=665)
        assert_array_equal(draws, correction.resample(parameters,weights,4096,seed=665))
        self.assertTrue(all(any(np.array_equal(row,p) for p in parameters[1:]) for row in draws))
        self.assertLess(abs(np.mean(draws[:,0]==.05)-.75), .025)

    def test_shape_support_weight_and_block_failures_are_explicit(self):
        proposal = correction.propose(AnalyticWorkflow(), TASK, OBSERVATION, 32, seed=666)
        parameters, logq = proposal["parameters"], proposal["log_proposal"]
        for invalid in (np.zeros(9), np.full((9,1),np.nan), np.zeros((8,1))):
            with self.assertRaises(ValueError):
                correction.propose(AnalyticWorkflow(), TASK, invalid, 32, seed=666)
        for fraction in (0.,1.,-1.,np.nan,True):
            with self.assertRaises(ValueError):
                correction.propose(AnalyticWorkflow(),TASK,OBSERVATION,32,seed=666,prior_fraction=fraction)
        for density in (logq[:,None],np.full(32,np.nan),np.full(32,-np.inf)):
            with self.assertRaises(ValueError):
                correction.correct(parameters,density,OBSERVATION,TASK,prices=np.zeros((32,9)))
        with self.assertRaises(ValueError):
            correction.correct(BOUNDS.T,logq[:2],OBSERVATION,TASK,prices=np.zeros((2,9)))
        with self.assertRaises(ValueError):
            correction.correct(parameters,logq,OBSERVATION,TASK,prices=np.zeros((32,2)))
        for weights in (np.zeros(32),np.full(32,-1.),np.ones((32,1))):
            with self.assertRaises(ValueError):
                correction.summarize(parameters,weights,parameters)
        with self.assertRaises(ValueError):
            correction.summarize(parameters,np.ones(32),parameters,blocks=3)
        with self.assertRaises(ValueError):
            correction.summarize(parameters,np.r_[np.ones(4),np.zeros(28)],parameters)
        with self.assertRaises(ValueError):
            correction.resample(parameters,np.ones(32),0,seed=666)


class HKDECorrectionDensityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("KERAS_BACKEND","jax")
        cls.workflow=inference.build_workflow(TASK)
        cls.workflow.load_approximator(ROOT/"checkpoints/hkde-theta-intensity-provisional/model.keras")
        with np.load(ROOT/"docs/results/hkde-inference/validation-checks.npz",allow_pickle=False) as file:
            cls.observation=file["observations"][0]

    def test_raw_density_has_adapter_and_fitted_standardizer_jacobians_once(self):
        import keras
        fractions=np.array([[.01,.02],[.1,.9],[.5,.5],[.85,.3],[.98,.99]])
        parameters=BOUNDS[:,0]+WIDTHS*fractions
        actual=correction._flow_log_prob(self.workflow,TASK,self.observation,parameters)
        approximate=self.workflow.approximator
        target_layer=approximate.standardizer.standardize_layers["inference_variables"]
        context_layer=approximate.standardizer.standardize_layers["inference_conditions"]
        before=[np.asarray(weight).copy() for layer in (target_layer,context_layer) for weight in layer.weights]
        u=(parameters-BOUNDS[:,0])/WIDTHS
        logits=(np.log(u)-np.log1p(-u)).astype(np.float32)
        std=np.asarray(target_layer.moving_std(0))
        normalized=(logits-np.asarray(target_layer.moving_mean[0]))/std
        conditions=np.repeat(self.observation[:,0][None],len(parameters),axis=0).astype(np.float32)
        conditions=(conditions-np.asarray(context_layer.moving_mean[0]))/np.asarray(context_layer.moving_std(0))
        latent=np.asarray(keras.ops.convert_to_numpy(approximate.inference_network.log_prob(normalized,conditions=conditions)))
        jacobian=-np.sum(np.log(u)+np.log1p(-u)+np.log(WIDTHS),axis=1)-np.log(std.astype(np.float64)).sum()
        assert_allclose(actual,latent+jacobian,rtol=1e-6,atol=2e-5)
        for old,new in zip(before,[np.asarray(w) for layer in (target_layer,context_layer) for w in layer.weights],strict=True):
            assert_array_equal(old,new)
        self.assertGreater(np.max(np.abs(actual-latent)),1.)

    def test_saved_flow_raw_density_integrates_to_one_on_the_declared_box(self):
        # Open Gauss-Legendre nodes avoid boundary logits. This is a density
        # adapter/normalization check, not a posterior accuracy experiment.
        axis,weights=np.polynomial.legendre.leggauss(128)
        unit=np.stack(np.meshgrid((axis+1)/2,(axis+1)/2,indexing='ij'),axis=-1).reshape(-1,2)
        parameters=BOUNDS[:,0]+WIDTHS*unit
        logq=correction._flow_log_prob(self.workflow,TASK,self.observation,parameters)
        mass=np.sum(np.exp(logq)*np.outer(weights,weights).ravel()*np.prod(WIDTHS)/4)
        self.assertLess(abs(mass-1.),.003)


if __name__=="__main__":
    unittest.main()
