"""Predictive pictures keep the data, finite-frequency conventions and IID control distinct."""

import copy
import unittest

import numpy as np

from sbi4f_tutorial.trajectories import model_checks, predictive_visuals


def example(length=45, count=4):
    rng = np.random.default_rng(21)
    observed = rng.normal(.0003,.012,length)
    replicated = rng.normal(0,.015,(count,length))
    return {"observed_returns":observed[:,None],"replicated_prefix":replicated[:,:,None],
            "kind":"prior_predictive","length":length,"metadata":{"count":count},
            "ordering":model_checks._ordering(replicated[:,:,None],observed[:,None])}


class TrajectoryPredictiveVisualTests(unittest.TestCase):
    def tearDown(self):
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_gaussian_fit_uses_observations_without_model_parameters_or_future(self):
        result = example()
        first = predictive_visuals.compare(result)
        changed = copy.deepcopy(result)
        changed["replicated_prefix"] *= 100
        changed["selected_parameters"] = np.full((4,7),1e200)
        changed["replicated_returns"] = np.full((4,1260,1),np.nan)
        second = predictive_visuals.compare(changed)
        np.testing.assert_array_equal(first["gaussian"],second["gaussian"])
        observed = result["observed_returns"][:,0]
        self.assertAlmostEqual(first["mean"],observed.mean())
        self.assertAlmostEqual(first["sd"],observed.std(ddof=0))
        self.assertEqual(first["metadata"]["gaussian_count"],200)

    def test_all_model_rows_are_copied_not_replaced_or_modified(self):
        result = example(count=7)
        before = result["replicated_prefix"].copy()
        comparison = predictive_visuals.compare(result)
        np.testing.assert_array_equal(comparison["replicated"],before[:,:,0])
        comparison["replicated"][0,0]=2
        np.testing.assert_array_equal(result["replicated_prefix"],before)
        self.assertEqual(comparison["metadata"]["model_count"],7)

    def test_one_day_and_constant_histories_leave_gaussian_unavailable(self):
        for length in (1,45):
            with self.subTest(length=length):
                result=example(length=length)
                result["observed_returns"][:]=.001
                result["ordering"]=model_checks._ordering(result["replicated_prefix"],result["observed_returns"])
                comparison=predictive_visuals.compare(result)
                self.assertFalse(comparison["metadata"]["gaussian_available"])
                self.assertEqual(comparison["gaussian"].shape,(0,length))
                self.assertEqual(len(comparison["replicated"]),4)
                for row in predictive_visuals.feature_rows(comparison):
                    self.assertIn("Unavailable",row["Fitted IID Gaussian central 90%"])
                predictive_visuals.paths_and_distributions(comparison)
                predictive_visuals.timing(result,comparison)

    def test_constant_model_acfs_are_counted_as_undefined_not_removed(self):
        result=example(count=3)
        result["replicated_prefix"][0]=0
        result["ordering"]=model_checks._ordering(result["replicated_prefix"],result["observed_returns"])
        comparison=predictive_visuals.compare(result)
        acf=comparison["features"]["model"]["acf"]
        self.assertEqual(acf["available_count"],2)
        self.assertEqual(acf["count"],3)
        self.assertTrue(np.isnan(comparison["features"]["model"]["absolute_lag_one"][0]))
        rows=predictive_visuals.feature_rows(comparison)
        self.assertIn("2/3",rows[1]["Model central 90%"])

    def test_visible_histogram_does_not_renormalize_away_out_of_view_days(self):
        result=example(length=30,count=2)
        result["replicated_prefix"][:]=0
        result["replicated_prefix"][0,0,0]=50.
        comparison=predictive_visuals.compare(result)
        figure=predictive_visuals.paths_and_distributions(comparison)
        distribution=figure.axes[1]
        line=next(x for x in distribution.lines if x.get_label()=="Model replicas")
        self.assertAlmostEqual(float(np.sum(line.get_ydata())),59/60)
        self.assertEqual(comparison["replicated"][0,0],50.)
        self.assertTrue(any("model 1," in text.get_text() for text in figure.axes[0].texts))

    def test_zero_tail_frequencies_remain_in_the_plot(self):
        result=example()
        comparison=predictive_visuals.compare(result)
        figure=predictive_visuals.paths_and_distributions(comparison)
        tail=figure.axes[3]
        self.assertEqual(tail.get_yscale(),"symlog")
        model_line=next(line for line in tail.lines if line.get_label()=="Model median / 90%")
        self.assertEqual(model_line.get_ydata()[-1],0)
        self.assertTrue(np.isfinite(model_line.get_ydata()).all())

    def test_fixed_views_are_shared_between_prior_and_small_posterior_panels(self):
        prior=example(count=200)
        posterior=example(count=4)
        posterior["kind"]="posterior_predictive"
        figures=[predictive_visuals.paths_and_distributions(predictive_visuals.compare(r))
                 for r in (prior,posterior)]
        for figure in figures:
            self.assertEqual(figure.axes[0].get_ylim(),(-15.,15.))
            self.assertEqual(figure.axes[2].get_ylim(),(-200.,200.))
        self.assertTrue(any("all 4 paths" in x.get_label() for x in figures[1].axes[0].lines))

    def test_iid_reference_does_not_assume_finite_sample_acf_exactly_zero(self):
        result=example(length=200)
        comparison=predictive_visuals.compare(result)
        acf=comparison["features"]["gaussian"]["absolute_lag_one"]
        self.assertGreater(float(np.std(acf)),0)
        self.assertTrue(np.any(acf>0) and np.any(acf<0))
        self.assertEqual(len(acf),200)

    def test_malformed_inputs_are_not_coerced_into_valid_histories(self):
        for kind in ("complex","masked","nonfinite","shape","count"):
            with self.subTest(kind=kind):
                result=example()
                if kind=="complex":
                    result["observed_returns"]=result["observed_returns"].astype(complex)+1j
                elif kind=="masked":
                    result["replicated_prefix"]=np.ma.array(result["replicated_prefix"],mask=False)
                elif kind=="nonfinite":
                    result["replicated_prefix"][0,0,0]=np.inf
                elif kind=="shape":
                    result["observed_returns"]=result["observed_returns"][:,0]
                else:
                    result["metadata"]["count"]=5
                with self.assertRaises(ValueError):
                    predictive_visuals.compare(result)


if __name__=="__main__":
    unittest.main()
