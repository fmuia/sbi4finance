"""Forward-emulator task isolation, likelihood units and export compatibility."""
import os
os.environ.setdefault('KERAS_BACKEND','jax')
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import erf
from sbi4finance.calibration import correction, pricer_net, tasks


class PricerBackboneTests(unittest.TestCase):
    def test_known_curve_has_ten_inputs_joint_has_six_without_curve_leakage(self):
        known, joint = tasks.rough_task(), tasks.joint_rough_task()
        for task, size in ((known,10),(joint,6)):
            p=np.asarray(task['prior_bounds']).mean(axis=1)[None,:]
            levels=np.full((1,7),.04)
            x=pricer_net.features(task,p,levels)
            self.assertEqual(x.shape,(1,size))
            np.testing.assert_array_equal(x[:,:len(p[0])],correction.to_unbounded(p,correction.bounds_of(task)))
            if task is known:
                np.testing.assert_array_equal(x[:,3:],np.log(levels))
                masked=np.ma.array(levels,mask=False);masked.mask[0,0]=True
                for invalid in (np.zeros((1,7)),np.full((1,7),np.nan),levels.astype(complex),masked):
                    with self.subTest(invalid=np.shape(invalid)),self.assertRaisesRegex(ValueError,'known curve'):
                        pricer_net.features(task,p,invalid)
            else:
                np.testing.assert_array_equal(x,pricer_net.features(task,p,np.full((1,7),np.nan)))
                np.testing.assert_array_equal(x,pricer_net.features(task,p))

    def test_standardized_loss_restores_quote_noise_units(self):
        true=np.array([[1.,-2.],[3.,4.]])
        predicted=np.array([[2.,0.],[4.,4.]])
        scale=np.array([2.,5.])
        actual=np.asarray(pricer_net.quote_noise_mse(true,predicted,scale))
        np.testing.assert_allclose(actual,[52.,2.])
        mean=np.mean(actual)
        self.assertEqual(float(mean),27.)
        # Any output centering cancels; the physical scaled-target residual
        # produces the same objective independently of the standardization.
        center=np.array([10.,-1.])
        expected=np.mean(((predicted*scale+center)-(true*scale+center))**2)
        self.assertEqual(float(mean),expected)
        self.assertNotEqual(float(mean),np.mean((predicted-true)**2))

    def test_legacy_unmarked_npz_preserves_tanh_and_new_marker_roundtrips(self):
        weights=[(np.array([[1.,-2.]]),np.array([.2,-.1])),(np.array([[1.],[2.]]),np.array([.3]))]
        x=np.array([[-2.],[-.5],[.8],[2.]])
        legacy=pricer_net.Pricer(weights,[0.],[1.],[0.],[1.])
        a=x@weights[0][0]+weights[0][1]
        h=.5*a*(1+np.tanh(.7978845608028654*(a+.044715*a**3)))
        expected=h@weights[1][0]+weights[1][1]
        np.testing.assert_array_equal(legacy(x),expected)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);old=root/'old';old.mkdir()
            np.savez(old/'pricer.npz',x_mean=[0.],x_sd=[1.],y_mean=[0.],y_sd=[1.],
                     w0=weights[0][0],b0=weights[0][1],w1=weights[1][0],b1=weights[1][1])
            restored=pricer_net.Pricer.load(old)
            self.assertEqual(restored.activation,'gelu_tanh')
            np.testing.assert_array_equal(restored(x),expected)
            exact=pricer_net.Pricer(weights,[0.],[1.],[0.],[1.],activation='gelu_erf')
            exact.save(root/'new');reloaded=pricer_net.Pricer.load(root/'new')
            np.testing.assert_array_equal(exact(x),reloaded(x))
            np.testing.assert_allclose(exact(x),(.5*a*(1+erf(a/np.sqrt(2))))@weights[1][0]+weights[1][1])
            self.assertGreater(np.max(np.abs(exact(x)-legacy(x))),1e-6)

    def test_tiny_actual_fit_export_reload_matches_keras_and_noise_unit_loss(self):
        import keras
        task=tasks.joint_rough_task()
        rng=np.random.default_rng(906);bounds=np.asarray(task['prior_bounds'])
        matrix=rng.normal(size=(6,63))*.2
        def split(n):
            parameters=rng.uniform(bounds[:,0]+.01*(bounds[:,1]-bounds[:,0]),bounds[:,1]-.01*(bounds[:,1]-bounds[:,0]),(n,6))
            z=correction.to_unbounded(parameters,bounds)
            noise_values=3+np.sin(z@matrix)
            return {'parameters':parameters,'reference_prices':noise_values*tasks.sigma(task),
                    'context_values':np.full((n,7),np.nan)}
        train,validation=split(32),split(8)
        models=[];constructor=keras.Sequential
        def capture(*args,**kwargs):
            model=constructor(*args,**kwargs);models.append(model);return model
        with patch.object(keras,'Sequential',side_effect=capture):
            pricer,record=pricer_net.train(task,train,validation,widths=(8,8),epochs=3,batch_size=8,patience=3,seed=0)
        self.assertEqual(pricer.activation,'gelu_erf')
        self.assertEqual(record['hidden_activation'],'gelu_erf')
        self.assertIn('quote-noise units',record['selection'])
        x=pricer_net.features(task,validation['parameters'])
        standardized=(x-pricer.x_mean)/pricer.x_sd
        expected=np.asarray(models[0](standardized,training=False))*pricer.y_sd+pricer.y_mean
        np.testing.assert_allclose(pricer(x),expected,rtol=2e-6,atol=2e-6)
        target=pricer_net.targets(task,validation['reference_prices'])
        expected_loss=np.mean((expected-target)**2)
        self.assertAlmostEqual(record['best_val_mse_quote_noise'],expected_loss,places=5)
        with tempfile.TemporaryDirectory() as temporary:
            pricer.save(Path(temporary)/'new')
            np.testing.assert_array_equal(pricer(x),pricer_net.Pricer.load(Path(temporary)/'new')(x))
        self.assertLess(record['export_max_abs_difference_quote_noise'],2e-6)

if __name__=='__main__':unittest.main()
