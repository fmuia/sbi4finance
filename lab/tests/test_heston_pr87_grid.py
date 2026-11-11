"""63-feature native recipe, immutable physical pairing and stored-error reuse."""
from copy import deepcopy
from importlib import metadata
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scripts import reproduce_heston_pr87_grid as t
from test_heston_pr87_followup import parent_fixture


def fixture(root):
    parent, _ = parent_fixture(root)
    old = parent['task']['records']
    records = {k: [v[i % 54] for i in range(63)] for k, v in old.items()}
    records['strike'] = np.linspace(130., 200., 63).tolist()
    for key in records: records[key][7] = old[key][5]
    model = t.HestonSpreadPilot(records, infer_c=True, c_bounds=(.5, 2.))
    task = {'format': 'heston_pr87_grid_task_v1', 'records': records, 'target': model.task_definition()}
    truth = t._truth(parent['arrays'], 'train', model.names)[:5].copy()
    history = {'truth': truth, 'row_id': np.asarray([f'history:{i:03d}' for i in range(5)]),
               'identity': {'manifest': {'path': str(root/'history.json'), 'sha256': 'fixture'}}}
    c = {'format': 'heston_pr87_grid_bank_contract_v1', 'train_rows': 135, 'parent_train_rows': 128,
         'validation_rows': 32, 'historical_rows': 5, 'whitening_rows': 128, 'quote_count': 63,
         'shared_contracts': t.shared_contracts(parent['task'], task), 'parent': parent['identity'],
         'historical_inputs': history['identity'], 'smoke_test_only': True}
    return task, model, parent, history, c


class WhiteningAndRawBoundaryTests(unittest.TestCase):
    def test_native63_equation_and_corruption(self):
        x = np.random.default_rng(52).normal(size=(128, 63)); w = t.NotebookWhitening(x)
        ev, evec = np.linalg.eigh(np.cov(x, rowvar=False))
        matrix = (evec / np.sqrt(np.maximum(ev, ev[-1]*1e-8))) @ evec.T
        np.testing.assert_array_equal(w.transform(x), (x-x.mean(axis=0)) @ matrix)
        saved = w.to_dict(); loaded = t.NotebookWhitening.from_dict(saved)
        np.testing.assert_array_equal(w.transform(x), loaded.transform(x))
        saved['whitening_matrix'][0][0] += 1
        with self.assertRaises(ValueError): t.NotebookWhitening.from_dict(saved)
        with self.assertRaises(ValueError): w.transform(np.ones((3, 54)))

    def test_raw63_wrapper_keeps_unbounded_draws_and_transforms_once(self):
        class Native:
            estimator = None
            def sample(self, **kwargs):
                self.kwargs = kwargs
                return {n: np.full((2, 3, 1), i-100.) for i, n in enumerate(names)}
        names = ['initial_variance', 'kappa', 'theta', 'vol_of_vol', 'rho', 'c']
        task = {'records': {'half_spread': [2.]*63}, 'target': {'parameter_names': names}}
        w = t.NotebookWhitening(np.random.default_rng(44).normal(size=(128, 63)))
        native = Native(); wrapper = t.RawQuoteWorkflow(native, task, w); q = np.full((2, 63), -4.)
        draws = wrapper.sample(q, 3, 73)
        np.testing.assert_array_equal(draws[0, 0], np.arange(6)-100.)
        np.testing.assert_array_equal(native.kwargs['conditions']['observations'], w.transform(q/2))
        self.assertEqual(native.kwargs['seed'], 73)


try: CORE = metadata.version('bayesflow') == '2.1.0'
except metadata.PackageNotFoundError: CORE = False


@unittest.skipUnless(CORE, 'requires locked BF2.1 environment')
class NativeGridDataTests(unittest.TestCase):
    def test_exact_rng_prior_block_full_noise_draws_and_shared_error_pairing(self):
        with tempfile.TemporaryDirectory() as tmp:
            task, model, parent, history, c = fixture(Path(tmp))
            before = deepcopy(parent['identity']['continuation_rng_state']); clean = np.arange(63.)+100
            with patch.object(t.HestonSpreadPilot, 'prices', return_value=clean) as pricing:
                arrays, states, pairing = t.simulate_bank(task, parent, history, c)
            rng = np.random.default_rng(); rng.bit_generator.state = before
            old = t._truth(parent['arrays'], 'train', model.names)
            fresh = np.asarray([rng.uniform(model.bounds[:, 0], model.bounds[:, 1]) for _ in range(7)])
            h = np.asarray(task['records']['half_spread'])
            truths = {'train': np.concatenate((old, fresh)), 'validation': t._truth(parent['arrays'], 'validation', model.names), 'historical': history['truth']}
            for split, truth in truths.items():
                expected = np.asarray([clean+row[-1]*h*rng.standard_normal(63) for row in truth])
                if split != 'historical':
                    n = 128 if split == 'train' else 32; scale = truth[:n, -1]*h[7]
                    z = (parent['arrays'][split+'_quotes'][:, 5]-parent['arrays'][split+'_clean_prices'][:, 5])/scale
                    expected[:n, 7] = clean[7]+scale*z
                np.testing.assert_array_equal(arrays[split+'_quotes'], expected)
                np.testing.assert_array_equal(arrays[split+'_observations'], expected/h)
                np.testing.assert_array_equal(t._truth(arrays, split, model.names), truth)
                self.assertEqual(states['after_'+split], rng.bit_generator.state)
            self.assertEqual(pricing.call_count, 172)
            self.assertEqual(pairing['train']['rows'], 128)
            self.assertGreater(pairing['train']['max_absolute_clean_shift'], 1)
            self.assertEqual(parent['identity']['continuation_rng_state'], before)
            arrays['train_kappa'][0, 0] += .01
            with self.assertRaisesRegex(ValueError, 'truths'): t._validate_pairing(arrays, task, parent, history, c)

    def test_both_native_horizons_resolve_without_pricing(self):
        with tempfile.TemporaryDirectory() as tmp:
            task, _, parent, history, _ = fixture(Path(tmp))
            parent['identity'].update(train_rows=200000, validation_rows=100)
            with patch.object(t, '_read_parent', return_value=parent), patch.object(t, '_read_historical', return_value=history), patch.object(t, 'source_bindings', return_value={'producer': 'fixture'}), patch.object(t.HestonSpreadPilot, 'prices', side_effect=AssertionError('no pricing')):
                a = t.protocol_contract(task, 'parent', 'history', train_rows=200000)
                b = t.protocol_contract(task, 'parent', 'history', train_rows=300000)
            self.assertEqual(a['bank_contract'], b['bank_contract'])
            self.assertEqual((a['example_presentations'], a['expected_optimizer_updates'], a['tail_batch_rows']), (60000000, 234600, 64))
            self.assertEqual((b['example_presentations'], b['expected_optimizer_updates'], b['tail_batch_rows']), (90000000, 351600, 224))
            for c, warm, decay in [(a, 11730, 222870), (b, 17580, 334020)]:
                opt = c['resolved_defaults']['optimizer']; lr = opt['config']['learning_rate']['config']
                self.assertEqual((opt['class_name'], opt['config']['weight_decay'], opt['config']['clipnorm']), ('AdamW', .005, 1.5))
                self.assertEqual((lr['warmup_steps'], lr['decay_steps']), (warm, decay))
                self.assertEqual(c['bank_contract']['whitening_rows'], 10000)

    def test_bank_loader_hash_and_rehashed_prefix_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); task, _, parent, history, c = fixture(root)
            with patch.object(t.HestonSpreadPilot, 'prices', return_value=np.arange(63.)+100):
                arrays, _, _ = t.simulate_bank(task, parent, history, c)
            bank = root/'grid-bank'; bank.mkdir(); protocol = root/'bank-protocol.json'; t.write_json(protocol, c)
            t.write_json(bank/'whitening.json', t.NotebookWhitening(arrays['train_observations'][:128]).to_dict())
            r = {'format': 'heston_pr87_grid_bank_v1', 'status': 'complete', 'task': task, 'task_fingerprint': t.fingerprint(task), 'protocol': c, 'parent': parent['identity'], 'protocol_path': str(protocol), 'protocol_sha256': t.digest(protocol), 'data_payload': 'data.npz', 'whitening_sha256': t.digest(bank/'whitening.json')}
            def persist():
                t.write_npz(bank/'data.npz', arrays)
                r.update(data_payload_sha256=t.digest(bank/'data.npz'), data_arrays={k: t.array_summary(v) for k,v in arrays.items()})
                for name in ['bank.json', 'generation.json']: t.write_json(bank/name, r)
            persist()
            with patch.object(t, 'bank_contract', return_value=c), patch.object(t, '_read_parent', return_value=parent), patch.object(t, '_read_historical', return_value=history):
                loaded, _ = t.load_bank(bank); np.testing.assert_array_equal(loaded['train_quotes'], arrays['train_quotes'])
                # Real native63-feature fits/reloads on two bank prefixes. The
                # simulator prices above are software fixtures, not new labels.
                _, keras = t._runtime()
                for count in (128, 135):
                    workflow = t.build_workflow(task)
                    optimizer = workflow.build_optimizer(2, 1, strategy='offline')
                    fit = {'format': 'heston_pr87_grid_training_contract_v1', 'bank_contract': c,
                        'train_rows': count, 'epochs': 2, 'batch_size': 256, 'batches_per_epoch': 1,
                        'expected_optimizer_updates': 2, 'smoke_test_only': True,
                        'checkpoint_epochs': [1, 2], 'max_wall_seconds': 3600,
                        'resolved_defaults': {'optimizer': t._stable_config(keras.optimizers.serialize(optimizer))}}
                    fit_path = root/f'fit-{count}.json'; t.write_json(fit_path, fit)
                    with patch.object(t, 'protocol_contract', return_value=fit):
                        run = t.train(bank, fit_path, root/f'fit-{count}', train_rows=count)
                        self.assertEqual(run['actual_optimizer_updates'], 2)
                        self.assertEqual(run['pricing_calls'], 0)
                        wrapper, _ = t.load_checkpoint(run['primary_checkpoint'])
                        first = t.sample(wrapper, task, arrays['validation_quotes'][:2], 3, seed=73)
                        again, _ = t.load_checkpoint(run['primary_checkpoint'])
                        np.testing.assert_array_equal(first, t.sample(again, task, arrays['validation_quotes'][:2], 3, seed=73))
                        self.assertEqual(first.shape, (2, 3, 6))
                        self.assertEqual((root/f'fit-{count}'/'whitening.json').read_bytes(), (bank/'whitening.json').read_bytes())
                with (bank/'data.npz').open('ab') as f: f.write(b'corrupt')
                with self.assertRaises(ValueError): t.load_bank(bank)
                persist(); arrays['train_kappa'][0, 0] += .01; persist()
                with self.assertRaisesRegex(ValueError, 'truths'): t.load_bank(bank)


if __name__ == '__main__': unittest.main()
