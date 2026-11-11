"""Reference failures are independent of bank/fits in the new recovery only."""
import unittest
from unittest.mock import patch

from scripts import run_heston_pr87_grid_recovery as recovery
import test_heston_pr87_grid_campaign as fixtures


class GridRecoveryIsolationTests(unittest.TestCase):
    def setUp(self):
        # Reuse the established fake subprocess/output fixtures; never execute
        # the original tests or change the original controller/test source.
        module = patch.object(fixtures, 'controller', recovery)
        module.start(); self.addCleanup(module.stop)
        self.h = fixtures.GridCampaignTests()
        self.h.setUp(); self.addCleanup(self.h.doCleanups)
        self.h.reference['settings'].update(inner_cpu_and_wall_seconds=2400, outer_total_wall_seconds=2450)
        reference_path = self.h.campaign['reference_protocol']['path']
        self.h.dump(reference_path, self.h.reference)
        self.h.campaign['reference_protocol'] = recovery.identity(reference_path)
        for name, rows in recovery.RUNS:
            self.h.studies[name]['reference_protocol'] = self.h.campaign['reference_protocol']
            self.h.persist_study(name, rows)
        aborted = self.h.root/'aborted.json'
        self.h.dump(aborted, {'format': 'heston_pr87_grid_campaign_v1', 'status': 'failed'})
        self.h.campaign.update(recovery_of=recovery.identity(aborted), operational_recovery={'scientific_changes': 'None'})
        self.h.dump(self.h.protocol, self.h.campaign)

    def test_nonzero_reference_exit_during_bank_does_not_kill_or_retry_bank_fits(self):
        worker, calls, events, workers = self.h.fake_workers(failure='synthetic-market-like')
        class DelayedBank(worker):
            def poll(self):
                if self.kind == 'generate' and self.code is None and self.polls < 20:
                    self.polls += 1
                    return None
                return super().poll()
        with patch.object(recovery.subprocess, 'Popen', DelayedBank), patch.object(recovery.time, 'sleep'), patch.object(recovery.shared, 'stop') as stop:
            result = recovery.run(self.h.protocol, self.h.root/'campaign.json')
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(len(calls), 10)
            self.assertEqual(list(result['runs']), ['200k-300', '300k-300'])
            stop.assert_not_called()
            with self.assertRaisesRegex(FileExistsError, 'no retries'):
                recovery.run(self.h.protocol, self.h.root/'campaign.json')
            self.assertEqual(len(calls), 10)
        failed_at = next(i for i, e in enumerate(events) if e == ('end', 'synthetic-market-like', 1))
        bank_at = next(i for i, e in enumerate(events) if e == ('end', 'generate', 0))
        self.assertLess(failed_at, bank_at)
        failed = result['references']['synthetic-market-like']
        self.assertFalse(failed['reference_ready']); self.assertIn('inconclusive', failed['scope'])
        self.assertIsNone(failed['run'])
        ref_job = next(j for j in result['jobs'] if j['id'] == 'reference_synthetic-market-like')
        self.assertEqual(ref_job['status'], 'failed_inconclusive')
        self.assertTrue(all(j['status'] == 'complete' for j in result['jobs'] if j['kind'] != 'reference'))
        self.assertEqual(sum(w.label == 'synthetic-market-like' for w in workers), 1)
        self.assertFalse(result['scientific_acceptance'])

    def test_outer_reference_cap_terminates_only_that_reference_while_bank_finishes(self):
        worker, calls, events, workers = self.h.fake_workers()
        now, stopped = [0.], []
        class TimedReference(worker):
            def poll(self):
                if self.code is not None:
                    return self.code
                if self.kind == 'reference' and self.label == 'synthetic-market-like':
                    self.polls += 1
                    if self.polls >= 3:
                        now[0] = 2451.  # Reference cap2450; live bank cap3600.
                    return None
                if self.kind == 'reference':
                    self.polls = 11  # Other references complete before the jump.
                if self.kind == 'generate' and self.polls < 20:
                    self.polls += 1
                    return None
                return super().poll()
        def stop(process):
            stopped.append((process.kind, process.label))
            process.code = -15
            events.append(('stop', process.label, -15))
        with patch.object(recovery.subprocess, 'Popen', TimedReference), patch.object(recovery.time, 'sleep'), patch.object(
                recovery.time, 'monotonic', side_effect=lambda: now[0]), patch.object(recovery.shared, 'stop', side_effect=stop):
            result = recovery.run(self.h.protocol, self.h.root/'campaign.json')
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(stopped, [('reference', 'synthetic-market-like')])
        self.assertEqual(len(calls), 10)
        self.assertEqual(list(result['runs']), ['200k-300', '300k-300'])
        self.assertTrue(all(j['status'] == 'complete' for j in result['jobs'] if j['kind'] != 'reference'))
        failed = next(j for j in result['jobs'] if j['id'] == 'reference_synthetic-market-like')
        self.assertEqual(failed['status'], 'failed_inconclusive')
        self.assertEqual(failed['wall_seconds_limit'], 2450)
        self.assertIn('outer wall cap', failed['error'])
        self.assertFalse(result['references']['synthetic-market-like']['reference_ready'])
        self.assertTrue(result['references']['synthetic-broad']['reference_ready'])
        self.assertTrue(result['references']['market']['reference_ready'])
        self.assertEqual(sum(w.label == 'synthetic-market-like' for w in workers), 1)
        stop_index = next(i for i, event in enumerate(events) if event[:2] == ('stop', 'synthetic-market-like'))
        bank_index = next(i for i, event in enumerate(events) if event == ('end', 'generate', 0))
        self.assertLess(stop_index, bank_index)


if __name__ == '__main__': unittest.main()
