"""Recovery changes resource bounds and wrapper identity only, with no queries."""
import ast
from pathlib import Path
import unittest

from scripts import run_heston_pr87_grid_reference as original
from scripts import run_heston_pr87_grid_reference_recovery as recovery
from scripts import evaluate_heston_pr87_grid as original_evaluation
from scripts import evaluate_heston_pr87_grid_recovery as recovery_evaluation


class GridReferenceRecoveryTests(unittest.TestCase):
    def test_reference_source_diff_is_only_documentation_and_two_caps(self):
        before = ast.parse(Path(original.__file__).read_text())
        after = ast.parse(Path(recovery.__file__).read_text())
        before.body = before.body[1:]; after.body = after.body[1:]
        for tree in (before,after):
            assignment = next(node for node in tree.body if isinstance(node,ast.Assign)
                and isinstance(node.targets[0],ast.Tuple) and [item.id for item in node.targets[0].elts] == ['SEED','INNER_SECONDS','OUTER_SECONDS'])
            assignment.value = ast.Constant(value='verified resource assignment')
        self.assertEqual(ast.dump(before),ast.dump(after))
        self.assertEqual((original.SEED,original.INNER_SECONDS,original.OUTER_SECONDS),(261112,900,950))
        self.assertEqual((recovery.SEED,recovery.INNER_SECONDS,recovery.OUTER_SECONDS),(261112,2400,2450))

    def test_settings_keep_sampler_allocations_gates_and_smoke_identical(self):
        before,after = original.settings_contract(),recovery.settings_contract()
        self.assertEqual(after.pop('inner_cpu_and_wall_seconds'),2400)
        self.assertEqual(after.pop('outer_total_wall_seconds'),2450)
        self.assertEqual(before.pop('inner_cpu_and_wall_seconds'),900)
        self.assertEqual(before.pop('outer_total_wall_seconds'),950)
        self.assertEqual(before,after)
        self.assertEqual(original.settings_contract(smoke_test_only=True),recovery.settings_contract(smoke_test_only=True))
        self.assertEqual(original.CASES,recovery.CASES)

    def test_evaluator_source_changes_only_reference_import(self):
        expected = Path(original_evaluation.__file__).read_text().replace(
            'from scripts import run_heston_pr87_grid_reference as reference',
            'from scripts import run_heston_pr87_grid_reference_recovery as reference')
        self.assertEqual(expected,Path(recovery_evaluation.__file__).read_text())
        self.assertIs(recovery_evaluation.reference,recovery)
        self.assertEqual(recovery.source_hashes()['grid_reference']['path'],str(Path(recovery.__file__).resolve()))


if __name__ == '__main__':
    unittest.main()
