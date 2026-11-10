"""Replay diagnostics keep the original target, population and schedule."""
import copy
from pathlib import Path
import tempfile
import unittest

from scripts import evaluate_heston_spread_whitening_replay as evaluation
from scripts import render_heston_spread_whitening_replay as report
from test_heston_spread_training import fixture


class ReplayEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        design, _, _, _ = fixture(self.root)
        self.task, _ = evaluation.frozen.scale.make_task(design)
        self.target = self.task["target"]
        self.study = {"format":"heston_spread_whitening_study_v1", "joint_task":self.target,
            "evaluation":evaluation.original.evaluation_contract(self.target),
            "arms":[{"mode":"diagonal"},{"mode":"zca"}], "sources":{}}
        for key in ("original_study","reference_recovery","baseline_evaluation","matched_mcmc_report"):
            path=self.root/(key+".json");evaluation.fixed.write_json(path,{})
            self.study[key]=evaluation.binding(path)
        self.study_path=self.root/"study.json";evaluation.fixed.write_json(self.study_path,self.study)
        self.checkpoint=self.root/"checkpoint";self.checkpoint.mkdir()
        evaluation.fixed.write_json(self.checkpoint/"preprocessing.json",{"loader":"validates actual transform"})
        self.contract={"planned_updates":4000,"schedule_updates":8000,
            "checkpoint_updates":[2000,3000,3320,4000],"train_rows":10000,"batch_size":256,
            "standardize":["inference_variables"],
            "architecture":{"family":"spline","depth":6,"widths":[128,128],"dropout":0.}}
        self.replay={"format":"heston_spread_whitening_checkpoint_replay_study_v1",
            "contract":self.contract,"original_study":evaluation.binding(self.study_path),
            "evaluation":evaluation.evaluation_contract(self.target),
            "sources":{"evaluator":evaluation.binding(evaluation.__file__)}}
        self.replay_path=self.root/"replay.json";evaluation.fixed.write_json(self.replay_path,self.replay)
        self.metadata={"task":self.task,"format":"heston_spread_whitening_checkpoint_replay_checkpoint_v1",
            "mode":"zca","training_protocol":self.contract,"actual_optimizer_updates":3320,
            "protocol_path":str(self.replay_path),"protocol_sha256":evaluation.fixed.digest(self.replay_path),
            "reserved_arrays_opened":False,"preprocessing_sha256":evaluation.fixed.digest(self.checkpoint/"preprocessing.json")}

    def validate(self, metadata=None, replay=None):
        return evaluation.validate_checkpoint(metadata or self.metadata,self.study,self.study_path,
            self.checkpoint,replay or self.replay,self.replay_path)

    def test_population_and_draw_contract_is_paired_with_original_evaluation(self):
        contract=evaluation.evaluation_contract(self.target);old=evaluation.original.evaluation_contract(self.target)
        for key in ("seed","population_rows","posterior_draws_per_population_case","conditional_cases","conditional_draws","population_c","population_gaussian"):
            self.assertEqual(contract[key],old[key])
        self.assertEqual(contract["checkpoint_updates"],[2000,3000,3320,4000])
        self.assertFalse(contract["scientific_acceptance"])
        self.assertIn("posthoc",contract["checkpoint_selection"])
        self.assertNotIn("family_checks",contract)

    def test_endpoint_validation_rejects_shortened_schedule_or_final_masquerade(self):
        self.validate()
        for step in (2000,3000,3320,4000): self.validate({**self.metadata,"actual_optimizer_updates":step})
        with self.assertRaisesRegex(ValueError,"replay checkpoint"):
            self.validate({**self.metadata,"actual_optimizer_updates":8000})
        changed=copy.deepcopy(self.replay);changed["contract"]["schedule_updates"]=4000
        with self.assertRaisesRegex(ValueError,"replay checkpoint"):
            self.validate(replay=changed)

    def test_original_study_and_executed_evaluator_are_source_bound(self):
        changed=copy.deepcopy(self.replay);changed["original_study"]["sha256"]="wrong"
        with self.assertRaisesRegex(ValueError,"original whitening study binding"):
            self.validate(replay=changed)
        changed=copy.deepcopy(self.replay);changed["sources"]={}
        with self.assertRaisesRegex(ValueError,"evaluator was not frozen"):
            self.validate(replay=changed)

    def test_report_cannot_describe_an_inexact_prefix_as_exact(self):
        payload=self.root/"training";payload.mkdir()
        evaluation.fixed.write_json(payload/"training.json",{"status":"complete","completed_updates":4000,
            "completed_bank_passes":100,"presentations":1000000,"schedule_updates":8000,
            "all_original_initial_state_hashes_exact":True,"original_learning_curve_prefix_exact":False,
            "original_curve_points_compared":100,"same_seed_reload_exact":True})
        path=self.root/"report-protocol.json"
        evaluation.fixed.write_json(path,{"format":"heston_spread_whitening_checkpoint_replay_study_v1","output":str(payload)})
        with self.assertRaisesRegex(ValueError,"initialization, loss prefix"):
            report.collect(path)


if __name__=="__main__":unittest.main()
