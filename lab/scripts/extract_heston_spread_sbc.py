"""Reconstruct four existing development SBC populations, without fitting.

The original observation/sampling/rank seeds and validation rows are fixed.
Every rank statistic, coverage interval and support count must equal its saved
evaluation exactly. Full draws remain in memory; only ranks and small quantile
summaries are written. The CLI enforces a 300-second whole-worker wall limit.
"""
import argparse
import hashlib
from importlib import metadata as package_metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

PROCESS_STARTED = time.monotonic()
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from scripts import train_heston_spread_baseline as trainer
from scripts import evaluate_heston_spread_baseline as evaluator
from sbi4finance.calibration import validation


SEED, ROWS, DRAWS, CHUNK = 261112, 512, 512, 16
STAGES = (
    ("default-010000", "Dropout .05 at 10k", "evaluation-update-010000-provisional.json", 10000),
    ("zero-010000", "Dropout 0 at 10k", "evaluation-dropout-zero-update-010000-provisional.json", 10000),
    ("zero-050000", "Dropout 0 at 50k", "evaluation-dropout-zero-update-050000.json", 50000),
    ("annealed-060000", "Annealed at 60k", "annealing/evaluation-060000.json", 60000),
)
FROZEN_TRAINER_SHA256 = "3a8747332a3493adcd9298d701d0d6aab087dada35d8648827d7790330931f38"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False)+"\n")
    temporary.replace(path)


def evaluation_binding(evaluation, record, checkpoint, expected_updates):
    population = evaluation["population"]
    if (record["actual_optimizer_updates"] != expected_updates
            or trainer.fingerprint(record["task"]) != record["task_fingerprint"]
            or evaluation["task"] != record["task"]["target"]
            or evaluation["seed"] != SEED or population["rows"] != ROWS
            or population["draws_per_case"] != DRAWS
            or record["task"]["target"]["infer_c"] or record["task"]["target"]["fixed_c"] != 1.
            or trainer.digest(checkpoint/"model.keras") != record["model_sha256"]
            or trainer.digest(checkpoint/"state.npz") != record["state_sha256"]):
        raise ValueError("evaluation/checkpoint task, population protocol or payload differs")
    for field, expected in (("model_sha256",record["model_sha256"]), ("state_sha256",record["state_sha256"]),
                            ("checkpoint_record_sha256",trainer.digest(checkpoint/"checkpoint.json")),
                            ("checkpoint_source_sha256",record["source_sha256"])):
        if field in evaluation and evaluation[field] != expected:
            raise ValueError("explicit evaluation provenance differs: "+field)


def load_validation(bank, record, *, rows=ROWS):
    """Hash complete bytes; open exactly the two original validation members."""
    bank = Path(bank)
    bank_record = read(bank/"bank.json")
    payload = bank/bank_record["payload"]
    if (payload.resolve().parent != bank.resolve()
            or trainer.digest(bank/"bank.json") != record["bank_record_sha256"]
            or bank_record["identity"] != record["bank_identity"]
            or bank_record["payload_sha256"] != record["bank_payload_sha256"]
            or trainer.digest(payload) != record["bank_payload_sha256"]):
        raise ValueError("SBC bank manifest/identity/payload differs")
    with np.load(payload,allow_pickle=False) as archive:
        truth = archive["validation_parameters"][:rows].copy()
        clean = archive["validation_clean"][:rows].copy()
    target = record["task"]["target"]
    evaluator.validate_development_arrays(truth,clean,np.asarray(target["prior_bounds"]),rows,target["quote_count"])
    return truth,clean


def exact_population(draws, truth, evaluation):
    """Retain unrounded ranks only after exact original-metric reproduction."""
    ranks = validation.sbc(draws,truth,seed=SEED+2000)
    fractional = ranks.pop("fractional_ranks")
    coverage = validation.coverage(draws,truth)
    support = evaluator.support_report(draws,np.asarray(evaluation["task"]["prior_bounds"]))
    original = evaluation["population"]
    if ranks != original["rank_dkw"] or coverage != original["coverage"] or support != original["support"]:
        raise ValueError("reconstructed SBC/support/coverage does not exactly match the saved evaluation")
    return fractional,ranks,coverage,support


def extract(evidence, out, *, wall_seconds=300):
    started = time.monotonic()
    evidence,out = Path(evidence),Path(out)
    lab = Path(__file__).resolve().parents[1]
    if not 0 < wall_seconds <= 300:
        raise ValueError("whole extraction wall limit must be positive and at most 300 seconds")
    if out.exists():
        raise FileExistsError(out)
    sources = {"extractor":Path(__file__),"sampling_helper":Path(trainer.__file__),
               "evaluator":Path(evaluator.__file__),"validation":Path(validation.__file__)}
    source_hashes = {name:{"path":str(path.resolve()),"sha256":trainer.digest(path)} for name,path in sources.items()}
    if source_hashes["sampling_helper"]["sha256"] != FROZEN_TRAINER_SHA256:
        raise ValueError("frozen sampling-helper source differs")
    prepared = []
    for stage,label,relative,updates in STAGES:
        evaluation_path = evidence/relative
        evaluation = read(evaluation_path)
        checkpoint = Path(evaluation["checkpoint"])
        if not checkpoint.is_absolute():
            checkpoint = (lab/checkpoint).resolve()
        record = read(checkpoint/"checkpoint.json")
        evaluation_binding(evaluation,record,checkpoint,updates)
        prepared.append((stage,label,evaluation_path,evaluation,checkpoint,record))
    first = prepared[0][-1]
    if any(record["task_fingerprint"] != first["task_fingerprint"]
           or record["bank_record_sha256"] != first["bank_record_sha256"]
           or record["bank_payload_sha256"] != first["bank_payload_sha256"]
           for *_,record in prepared):
        raise ValueError("four SBC stages do not share the same task and physical bank")
    truth,clean = load_validation(first["bank_path"],first)
    rng = np.random.default_rng(np.random.SeedSequence([0x53504556,SEED]))
    noisy = clean + np.asarray(first["task"]["records"]["half_spread"])*rng.standard_normal(clean.shape)
    row_ids = np.array([f"validation:{index:08d}" for index in range(ROWS)])
    out.mkdir(parents=True,exist_ok=False)
    trainer.write_npz(out/"population.npz",{"truth":truth,"row_id":row_ids})
    common = {"file":"population.npz","sha256":trainer.digest(out/"population.npz"),
              "truth":trainer.array_summary(truth),"clean":trainer.array_summary(clean),
              "observations":trainer.array_summary(noisy),"row_count":ROWS,
              "row_id_protocol":"derived validation:{index:08d}, first512; row-ID array never opened",
              "row_id_sha256":hashlib.sha256(("\n".join(row_ids)+"\n").encode()).hexdigest()}
    completed = []
    try:
        os.environ.setdefault("KERAS_BACKEND","jax")
        import jax
        import keras
        versions = {"python":platform.python_version(),**{name:package_metadata.version(name) for name in ("bayesflow","keras","jax","numpy","scipy")}}
        for stage,label,path,evaluation,checkpoint,record in prepared:
            if time.monotonic()-started >= wall_seconds:
                raise TimeoutError("controlled SBC extraction wall budget exhausted")
            workflow,loaded = trainer.load_checkpoint(checkpoint)
            if loaded != record or versions != record["versions"]:
                raise ValueError("checkpoint bytes or exact numerical versions changed")
            if (bool(jax.config.x64_enabled) != record["numeric_configuration"]["jax_enable_x64"]
                    or keras.backend.floatx() != record["numeric_configuration"]["keras_floatx"]):
                raise ValueError("checkpoint numerical configuration differs")
            chunks = []
            for start in range(0,ROWS,CHUNK):
                if time.monotonic()-started >= wall_seconds:
                    raise TimeoutError("controlled SBC extraction wall budget exhausted")
                chunks.append(trainer.sample(workflow,record["task"],noisy[start:start+CHUNK],DRAWS,seed=SEED+1000+start))
            draws = np.concatenate(chunks)
            fractional,ranks,coverage,support = exact_population(draws,truth,evaluation)
            quantiles = np.quantile(draws,[.05,.5,.95],axis=1)
            compact_quantiles = quantiles.astype(np.float32)
            qfile = stage+".quantiles.npz"
            trainer.write_npz(out/qfile,{"quantiles":compact_quantiles})
            provenance = {"evaluation_path":str(path.resolve()),"evaluation_sha256":trainer.digest(path),
                "evaluation_producer_sha256":evaluation["source_sha256"],"checkpoint_path":str(checkpoint),
                "checkpoint_record_sha256":trainer.digest(checkpoint/"checkpoint.json"),
                "model_sha256":record["model_sha256"],"state_sha256":record["state_sha256"],
                "checkpoint_source_sha256":record["source_sha256"],"task_fingerprint":record["task_fingerprint"],
                "design_sha256":record["design_sha256"],"bank_record_sha256":record["bank_record_sha256"],
                "bank_payload_sha256":record["bank_payload_sha256"],"source_hashes":source_hashes,
                "legacy_evaluation_payload_hashes_present":"model_sha256" in evaluation}
            result = {"format":"heston_spread_sbc_stage_v1","stage_id":stage,"label":label,
                "parameter_names":record["names"],"rows":ROWS,"draws_per_case":DRAWS,"seed":SEED,
                "fractional_ranks":fractional,"rank_dkw":ranks,"coverage":coverage,"support":support,
                "posterior_quantiles":{"file":qfile,"sha256":trainer.digest(out/qfile),"levels":[.05,.5,.95],
                    "shape":list(compact_quantiles.shape),"dtype":"float32","scope":"visual recovery summary only; all validation metrics computed before rounding",
                    "max_abs_float32_rounding":float(np.max(np.abs(quantiles-compact_quantiles)))},
                "population":common,"provenance":provenance,"metrics_exact_match":True,
                "reserved_arrays_opened":False,"scientific_acceptance":False,
                "scope":"same original prior-predictive development population; no conditional or market acceptance"}
            write(out/(stage+".json"),result)
            completed.append({"id":stage,"file":stage+".json","sha256":trainer.digest(out/(stage+".json")),
                              "quantiles_file":qfile,"quantiles_sha256":trainer.digest(out/qfile)})
            print(json.dumps({"stage":stage,"metrics_exact_match":True,"rank_passed":ranks["passed"],
                              "seconds":time.monotonic()-started}),flush=True)
            del workflow,draws,chunks
        if any(trainer.digest(path) != source_hashes[name]["sha256"] for name,path in sources.items()):
            raise ValueError("pinned extraction source changed during inference")
        manifest = {"format":"heston_spread_sbc_manifest_v1","stages":completed,"parameter_names":first["names"],
            "population":common,"source_hashes":source_hashes,"versions":versions,
            "protocol":{"seed":SEED,"rows":ROWS,"draws_per_case":DRAWS,"chunk_rows":CHUNK,
                "noise":"PCG64 SeedSequence([0x53504556,261112]); one full512x54 Gaussian draw",
                "sampling_seeds":"261112+1000+start, start=0,16,...,496","rank_seed":SEED+2000,
                "inference_population_passes":4,"full_draws_written":False,"maximum_wall_seconds":wall_seconds},
            "all_metrics_exact_match":True,"reserved_arrays_opened":False,"scientific_acceptance":False,
            "seconds":time.monotonic()-started}
        write(out/"manifest.json",manifest)
        return manifest
    except BaseException as error:
        write(out/"failure.json",{"error_type":type(error).__name__,"error":str(error),"completed_stages":completed,
            "seconds":time.monotonic()-started,"automatic_retry":False,"reserved_arrays_opened":False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--wall-seconds",type=float,default=300)
    parser.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 0 < args.wall_seconds <= 300:
        parser.error("wall-seconds must be positive and at most300")
    if not args.worker:
        if args.out.exists():
            raise FileExistsError(args.out)
        process = subprocess.Popen([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],"--worker"])
        try:
            code = process.wait(timeout=max(.001,args.wall_seconds-(time.monotonic()-PROCESS_STARTED)))
        except subprocess.TimeoutExpired:
            process.kill(); process.wait()
            args.out.mkdir(parents=True,exist_ok=True)
            write(args.out/"hard-timeout.json",{"status":"killed_at_300s_whole_extraction_limit","automatic_retry":False})
            raise TimeoutError("whole SBC extraction exceeded its hard wall limit")
        if code:
            raise SystemExit(code)
    else:
        result = extract(args.evidence,args.out,wall_seconds=args.wall_seconds)
        print(json.dumps({"out":str(args.out),"all_metrics_exact_match":True,"seconds":result["seconds"]}))


if __name__ == "__main__":
    main()
