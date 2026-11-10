"""Exploratory training-only covariance diagnostic; no neural/model queries."""
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(array):
    array=np.ascontiguousarray(array)
    return {"shape":list(array.shape),"dtype":str(array.dtype),"sha256":hashlib.sha256(array.tobytes()).hexdigest()}


root=Path(sys.argv[1]).resolve()
output=root/"lab/evidence/pilot-results/2026-11-09/heston-spread-baseline/noise-scale/capacity-study/diagnostics/input-conditioning.json"
if output.exists():
    raise FileExistsError(output)
protocol_path=root/"lab/evidence/pilot-results/2026-11-09/heston-spread-baseline/noise-scale/capacity-study/protocol.json"
protocol=json.loads(protocol_path.read_text())
contract=protocol["arms"][0]["contract"]
parent_path=Path(protocol["parent_checkpoint"]["record"]["path"])
assert sha(parent_path)==protocol["parent_checkpoint"]["record"]["sha256"]
parent=json.loads(parent_path.read_text())
bank=Path(contract["bank_path"])
assert sha(bank/"bank.json")==contract["bank_record_sha256"]
bank_record=json.loads((bank/"bank.json").read_text())
payload=bank/bank_record["payload"]
assert sha(payload)==contract["bank_payload_sha256"]
assert bank_record["identity"]["train_rows"]==100000
with np.load(payload,allow_pickle=False) as archive:
    clean=np.asarray(archive["train_clean"],dtype=np.float64).copy()
assert clean.shape==(100000,54) and np.isfinite(clean).all()
h=np.asarray(parent["task"]["records"]["half_spread"],dtype=np.float64)
assert h.shape==(54,) and np.all(h>0)
lo,hi=parent["task"]["target"]["c_bounds"]
expected_c2=(lo*lo+lo*hi+hi*hi)/3.
assert expected_c2==1.75
x=clean/h
feature_mean=x.mean(axis=0)
feature_sd=x.std(axis=0,ddof=1)
standardized=(x-feature_mean)/feature_sd
C=standardized.T@standardized/(len(x)-1)
eigen,U=np.linalg.eigh(C)
order=np.argsort(eigen)[::-1]
eigen,U=eigen[order],U[:,order]
noise_diagonal=expected_c2/feature_sd**2
projected_noise=np.sum(U**2*noise_diagonal[:,None],axis=0)
total_eigen=np.linalg.eigvalsh(C+np.diag(noise_diagonal))
rounding=(x.astype(np.float32).astype(np.float64)-x)/feature_sd
rounding-=rounding.mean(axis=0)
rounding_cov=rounding.T@rounding/(len(x)-1)
pc_rounding=np.einsum("ij,ik,kj->j",U,rounding_cov,U)
ratio=np.sqrt(np.maximum(pc_rounding,0)/projected_noise)
raw_rounding_ratio=np.abs(x.astype(np.float32).astype(np.float64)-x)/np.sqrt(expected_c2)
assert eigen[-1]>0 and total_eigen[0]>0 and np.isfinite(ratio).all()
rows=[{"pc":int(i+1),"clean_variance":float(eigen[i]),"expected_noise_variance":float(projected_noise[i]),
       "noise_to_clean_variance":float(projected_noise[i]/eigen[i]),
       "noise_fraction_of_total_projected_variance":float(projected_noise[i]/(eigen[i]+projected_noise[i])),
       "clean_float32_cast_error_sd_over_expected_noise_sd":float(ratio[i])} for i in range(54)]
result={"format":"heston_input_conditioning_exploratory_v1","scope":"exploratory training-only linear covariance geometry; no causal or performance conclusion",
 "source":{"path":str(Path(__file__).resolve()),"sha256":sha(__file__),
           "git_revision":subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip(),
           "python":platform.python_version(),"numpy":np.__version__},
 "bindings":{"protocol_path":str(protocol_path),"protocol_sha256":sha(protocol_path),
             "parent_checkpoint_path":str(parent_path),"parent_checkpoint_sha256":sha(parent_path),
             "task_fingerprint":parent["task_fingerprint"],"bank_record_sha256":sha(bank/"bank.json"),
             "bank_payload_sha256":sha(payload),"train_clean":identity(clean),"half_spread":identity(h)},
 "data_access":{"named_arrays_opened":["train_clean"],"train_rows":100000,"validation_opened":False,
                "reserved_opened":False,"neural_model_loaded":False,"posterior_queries":0,"fitting":False},
 "method":{"input":"x=P_clean/half_spread","standardization":"training-only sample mean and per-feature SD, ddof=1",
           "clean_covariance":"Cov((x-mean)/sd)","expected_noise_covariance":"E[c^2]*diag(1/sd^2)",
           "c_prior":[lo,hi],"expected_c_squared":expected_c2,
           "pcs":"descending eigenvectors of clean covariance; noise projected as u.T N u",
           "rounding":"cast clean x to float32, then back to float64; covariance in same standardized coordinates"},
 "summary":{"clean_covariance_condition_number":float(eigen[0]/eigen[-1]),
            "expected_noisy_covariance_condition_number":float(total_eigen[-1]/total_eigen[0]),
            "expected_noisy_covariance_min_eigenvalue":float(total_eigen[0]),
            "expected_noisy_covariance_max_eigenvalue":float(total_eigen[-1]),
            "top3_clean_variance_fraction":float(eigen[:3].sum()/eigen.sum()),
            "top5_clean_variance_fraction":float(eigen[:5].sum()/eigen.sum()),
            "expected_noise_trace_over_clean_trace":float(noise_diagonal.sum()/np.trace(C)),
            "noise_dominant_clean_PCs":int(np.count_nonzero(projected_noise>eigen)),
            "feature_expected_noise_sd_quantiles_10_50_90":np.quantile(np.sqrt(noise_diagonal),[.1,.5,.9]).tolist(),
            "pc_rounding_sd_over_noise_sd_min_median_max":[float(ratio.min()),float(np.median(ratio)),float(ratio.max())],
            "raw_input_cast_absolute_error_over_noise_sd_quantiles_50_90_99_max":np.quantile(raw_rounding_ratio,[.5,.9,.99,1.]).tolist(),
            "hypothetical_expected_total_covariance_whitening_scale_min_max":[float(1/np.sqrt(total_eigen[-1])),float(1/np.sqrt(total_eigen[0]))]},
 "principal_components":rows,
 "feature_sd_in_halfspread_units":feature_sd.tolist(),
 "expected_standardized_noise_variance_diagonal":noise_diagonal.tolist(),
 "matrix_identities":{"clean_covariance":identity(C),"expected_noise_diagonal":identity(noise_diagonal),"clean_pcs":identity(U)},
 "limitations":["The physical-parameter prior and fixed design determine this unconditional covariance; local posterior geometry can differ.",
                "Noise projected into clean PCs is generally correlated; projected diagonal variances are not a diagonalized likelihood.",
                "Clean-only whitening would amplify an almost singular covariance. Expected total covariance includes the specified noise and is better conditioned, but no whitener was fitted or applied.",
                "Float32 figures measure only the clean input cast, not normalization, deep-network arithmetic or likelihood accuracy.",
                "This shows a possible optimization conditioning issue, not proof of the cause of c failure or that whitening improves training."]}
output.parent.mkdir(parents=True,exist_ok=True)
output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
print(json.dumps(result["summary"],indent=2))
