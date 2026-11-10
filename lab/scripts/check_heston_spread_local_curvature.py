"""Local likelihood sanity check, not a substitute for posterior references."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_heston_spread_reference import digest, load_case, write_json


def check(panel_path, case_id, reference_path):
    panel, case, pilot, y = load_case(panel_path, case_id)
    directory = Path(reference_path)
    run = json.loads((directory/"run.json").read_text())
    diagnostic = json.loads((directory/"diagnostics.json").read_text())
    posterior = json.loads((directory/"posterior.json").read_text())
    if (run["task"] != panel["fixed_task"] or run["case_id"] != case_id
            or run["panel_sha256"] != digest(panel_path) or not diagnostic["reference_ready"]):
        raise ValueError("a ready matched fixed-c reference is required")
    payload = Path(run["chain_payload_path"])
    if digest(payload) != diagnostic["chain_payload_sha256"]:
        raise ValueError("reference chain payload changed")
    with np.load(payload, allow_pickle=False) as values:
        chains = values["parameters"]
    center = np.array([posterior["parameter_means"][name] for name in pilot.names])
    empirical_sd = chains.reshape(-1, len(pilot.names)).std(axis=0, ddof=1)
    h = pilot.records["half_spread"]
    if not np.array_equal(pilot.observation_sigma(center), h):
        raise ValueError("fixed c=1 does not produce the declared half-spread SD")
    price = pilot.prices(center)
    manual = -.5*np.sum(((y-price)/h)**2 + np.log(2*np.pi*h*h))
    actual = pilot.log_likelihood(center, observations=y)
    if not np.isclose(actual, manual, atol=1e-9, rtol=0):
        raise ValueError("normalized likelihood differs from direct Gaussian calculation")
    width = pilot.bounds[:, 1]-pilot.bounds[:, 0]
    results = []
    for relative_step in (1e-5, 5e-6):
        steps = np.minimum(relative_step*width,
            .25*np.minimum(center-pilot.bounds[:, 0], pilot.bounds[:, 1]-center))
        if np.any(steps <= 0):
            raise ValueError("curvature center must be inside all prior bounds")
        columns=[]
        for i, step in enumerate(steps):
            shift=np.eye(len(center))[i]*step
            columns.append((pilot.prices(center+shift)-pilot.prices(center-shift))/(2*step))
        whitened=np.column_stack(columns)*width/h[:, None]
        _, singular, right = np.linalg.svd(whitened, full_matrices=False)
        if singular[-1] <= singular[0]*max(whitened.shape)*np.finfo(float).eps:
            raise FloatingPointError("local standardized price Jacobian is numerically rank deficient")
        covariance=(right.T/(singular*singular))@right
        local_sd=width*np.sqrt(np.diag(covariance))
        results.append({"relative_step":relative_step,"physical_steps":steps.tolist(),
            "standardized_jacobian_condition":float(singular[0]/singular[-1]),
            "local_fisher_sd":dict(zip(pilot.names,local_sd.tolist())),
            "local_fisher_sd_over_reference_sd":dict(zip(pilot.names,(local_sd/empirical_sd).tolist()))})
    return {"format":"heston_spread_local_curvature_check_v1","case_id":case_id,
        "panel_sha256":digest(panel_path),"reference_run_sha256":digest(directory/"run.json"),
        "source_sha256":digest(__file__),"center":center.tolist(),
        "direct_gaussian_log_likelihood":float(manual),"pilot_log_likelihood":float(actual),
        "steps":results,"scientific_acceptance":False,
        "scope":"Local linearization and exact Gaussian-scale sanity check. Fisher covariance ignores prior truncation/nonlinearity and cannot establish global posterior or neural correctness."}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel",type=Path,required=True)
    parser.add_argument("--case",required=True)
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists(): raise FileExistsError(args.out)
    value=check(args.panel,args.case,args.reference)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.out,value)
    print(json.dumps(value["steps"],indent=2))


if __name__=="__main__": main()
