"""Reproducible, bounded HKDE numerical validation; no market data or inference.

PYTHONPATH=src python scripts/validate_hkde.py --out /tmp/hkde-validation.json
MC compares 3 strikes jointly, retaining correlated payoff errors. The fine
simulation must meet a declared bias budget plus a sampling allowance; that
check is not a proof of the continuous model's validity or arbitrary-prior
accuracy. Fourier refinement is checked independently of Monte Carlo.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
from scipy.special import roots_legendre

from sbi4finance import gbm, hkde


BASE = dict(kappa=2., theta=.04, vol_of_vol=.3, rho=-.6, jump_intensity=.8,
            jump_up_probability=.35, jump_up_rate=12., jump_down_rate=8.)
PRICE_REFINEMENT_BUDGET = 2e-6  # Units of D*F.
PRICE_MC_BUDGET = 3e-4          # Separate finite-time-step allowance, D*F units.
SE_MULTIPLIER = 5.


def ode_characteristic(u, maturity, parameters, initial_variance, *, steps=4096):
    """Independent complex RK4 solution of the affine-generator Riccati ODE."""
    u = np.asarray(u, dtype=np.complex128)
    p = parameters
    iu = 1j*u
    jump = (p["jump_up_probability"]*p["jump_up_rate"]/(p["jump_up_rate"]-iu)
            + (1-p["jump_up_probability"])*p["jump_down_rate"]/(p["jump_down_rate"]+iu))
    comp = p["jump_intensity"]*(p["jump_up_probability"]/(p["jump_up_rate"]-1)
                                 -(1-p["jump_up_probability"])/(p["jump_down_rate"]+1))
    constant = p["jump_intensity"]*(jump-1)-iu*comp

    def derivative(state):
        c, d = state
        return np.stack((p["kappa"]*p["theta"]*d + constant,
                         .5*p["vol_of_vol"]**2*d*d
                         +(iu*p["rho"]*p["vol_of_vol"]-p["kappa"])*d-.5*(u*u+iu)))

    state = np.zeros((2, u.size), dtype=np.complex128)
    h = maturity/steps
    for _ in range(steps):
        a = derivative(state)
        b = derivative(state+.5*h*a)
        c = derivative(state+.5*h*b)
        d = derivative(state+h*c)
        state += h*(a+2*b+2*c+d)/6
    return np.exp(state[0]+initial_variance*state[1])


def probability_inversion(strikes, maturity, parameters, initial_variance):
    """P1/P2 inversion, distinct payoff transform from the half-shift pricer.

    Uses the same characteristic function, so this checks inversion, not the
    model formula independently. The Riccati and MC checks provide that part.
    """
    roots, weights = roots_legendre(2048)
    u, weights = (roots+1)*400, weights*400
    phase = np.exp(-1j*np.log(strikes)[:, None]*u)
    p2 = .5+np.sum(weights*np.real(phase*hkde.characteristic_function(u, maturity, parameters, initial_variance)/(1j*u)), axis=1)/np.pi
    p1 = .5+np.sum(weights*np.real(phase*hkde.characteristic_function(u-1j, maturity, parameters, initial_variance)/(1j*u)), axis=1)/np.pi
    return p1-strikes*p2


def fixture_checks():
    fixtures = [
        ("central", BASE, .04),
        ("low_variance_high_vol_of_vol", dict(BASE, kappa=1., theta=.025, vol_of_vol=.45,
            rho=-.8, jump_intensity=1.2, jump_up_probability=.2, jump_up_rate=8., jump_down_rate=6.), .025),
        ("higher_variance_weak_leverage", dict(BASE, kappa=3., theta=.065, vol_of_vol=.15,
            rho=-.3, jump_intensity=.1, jump_up_probability=.6, jump_up_rate=20., jump_down_rate=15.), .065),
        ("downward_only", dict(BASE, jump_up_probability=0., jump_intensity=1.2, jump_down_rate=6.), .04),
        ("upward_only", dict(BASE, jump_up_probability=1., jump_intensity=1.2, jump_up_rate=8.), .04),
        ("heston_limit", dict(BASE, jump_intensity=0.), .04),
        ("kou_constant_variance", dict(BASE, vol_of_vol=0.), .04),
        ("black_scholes_limit", dict(BASE, vol_of_vol=0., jump_intensity=0.), .04),
    ]
    strikes, maturities = np.array([.8, 1., 1.2]), (.125, .5, 1.)
    rows = []
    for name, parameters, v0 in fixtures:
        contracts = []
        for maturity in maturities:
            values = [hkde.european_price(1., strikes, 1., maturity, parameters, v0,
                                         nodes=nodes, upper_limit=cutoff)
                      for nodes, cutoff in ((512, 200), (1024, 400), (2048, 800))]
            alternative = probability_inversion(strikes, maturity, parameters, v0)
            delta = max(float(np.max(abs(values[0]-values[1]))), float(np.max(abs(values[1]-values[2]))))
            inversion_error = float(np.max(abs(alternative-values[2])))
            contracts.append(dict(maturity=maturity, strikes=strikes.tolist(),
                                  prices=values[0].tolist(), refined_prices=values[2].tolist(),
                                  cutoff_node_refinement_max_error=delta,
                                  independent_payoff_inversion_max_error=inversion_error,
                                  passed=delta <= PRICE_REFINEMENT_BUDGET and inversion_error <= PRICE_REFINEMENT_BUDGET))
        frequencies = np.array([0, 1, 5, 20, 1-.5j, 20-.5j])
        closed = hkde.characteristic_function(frequencies, 1., parameters, v0)
        ode = [ode_characteristic(frequencies, 1., parameters, v0, steps=n) for n in (2048, 4096)]
        error, refinement = float(np.max(abs(closed-ode[1]))), float(np.max(abs(ode[1]-ode[0])))
        rows.append(dict(name=name, parameters=parameters, initial_variance=v0, contracts=contracts,
                         riccati_reference_max_error=error, riccati_refinement_max_error=refinement,
                         passed=all(row["passed"] for row in contracts) and max(error,refinement) < 1e-8))
    exact = gbm.european_price(1., strikes, 1., .04*.5)
    return rows, float(np.max(abs(exact-hkde.european_price(1., strikes, 1., .5, fixtures[-1][1], .04))))


def monte_carlo_check(parameters, initial_variance, *, paths, seed):
    """Coupled 128/512-step paths at T=.5 with common interval jump totals."""
    dt = np.full(32, .5/32)
    strikes = np.array([.8, 1., 1.2])
    payoffs = {4: [], 16: []}
    terminal_gross = []
    negative = {4: 0, 16: 0}
    for start in range(0, paths, 256):
        ids = range(start, min(start+256, paths))
        noise = hkde.draw_innovations(dt, 16, seed=seed, example_ids=ids, parameters=parameters)
        for level in (4, 16):
            inputs = dict(noise)
            if level == 4:
                inputs["brownian"] = noise["brownian"].reshape(len(ids),32,4,4,2).sum(axis=3)/2
            output = hkde.simulate(parameters, dt, initial_variance, inputs)
            gross = np.exp(output["log_returns"].sum(axis=1)[:,0])
            payoffs[level].append(np.maximum(gross[:,None]-strikes, 0))
            negative[level] += int(np.count_nonzero(output["variance_state"][:,1:] < 0))
            if level == 16:
                terminal_gross.append(gross)
    payoffs = {key: np.concatenate(value) for key,value in payoffs.items()}
    reference = hkde.european_price(1., strikes, 1., .5, parameters, initial_variance,
                                    nodes=2048, upper_limit=800)
    fine = payoffs[16]
    means, errors = fine.mean(axis=0), fine.std(axis=0,ddof=1)/np.sqrt(paths)
    paired = fine-payoffs[4]
    paired_mean, paired_se = paired.mean(axis=0), paired.std(axis=0,ddof=1)/np.sqrt(paths)
    gross = np.concatenate(terminal_gross)
    martingale_mean, martingale_se = gross.mean(), gross.std(ddof=1)/np.sqrt(paths)
    passed = (np.all(abs(means-reference) <= PRICE_MC_BUDGET+SE_MULTIPLIER*errors)
              and np.all(abs(paired_mean) <= PRICE_MC_BUDGET+SE_MULTIPLIER*paired_se)
              and abs(martingale_mean-1) <= SE_MULTIPLIER*martingale_se)
    return dict(parameters=parameters,initial_variance=initial_variance, paths=paths, seed=seed,
                maturity=.5,strikes=strikes.tolist(), diffusion_steps=[128,512],
                fourier_prices=reference.tolist(),mc_prices=means.tolist(),
                mc_standard_errors=errors.tolist(),correlated_mean_covariance=(np.cov(fine,rowvar=False)/paths).tolist(),
                paired_refinement_mean=paired_mean.tolist(),paired_refinement_se=paired_se.tolist(),
                terminal_gross_mean=float(martingale_mean),terminal_gross_mcse=float(martingale_se),
                negative_observation_state_fraction={str(key):value/(paths*32) for key,value in negative.items()},
                passed=bool(passed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--paths", type=int, default=16384)
    args = parser.parse_args()
    if args.paths < 1024:
        parser.error("paths must be >=1024 for this bounded pilot")
    if args.out.exists():
        parser.error("output exists; choose a new path")
    started = perf_counter()
    fixtures, black_scholes_error = fixture_checks()
    mc = [monte_carlo_check(row["parameters"],row["initial_variance"],paths=args.paths,seed=60300+i)
          for i,row in enumerate(fixtures[:2])]
    report = dict(model_id=hkde.MODEL_ID,data_origin="synthetic",measure="Q",evidence_scope="bounded numerical pilot; no neural or empirical validation",
                  price_units="discount * forward (both 1 in fixtures)",
                  fixed_budgets=dict(fourier_refinement=PRICE_REFINEMENT_BUDGET,mc_discretization=PRICE_MC_BUDGET,
                                     mc_standard_error_multiplier=SE_MULTIPLIER,riccati_reference=1e-8),
                  quadrature_controls=[dict(nodes=n,upper_limit=u) for n,u in ((512,200),(1024,400),(2048,800))],
                  pricing_fixtures=fixtures,black_scholes_limit_max_error=black_scholes_error,
                  monte_carlo=mc,passed=all(row["passed"] for row in fixtures+mc),
                  seconds=perf_counter()-started,python=platform.python_version(),numpy=np.__version__,
                  source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in (Path("src/sbi4finance/hkde.py"),Path(__file__).relative_to(Path.cwd()))},
                  limits=["Fixed fixtures do not certify every prior draw; bank checks each panel's quadrature refinement.",
                          "MC sampling error, diffusion bias and Fourier truncation are separate quantities.",
                          "Price-path jumps are exact only at observation boundaries; barrier and American pricing are absent.",
                          "The Q continuous model differs from the full-truncation discrete path scheme."])
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(dict(passed=report["passed"],seconds=report["seconds"],output=str(args.out)),indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
