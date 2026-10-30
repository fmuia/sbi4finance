"""Early exercise in SPY's surface (study S8): how far reading American quotes as European moves the design grid.

    python checks/spy_early_exercise/spy_early_exercise.py chain RAW_CHAIN.json ASOF OUT.json
    python checks/spy_early_exercise/spy_early_exercise.py surface ASOF OUT.json

Study S8 inverts SPY's out-of-the-money mids with the Black formula, with the discount from Treasury yields and the
forward from parity on the six pairs nearest the money (Lecture 5, Section 5.4). SPY's options are American, so the
inversion reads early-exercise value as volatility, and parity reads it as carry. This check takes it out and reads
the grid again.

The premium. A quote's early-exercise value is its American less its European value on one CRR tree with escrowed
dividends: the tree's spot is F D, the spot net of the dividends to expiry, and exercise is against it plus the
value of the dividends still to come. It is the average over STEPS and STEPS + 1 steps, which damps the tree's
odd-even wobble. The volatility is the quote's own American implied volatility, the fixed point
sigma = Black^-1(mid - premium(sigma)). Dividends are SPY's quarterly schedule in settings.json. Times and the rate
are on S8's clock: weekdays / 252, and the rate curve's r(T) throughout the tree.

'chain' needs a raw Cboe file (data/SOURCES.md). The forward is parity on the de-Americanized pairs, iterated with
the premia; bid and ask each lose the mid's premium; chain.build_surface then reads the result exactly as S8 read
the raw chain. The output gives the shift of the 63 design points, of the forward-variance levels and of the
forwards, and the premia of the out-of-the-money puts in half-spreads, as the EWG check does
(checks/thin_chains/). It also runs the 'surface' estimate on the European grid it built, to test that estimate.

'surface' needs only S8's committed surface.json, for a date whose raw file is not at hand. For each design point
and each expiration S8 read there, the American quote is rebuilt from the grid's volatility, S8's forward and
discount and the grid's slope in k, and inverted again at the de-Americanized forward, which comes from the
at-the-money pair. It assumes the two expirations share the grid's volatility at the point's log-moneyness, and it
does not smooth across the money as S8's splines do. On the chain of 29 October it overstates the at-the-money
shift 1.4 to 1.7 times from 21 days on (0.86 against 0.60 points at 252 days), and is within 0.05 points of the
per-quote shift at the puts with delta <= -0.5 and within 0.16 at the calls.

spot_less_carry_* compares the spot with F D + PV(dividends) for each forward. With S8's forward it alternates
between SPY's quarterly expirations, whose calls carry the dividend paid on the expiry day, and the others; with the
de-Americanized forward it is smooth in maturity, and what is left grows with it (the market's financing rate above
the Treasury yield, and a spot that is not synchronous with the options' close).

Shifts are European (S8) less American: positive means S8 reads too high. Units: volatility points; the design
noise s_iv(delta) of Lecture 5, Section 5.3 (0.3 points at the money to 1 point at |delta| = 2); and, for prices,
the noise sd S8 gives the point, which on SPY is the design noise at sigma_ref everywhere.
"""

import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LAB = HERE.parents[1]
sys.path.insert(0, str(LAB / "src"))

from sbi4finance.calibration import chain, surface  # noqa: E402

STEPS = 800
NEAR = 0.05  # in-the-money quotes within this log-moneyness are de-Americanized too: parity reads them


def dividend_times(settings, valuation):
    """(the last cum-dividend close on S8's clock, amount): the stock opens ex on the ex-date, so exercising at the
    close before it still collects the dividend, and an option expiring on the ex-date ends on an ex-dividend price."""
    return [((chain.trading_days(valuation, date.fromisoformat(d)) - 1) / surface.TRADING_DAYS, a)
            for d, a in settings["dividends"] if date.fromisoformat(d) > valuation]


def tree(forward, discount, T, dividends, strikes, sigmas, is_put, steps):
    """American and European values per strike, each on its own CRR tree with escrowed dividends."""
    r = -np.log(discount) / T
    due = [(t, a) for t, a in dividends if t < T]
    dt = T / steps
    sign = np.where(is_put, -1.0, 1.0)[:, None]
    K = np.asarray(strikes, dtype=float)[:, None]
    up = (np.asarray(sigmas, dtype=float) * np.sqrt(dt))[:, None]
    p = (np.exp(r * dt) - np.exp(-up)) / (np.exp(up) - np.exp(-up))
    disc = np.exp(-r * dt)
    s0 = forward * discount
    eu = am = np.maximum(sign * (s0 * np.exp(up * (steps - 2 * np.arange(steps + 1))) - K), 0.0)
    for i in range(steps - 1, -1, -1):
        t = i * dt
        x = s0 * np.exp(up * (i - 2 * np.arange(i + 1)))
        pv = sum(a * np.exp(-r * (td - t)) for td, a in due if td >= t - 1e-12)
        eu = disc * (p * eu[:, :-1] + (1 - p) * eu[:, 1:])
        am = np.maximum(disc * (p * am[:, :-1] + (1 - p) * am[:, 1:]), sign * (x + pv - K))
    return am[:, 0], eu[:, 0]


def premium(forward, discount, T, dividends, strikes, sigmas, is_put, steps=STEPS):
    return 0.5 * sum(np.subtract(*tree(forward, discount, T, dividends, strikes, sigmas, is_put, n))
                     for n in (steps, steps + 1))


def black_vol(price, forward, discount, T, strikes, is_put):
    return surface.implied_vol(price / (discount * forward), np.log(strikes / forward), T,
                               np.where(is_put, "put", "call"))


def american_vol(mid, forward, discount, T, dividends, strikes, is_put, iterations=3, start=None):
    """The fixed point sigma = Black^-1(mid - premium(sigma)), from a starting premium; NaN where none exists."""
    eep = np.zeros_like(mid) if start is None else start
    sigma = black_vol(mid - eep, forward, discount, T, strikes, is_put)
    for _ in range(iterations):
        ok = np.isfinite(sigma)
        eep = np.where(ok, premium(forward, discount, T, dividends, strikes, np.where(ok, sigma, 0.2), is_put), eep)
        sigma = black_vol(mid - eep, forward, discount, T, strikes, is_put)
    return sigma, eep


def de_americanize(rows, T, discount, dividends, iterations=3):
    """The expiration's rows with each quote's premium taken off bid and ask, and the forward that goes with them."""
    forward = chain.forward_given_discount(rows, discount)
    quoted = [r for r in rows if r["bid"] > 0 and r["ask"] > r["bid"]]
    K = np.array([r["strike"] for r in quoted])
    is_put = np.array([r["type"] == "put" for r in quoted])
    mid = np.array([0.5 * (r["bid"] + r["ask"]) for r in quoted])
    dividend_due = any(t < T for t, _ in dividends)  # without one, an American call is a European call
    eep = np.zeros(len(K))
    for n in range(iterations):
        k = np.log(K / forward)
        use = np.where(is_put, k <= NEAR, (k >= -NEAR) & dividend_due)
        sigma, start, eep = np.full(len(K), np.nan), eep[use], np.zeros(len(K))
        sigma[use], eep[use] = american_vol(mid[use], forward, discount, T, dividends, K[use], is_put[use],
                                            iterations=3 if n == 0 else 2, start=start)
        lowered = {id(r): e for r, e in zip(quoted, eep)}
        new = [dict(r, bid=r["bid"] - lowered.get(id(r), 0.0), ask=r["ask"] - lowered.get(id(r), 0.0)) for r in rows]
        updated = chain.forward_given_discount(new, discount)
        moved, forward = abs(updated / forward - 1), updated
        if moved < 1e-7:
            break
    k = np.log(K / forward)
    half = np.array([0.5 * (r["ask"] - r["bid"]) for r in quoted])
    kept = is_put & (k <= 0) & ((2 * half) / mid <= 0.5) & np.isfinite(sigma)
    gross = black_vol(mid, forward, discount, T, K, is_put) - sigma
    nearest = int(np.argmin(np.where(kept, -k, np.inf))) if kept.any() else None
    return new, {"forward_american": forward, "last_forward_move": moved,
                 "out_of_the_money_puts": int(kept.sum()),
                 "premium_over_half_spread_median": float(np.median(eep[kept] / half[kept])) if kept.any() else None,
                 "premium_over_half_spread_max": float(np.max(eep[kept] / half[kept])) if kept.any() else None,
                 "put_nearest_the_money": None if nearest is None else
                 {"strike": float(K[nearest]), "k": float(k[nearest]), "premium": float(eep[nearest]),
                  "half_spread": float(half[nearest]), "premium_vol_points": 100 * float(gross[nearest])}}


def readings(Ts, reach, design_T, deltas):
    """For each design point, the expirations S8 read and their weights (chain.build_surface's rule)."""
    def covers(i, T, delta):
        return reach[i][0] <= delta * np.sqrt(T / Ts[i]) <= reach[i][1]

    out = []
    for T in design_T:
        j = int(np.searchsorted(Ts, T))
        nearest = (j, j) if np.isclose(Ts[min(j, len(Ts) - 1)], T) else (j - 1, j)
        nearest = (max(nearest[0], 0), min(nearest[1], len(Ts) - 1))
        for delta in deltas:
            below = [i for i in range(len(Ts)) if Ts[i] <= T + 1e-12 and covers(i, T, delta)]
            above = [i for i in range(len(Ts)) if Ts[i] >= T - 1e-12 and covers(i, T, delta)]
            if below and above:
                lo, hi = below[-1], above[0]
            elif below or above:
                lo = hi = below[-1] if below else above[0]
            else:
                lo, hi = nearest
            out.append([(lo, 1.0)] if lo == hi else
                       [(lo, (Ts[hi] - T) / (Ts[hi] - Ts[lo])), (hi, (T - Ts[lo]) / (Ts[hi] - Ts[lo]))])
    return out


def from_grid(grid, expirations, design_days, deltas, dividends):
    """The 'surface' estimate: the American design grid from a European one and its expirations' forwards."""
    Ts = np.array([e["days"] for e in expirations]) / surface.TRADING_DAYS
    F = np.array([e["forward"] for e in expirations])
    D = np.array([e["discount"] for e in expirations])
    design_T = np.array(design_days) / surface.TRADING_DAYS
    deltas = np.array(deltas)
    grid = np.asarray(grid).reshape(len(design_T), len(deltas))
    k = deltas[None, :] * surface.SIGMA_REF * np.sqrt(design_T)[:, None]
    slope = np.gradient(grid, axis=1) / np.gradient(k, axis=1)
    atm = grid[:, int(np.argmin(np.abs(deltas)))]
    atm_at = np.sqrt(np.interp(Ts, design_T, atm**2 * design_T) / Ts)
    atm_at = np.where(Ts < design_T[0], atm[0], np.where(Ts > design_T[-1], atm[-1], atm_at))
    F_A = F.copy()
    for i in range(len(Ts)):
        for _ in range(3):
            put, call = premium(F_A[i], D[i], Ts[i], dividends, np.array([F[i], F[i]]), np.full(2, atm_at[i]),
                                np.array([True, False]))
            F_A[i] = F[i] + (put - call) / D[i]
    plan = readings(Ts, [e["strip_deltas"] for e in expirations], design_T, deltas)
    shift = {}
    for i in sorted({idx for point in plan for idx, _ in point}):
        points = [n for n, point in enumerate(plan) if any(idx == i for idx, _ in point)]
        kk = k.ravel()[points]
        K = F_A[i] * np.exp(kk)
        is_put = deltas[np.array(points) % len(deltas)] <= 0
        sigma_E = grid.ravel()[points] + slope.ravel()[points] * np.log(F_A[i] / F[i])
        mid = D[i] * F[i] * surface.bs_price(np.log(K / F[i]), Ts[i], sigma_E, np.where(is_put, "put", "call"))
        sigma_A, _ = american_vol(mid, F_A[i], D[i], Ts[i], dividends, K, is_put)
        for n, s in zip(points, grid.ravel()[points] - sigma_A):
            shift[(n, i)] = s
    american = []
    for n, point in enumerate(plan):
        sigma, T = grid.ravel()[n], design_T[n // len(deltas)]
        if len(point) == 1:  # one expiration: S8 takes its volatility at this log-moneyness
            american.append(sigma - shift[(n, point[0][0])])
        else:
            american.append(np.sqrt(sum(w * (sigma - shift[(n, i)]) ** 2 * Ts[i] for i, w in point) / T))
    return np.array(american), {"forward_american": F_A.tolist(), "atm_vol": atm_at.tolist(),
                                "two_expiration_points": sum(len(p) == 2 for p in plan)}


def grid_shift(european, american, prices_european, prices_american, noise_sd, design_days, deltas):
    eu, am = np.asarray(european), np.asarray(american)
    shape = (len(design_days), len(deltas))
    s_iv = np.tile(surface.implied_vol_noise(np.array(deltas)), len(design_days))
    vol = (eu - am).reshape(shape)
    units = ((eu - am) / s_iv).reshape(shape)
    price = ((np.asarray(prices_european) - np.asarray(prices_american)) / np.asarray(noise_sd)).reshape(shape)
    atm = int(np.argmin(np.abs(deltas)))
    puts = np.array(deltas) <= 0
    flat = units.ravel()
    largest = int(np.argmax(np.abs(flat)))
    k = np.tile(np.array(deltas), len(design_days)) * surface.SIGMA_REF * np.sqrt(
        np.repeat(np.array(design_days) / surface.TRADING_DAYS, len(deltas)))
    T = np.repeat(np.array(design_days) / surface.TRADING_DAYS, len(deltas))
    smile = {}
    for name, grid in (("european", eu), ("american", am)):
        maturities, _, skew, _ = surface.smile_statistics(grid, k, T)
        exponent, _ = surface.skew_power_law(maturities, skew)
        smile[name] = {"atm_skew": np.round(skew, 4).tolist(), "skew_exponent": round(exponent, 4)}
    return {
        "european_vol_percent": np.round(100 * eu.reshape(shape), 4).tolist(),
        "american_vol_percent": np.round(100 * am.reshape(shape), 4).tolist(),
        "vol_points": np.round(100 * vol, 4).tolist(),
        "design_noise_units": np.round(units, 4).tolist(),
        "price_shift_over_noise_sd": np.round(price, 4).tolist(),
        "summary": {
            "at_the_money_vol_points_by_maturity": np.round(100 * vol[:, atm], 3).tolist(),
            "at_the_money_design_noise_units_by_maturity": np.round(units[:, atm], 3).tolist(),
            "largest_put_vol_points": round(100 * float(vol[:, puts].max()), 3),
            "largest_call_vol_points_in_size": round(100 * float(np.abs(vol[:, ~puts]).max()), 3),
            "largest_design_noise_units": {"value": round(float(flat[largest]), 3),
                                           "days": design_days[largest // len(deltas)],
                                           "delta": deltas[largest % len(deltas)]},
            "points_at_least_half_the_design_noise": int(np.sum(np.abs(flat) >= 0.5)),
            "points_at_least_the_design_noise": int(np.sum(np.abs(flat) >= 1.0)),
            "mean_over_63_points_design_noise_units": round(float(flat.mean()), 3)},
        "smile": smile}


def design_prices(implied, design_days, deltas):
    T = np.repeat(np.array(design_days) / surface.TRADING_DAYS, len(deltas))
    d = np.tile(np.array(deltas), len(design_days))
    k = d * surface.SIGMA_REF * np.sqrt(T)
    return surface.bs_price(k, T, np.asarray(implied), np.where(d <= 0, "put", "call"))


def chain_check(raw_path, asof, settings):
    text = Path(raw_path).read_text()
    payload = json.loads(text)
    rows = chain.parse_cboe_json(text)
    valuation = rows[0]["date"]
    if valuation.isoformat() != asof:
        raise ValueError(f"the file holds quotes of {valuation}, not {asof}")
    day = settings[asof]
    curve = chain.rate_curve(day["rate_curve"])
    dividends = dividend_times(settings, valuation)
    options = {"replication_deltas": tuple(day["replication_deltas"])} if "replication_deltas" in day else {}
    european = chain.build_surface(rows, rate=curve, **options)
    by_expiry = {}
    for r in rows:
        by_expiry.setdefault(r["expiration"], []).append(r)
    american_rows, expirations = [], []
    used = {e["expiration"] for e in european["diagnostics"]["expirations"]}
    for expiry, quotes in sorted(by_expiry.items()):
        days = chain.trading_days(valuation, expiry)
        if expiry.isoformat() not in used:
            american_rows += quotes
            continue
        T = days / surface.TRADING_DAYS
        discount = float(np.exp(-curve(T) * T))
        new, info = de_americanize(quotes, T, discount, dividends)
        american_rows += new
        forward = chain.forward_given_discount(quotes, discount)
        pv = sum(a * discount ** (t / T) for t, a in dividends if t < T)
        expirations.append({"expiration": expiry.isoformat(), "days": days, "forward_european": forward,
                            "forward_change": info["forward_american"] / forward - 1,
                            "spot_less_carry_european": day["spot"] - forward * discount - pv,
                            "spot_less_carry_american": day["spot"] - info["forward_american"] * discount - pv} | info)
        print(f"{expiry} ({days} d): forward {forward:.3f} -> {info['forward_american']:.3f}; put premium up to "
              f"{info['premium_over_half_spread_max']:.2f} half-spreads", flush=True)
    american = chain.build_surface(american_rows, rate=curve, **options)
    design = european["diagnostics"]
    days_, deltas = list(surface.DESIGN_DAYS), list(surface.DESIGN_DELTAS)
    exact = grid_shift(design["implied_vol_grid"], american["diagnostics"]["implied_vol_grid"], european["prices"],
                       american["prices"], european["noise_sd"], days_, deltas)
    approximate, estimate_info = estimate(design, european["prices"], european["noise_sd"], days_, deltas, dividends)
    gap = np.array(approximate["vol_points"]) - np.array(exact["vol_points"])
    levels_e, levels_a = european["forward_variance_levels"], american["forward_variance_levels"]
    return {
        "check": "spy_early_exercise", "mode": "chain", "asof": asof,
        "source": f"Cboe delayed quotes for SPY, file timestamp {payload.get('timestamp')}; quotes of the {asof} close",
        "chain_sha256": hashlib.sha256(text.encode()).hexdigest(), "raw_quotes_committed": False,
        "inputs": {"rate_curve": day["rate_curve"], "rate_source": day["rate_source"], "spot": day["spot"],
                   "dividends": settings["dividends"], "tree_steps": [STEPS, STEPS + 1],
                   "replication_deltas": design["replication_deltas"]},
        "design": {"days": days_, "deltas": deltas},
        "readings": {"european": design["grid_point_readings"], "american": american["diagnostics"]["grid_point_readings"]},
        "shift": exact,
        "forward_variance_levels": {"european": levels_e.tolist(), "american": levels_a.tolist(),
                                    "relative_change": (levels_a / levels_e - 1).tolist()},
        "expirations": expirations,
        "surface_estimate": {"shift": approximate, "less_exact_vol_points_max_abs": round(float(np.abs(gap).max()), 4),
                             "less_exact_vol_points_mean": round(float(gap.mean()), 4)} | estimate_info,
    }


def estimate(design, prices, noise_sd, days_, deltas, dividends):
    grid, info = from_grid(design["implied_vol_grid"], design["expirations"], days_, deltas, dividends)
    return grid_shift(design["implied_vol_grid"], grid, prices, design_prices(grid, days_, deltas), noise_sd, days_,
                      deltas), info


def surface_check(asof, settings):
    day = settings[asof]
    path = LAB.parent / day["surface"]
    artifact = json.loads(path.read_text())
    design = artifact["diagnostics"]
    valuation = date.fromisoformat(design["valuation"])
    if valuation.isoformat() != asof:
        raise ValueError(f"{path} holds the surface of {valuation}, not {asof}")
    days_, deltas = artifact["design"]["days"], artifact["design"]["deltas"]
    dividends = dividend_times(settings, valuation)
    shift, info = estimate(design, artifact["prices_forward_units"], artifact["noise_sd"], days_, deltas, dividends)
    D = np.array([e["discount"] for e in design["expirations"]])
    T = np.array([e["days"] for e in design["expirations"]]) / surface.TRADING_DAYS
    pv = np.array([sum(a * d ** (t / tt) for t, a in dividends if t < tt) for d, tt in zip(D, T)])
    F = np.array([e["forward"] for e in design["expirations"]])
    return {
        "check": "spy_early_exercise", "mode": "surface", "asof": asof,
        "source": f"{day['surface']} (study S8; {artifact['source']})", "chain_sha256": artifact["chain_sha256"],
        "raw_quotes_committed": False,
        "inputs": {"rate_curve": day["rate_curve"], "rate_source": day["rate_source"], "spot": day["spot"],
                   "dividends": settings["dividends"], "tree_steps": [STEPS, STEPS + 1]},
        "design": {"days": days_, "deltas": deltas},
        "shift": shift,
        "expirations": [{"expiration": e["expiration"], "days": e["days"], "forward_european": e["forward"],
                         "forward_american": fa, "forward_change": fa / e["forward"] - 1,
                         "spot_less_carry_european": day["spot"] - e["forward"] * e["discount"] - p,
                         "spot_less_carry_american": day["spot"] - fa * e["discount"] - p}
                        for e, fa, p in zip(design["expirations"], info["forward_american"], pv)],
        "two_expiration_points": info["two_expiration_points"],
    }


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def main(mode, *args):
    settings = json.loads((HERE / "settings.json").read_text())
    if mode == "chain":
        raw_path, asof, out_path = args
        out = chain_check(raw_path, asof, settings)
    elif mode == "surface":
        asof, out_path = args
        out = surface_check(asof, settings)
    else:
        raise SystemExit(__doc__)
    Path(out_path).write_text(json.dumps(clean(out), indent=1) + "\n")
    print(json.dumps(out["shift"]["summary"], indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
