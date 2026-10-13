# Francesco's SVJ contribution and the main integration

Checked 13 October 2026 against draft [PR #33](https://github.com/fmuia/sbi4finance/pull/33),
head `2e3165f7b8a25bfaaf2639fa6b759e20d1a7516b`, and combined PR #31
at `1d88759e401b172ab1b88cd66bb5899631fb9abf`. This is a code/interface review,
not reproduction or scientific acceptance of the SVJ study.

Francesco's work is **not yet on main** at this snapshot. It is a draft proposal
for a physical-measure returns-to-VaR/ES application. Our selected participant
route remains GBM foundations and provisional risk-neutral HKDE calibration.
His contribution can coexist with the simulator ladder; adopting its financial
question as the running example would be a separate curriculum change.

## Purpose and relationship to our work

The new `svj/` package supplies a larger returns-based inference application:
eight model coordinates plus current volatility, a particle filter and particle
MCMC reference, neural inference, importance auditing, and risk forecasts.
It is materially beyond our restricted two-target Heston inference pilot.
Its training generates fresh simulation batches online; it does not replace
our seeded offline-bank/cluster infrastructure.

| Boundary | PR #33 SVJ | Existing simulator/data stack |
|---|---|---|
| Returns | Exactly 1,000 daily percent log returns, `[batch,time]` | Decimal log returns, explicit `dt_years`; trajectory shape `[batch,time,1]` |
| Discrete dynamics | Daily Bernoulli Gaussian jumps, positive variance floor, additive log-return drift | Substepped compound-Poisson jumps, full-truncation signed auxiliary variance, explicit drift/compensation conventions |
| Initial/terminal state | Gamma initialization; ninth target is log annualized terminal spot volatility | Task-specific initialization and signed auxiliary continuation state for Heston |
| Network | Nine-target flow-matching checkpoint and its own feature transform | Distinct restricted-task checkpoints/adapters and independently recorded gates |
| Default lesson | Bundled real VOO/SPMO window, live audit and forecast | Required synthetic route runs with market fixtures absent; optional market application is separate |

Percent-to-decimal conversion alone does not make these models interchangeable.
Keep distinct model/task identities, simulators, likelihoods, priors and
checkpoints. Joint parameter/state continuation is valuable in both workflows,
but the required state must come from that workflow's own dynamics.

## Mechanical compatibility

A read-only `git merge-tree` comparison finds only three files changed by both
branches: `.gitignore`, `README.md`, `pyproject.toml`. Conflicts occur in README
and the task/dependency table; `.gitignore` merges automatically. There are no
collisions in model modules, notebooks, tests, checkpoints or result paths.
Both use the same Python/BayesFlow/Keras/JAX pins; the contributed SciPy 1.18.1
pin is already present in our integration. A later merge should union tasks and
preserve the established Numba/environment settings rather than replacing the
whole configuration. No dependency upgrade is currently indicated.

## Bounded review items before participant integration

1. Validate the local observation boundary explicitly: P measure, declared log
   returns, 1,000 complete fixed daily steps and the 252-day convention; convert
   decimal to percent exactly once. The contributed loader currently drops
   missing values and sorts dates without our declared schedule/completeness
   checks. See [data.py](https://github.com/fmuia/sbi4finance/blob/2e3165f7b8a25bfaaf2639fa6b759e20d1a7516b/src/sbi4finance/svj/data.py).
2. Add or identify a complete synthetic lesson route and make the real-data
   section optional before claiming our market-fixtures-absent S acceptance.
   Its current notebook loads the real CSV unconditionally. See
   [notebook](https://github.com/fmuia/sbi4finance/blob/2e3165f7b8a25bfaaf2639fa6b759e20d1a7516b/notebooks/svj_var_pipeline.py).
3. Review its own conditional/reference/calibration evidence. Our gates do not
   accept a different statistical task. Retain its reported failed/flagged
   cases and initial versus replicated SBC outcomes; no automatic new study is
   proposed here. See [results](https://github.com/fmuia/sbi4finance/blob/2e3165f7b8a25bfaaf2639fa6b759e20d1a7516b/docs/svj-pipeline.md).
4. Display failed audit status clearly on forecasts. With `pmmh_on_flag=False`,
   the pipeline still forecasts from an importance result flagged as unreliable;
   the notebook displays efficiency but proceeds. This needs an explicit
   provisional/unsupported-result explanation, not a claim that particle MCMC
   always repaired it. See [pipeline.py](https://github.com/fmuia/sbi4finance/blob/2e3165f7b8a25bfaaf2639fa6b759e20d1a7516b/src/sbi4finance/svj/pipeline.py).
5. Clarify risk units. The current risk code sums percent log returns and
   negates their tail quantiles/means, while prose calls them percent-of-position
   losses. For cumulative percent log return `R`, simple-position loss in percent
   is `100 * (1 - exp(R/100))`. Either preserve and accurately label log-loss
   risk or deliberately change the target and its evidence; do not silently
   reinterpret existing numbers. See [risk.py](https://github.com/fmuia/sbi4finance/blob/2e3165f7b8a25bfaaf2639fa6b759e20d1a7516b/src/sbi4finance/svj/risk.py).

These are future PR #33 review/integration items, not blockers to merging our
already reviewed stack. The draft's code and reported tests were inspected but
its scientific experiments and software suite were not rerun in this closeout.
Its branch, draft status and recorded results remain untouched.
