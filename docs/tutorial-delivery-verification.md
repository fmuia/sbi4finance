# Participant delivery verification for issue #26

The selected **120-minute Q/options route** passed fresh locked setup and
isolated execution on the combined source supplied by #27. The route uses
GBM foundations, the local synthetic panel, provisional corrected HKDE and the
#29 diagnostic decision. All five exports passed strict marimo checks with
zero cell errors; all 19 displayed figure appearances and the six exercises
were reviewed. **Browser layout, interactive controls and offline browser
runtime remain unverified** because Computer Use permissions were unavailable.
This report does not claim standalone offline HTML delivery.

The [route card](tutorial-participant-route.md) supplies the curriculum and
exercise answers. [PR #31](https://github.com/fmuia/sbi4finance/pull/31) owns
the combined delivery; [PR #30](https://github.com/fmuia/sbi4finance/pull/30)
supplies the curriculum and this evidence. No GitHub PR was merged.

## Exact source and scope

| Binding | Verified value |
|---|---|
| Executable source | `ca63424a140c2e512db97e40f0bb30010da3956a` |
| Git tree | `0eaff7f14ae721efc74de6bc8784b7fab44f67a9` |
| Verification checkout | Separate detached `sbi4finance-tutorial-verified`; clean before/after |
| Environment | Fresh real `.pixi`, created in that checkout; no shared-environment symlink |
| Manifest SHA256 | `bd9c7de43610b62d559487aeab11c5caa5e52fbfc55795a749974fc46c575027` |
| Lock SHA256 | `749dac2b9ae02df22ddaa8ae298d18804c53d7a040103a8f1df0dffc488d8933` |
| Scientific decision | `hkde_2048_provisional_scope_2026_10_12`: provisional, scientific acceptance false, formal I4 deferred |

The pinned tree includes #22's local panel, cached default and explicit
2,048-proposal opt-in, #29's five-figure decision notebook11, and the selected
route introduction/closing in notebook10. Later evidence/prose commits in
#31 do not replace this executable pin. The integration checkout was not edited.

The [execution record](results/tutorial-participant-delivery/verification.json)
binds all **158 copied inputs** to their Git content and SHA256, before and
after execution. It also verifies 59 required frozen scientific dependencies
against #29's manifest. That is a required-input subset, not the whole historical
inventory: raw surrogate training banks are deliberately absent. Integrated
teaching interfaces are bound to the combined Git revision.

Only the five required `data/synthetic/hkde-walkthrough` files were copied from
`data/`. Market fixtures and raw training banks were absent. The existing GBM
exercise creates and removes its tiny temporary 64/16/8-example bank. There
was no neural training, new acceptance cohort, reserved-test execution or
market acquisition. Existing held-out reports were read as historical evidence.
The integration owner's wider tests and rough-model review were not repeated.

## Fresh setup and measured execution

The [setup record](results/tutorial-participant-delivery/setup.json) and
[install log](results/tutorial-participant-delivery/install.log) record
`pixi install --locked`, exit 0, **6.789 s**, starting
`2026-10-12T21:20:16.740686+00:00`. `.pixi` did not exist beforehand. Normal
package caches were allowed: this is fresh environment construction, not a
cold-download measurement. No dependency or lock change was needed.

Machine: **Apple M5 Max, 128 GiB memory, macOS 26.4.1 arm64**, CPU backend.
Pixi 0.70.2 installed Python 3.12.15, BayesFlow 2.0.14, Keras 3.15.1,
JAX/JAXlib 0.11.1 and marimo 0.25.1; setup JSON lists all checked versions.
Linux was not installed. This machine is an observation, not a minimum
hardware requirement or a performance guarantee.

Each export used a fresh process and an isolated source copy. A tested Python
socket/DNS guard blocked external TCP, UDP and DNS while permitting loopback
and Unix IPC. This is a Python-level guard, not an OS network sandbox or a
browser network test. Cached/analytic processes also blocked BayesFlow imports;
only the explicit live export could load it.

| Notebook / mode | Output cells | Figures | Cell errors | Export seconds |
|---|---:|---:|---:|---:|
| `01_gbm_foundations`, bounded analytic/live | 21 | 2 | 0 | 28.267 |
| `hkde_data_walkthrough`, cached | 8 | 2 | 0 | 1.675 |
| `10_hkde_corrected_workflow`, cached default | 20 | 4 | 0 | 25.471 |
| `10_hkde_corrected_workflow`, explicit provisional live | 20 | 6 | 0 | 25.765 |
| `11_hkde_calibration_audit`, #29 decision | 15 | 5 | 0 | 1.167 |

All five strict checks and exports passed in one successful checker invocation.
Total export time was **82.345 s**; isolation, hashes, checks, exports and the
initial review bundle took **92.011 s**. Subsequent link-only postprocessing
and manual review are outside that wall time. Initial font-cache/framework
costs and concurrent machine activity can affect these observations.
The live export retained one Python semaphore-cleanup warning at shutdown;
the process exited successfully and no notebook cell failed.

The live cell measured **21.555 s**, with these overlapping components:

| Component | Seconds |
|---|---:|
| Imports, artifact bindings and checkpoint loading | 18.328 |
| Proposal sampling | 1.040 |
| Proposal density | 0.183 |
| Proposal total, includes the preceding two rows | 1.223 |
| Nine-price numerical likelihood and weights | 1.998 |
| Two-target surrogate | 0.000600 |
| Reference comparison and summary | 0.00580 |

Do not add the cell, export and component totals. No comparison here establishes
an amortization speedup. The raw cache contains 4,096 draws, the corrected
precision cache uses 8,192 proposals, and the explicit live calculation uses
2,048 proposals at fixed `validation:257`, seed `3667540760`.

All eight existing selected-case checks passed: maximum target-quantile error
was 0.005299 currency (limit 0.03), ESS fraction 0.687767 (minimum 0.5), and
maximum normalized weight 0.002085 (maximum 0.01). The broader historical
calibration failure remains visible and scientific acceptance remains false.
The export retained tables, plots, flags and timings, not the raw live rows;
it is not an additional independent row-alignment experiment.

## Figures, tables and exercises

The [review record](results/tutorial-participant-delivery/review.json) and
[figure inventory](results/tutorial-participant-delivery/visual-review.json)
identify every displayed figure by SHA256 and repository path. There are
**19 appearances and 15 distinct PNGs**. All were visually inspected directly
or matched byte-for-byte to an already inspected figure. Eight new PNGs are
committed; existing identical figures are referenced without duplication.
Axes, units, legends, paired-posterior structure and failure/budget labels are
readable. This is extracted-figure inspection, not browser-page layout review.

Exported tables and exercise text were read alongside the selected route:

- E1 was replayed through the actual marimo graph with both slider definitions
  overridden, seed 2026, at 252 and 504 days. The longer history extends the
  same path. Drift variance falls from 0.008 to 0.0066667; doubling sampling
  frequency within one year leaves it 0.008. The
  [exercise result](results/tutorial-participant-delivery/exercise-e1.json)
  records this separately from browser slider interaction, which was not tested.
- E2 preserves the observation/context/truth boundary: nine ordered premiums
  condition inference; synthetic truth, timestamps and target prices do not.
  The cached raw and corrected tables refer to the same selected observation.
- E3/E4 retain the full mixture denominator, reference comparison and paired
  parameter/weight/target interpretation. A target-price interval describes
  uncertainty in a Q expectation; it is not a payoff or physical-loss interval.
- E5/E6 retain the shared historical rank alarm, the conditional-versus-SBC
  distinction, provisional decision, and limits of changing the prior, payoff
  type or P/Q question. These are reasoning exercises, not new experiments.

Missing optional M/E inputs do not block the synthetic route. No real-panel
or empirical-market claim was added. The original MMAR material remains an
optional companion; its earlier prose-only scope correction passed strict
checking in the [preflight record](results/tutorial-participant-preflight/verification.json).

## Browser delivery findings and remaining check

The five original marimo exports each declare **107 CDN frontend references**.
They also contain temporary absolute and source-relative review links.
Successful Python export therefore does not certify portable standalone
offline HTML. Python-backed controls require a running marimo session.

For review, the checker creates a derived `browser/` directory containing
482 installed marimo 0.25.1 frontend assets, the 158-file source snapshot,
and five transformed HTML files. It replaces CDN destinations with matching
local assets and rewrites rendered anchor destinations. Required lesson/docs
links resolve within the bundle; two optional provenance links explicitly
point to pinned GitHub sources (Heston05 and #29's historical verification).
Those two links still need internet access. Displayed reproduction commands
retain their original text and are not promised portable shell commands.

The [delivery manifest](results/tutorial-participant-delivery/browser-delivery-manifest.json)
hashes every bundled file and records the link mappings. Static checks passed
for **583 local targets across the five pages**. They do not audit every
transitive link in copied Markdown or execute frontend JavaScript. After the
original export, a link audit found 15 broken rendered link occurrences; these
were repaired in the derived bundle only. The record preserves the original
execution-verifier hash and the later postprocessor hash separately. All
original HTML, notebook code and 19 PNGs remained byte-identical; no scientific
calculation was rerun to repair links.

Attempting browser inspection returned **“Computer Use permissions are not
granted.”** No browser screenshot, interaction or request trace was obtained.
`browser_offline_verified` and `full_selected_route_verified` therefore remain
false. The remaining manual check is to serve the bundle over loopback with
external requests blocked, inspect all five pages and navigation, and separately
exercise the live marimo controls. No permission-dependent result is inferred
from static file checks. H26.3's fresh setup and isolated execution evidence is
complete; browser delivery retains this explicit limitation. Formal I4 remains
deferred under the scientific decision regardless of that browser check.

## Reproduction and evidence provenance

Use a new checkout at the full source pin, install its locked environment,
then run the final checker from the evidence branch with that environment:

```bash
git worktree add --detach ../sbi4finance-participant-check \
  ca63424a140c2e512db97e40f0bb30010da3956a
cd ../sbi4finance-participant-check
pixi install --locked
.pixi/envs/default/bin/python /path/to/evidence-checkout/scripts/check_tutorial_delivery.py \
  --scope participant --repo "$PWD" \
  --revision ca63424a140c2e512db97e40f0bb30010da3956a \
  --out /tmp/tutorial-participant-new
```

The output directory must be new. `--scope preflight` still describes the
older three-notebook tree and must not be used unchanged on #29's notebook11.
The current script incorporates the later link repair; the successful original
execution used the retained
[verifier snapshot](results/tutorial-participant-delivery/execution-verifier.py).
The checker always leaves installation/manual/browser acceptance separate.

Original HTML, the isolated source, derived bundle and prior mirror are retained
locally in `/tmp/sbi4finance-participant-final`; they are temporary review
artifacts, not a published participant hosting service. Small reports, logs,
verification-procedure snapshots and new figures are committed under
`docs/results/tutorial-participant-delivery/`. The 26 MB frontend and full HTML
are not committed. [review.json](results/tutorial-participant-delivery/review.json)
binds the retained records and records manual findings; the automated
`verification.json` deliberately retains its original pending manual fields.

The earlier [preflight evidence](results/tutorial-participant-preflight/verification.json)
remains historical and is not substituted for this combined-tree run. This
final commit is confined to the delivery report/results and verifier. The
integration owner updates the route, README and handover links in #31.
