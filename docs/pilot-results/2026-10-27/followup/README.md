# Bounded follow-up evidence

Read the [human results and next decisions](../../../followup-results-2026-10-27.md). All seven Oscar allocations completed; no final acceptance or new bank is claimed.

- `execution.json`, `accounting.psv`, `accounting-summary.json`: exact reviewed scope, jobs, sources, GPU request amendments, terminal accounting and budgets. Top-level CPU is counted once.
- `archived-heston/`: frozen BF2.0.14 reader declaration, original checkpoint/accepted-record hashes, exact reference identities, three saved neural-draw arrays and comparisons. The local process CPU measurement counts all JAX threads. To reproduce the original seed audit, use the frozen Heston source revision `51fec86`; this unit's later results were not part of that revision's prior-seed inventory.
- `heston/{spline-1000,flow-matching-500}/`: run journals, command logs, training history/selected epoch, 500-case validation reports and three saved-reference comparisons. Neither fit passes the full development checks. `artifacts-retained-on-oscar.json` records model, recovery-checkpoint and large validation-draw hashes/sizes; those payloads remain on Oscar. `downloaded-files.sha256` preserves retrieved artifact identities.
- `rough/declaration.json`: exact 52-call cohort, prior reference declaration, policies, sources/runtime and stream audit.
- `rough/groups/group-*/report.json.gz`: all four raw reports, including each of the 52 calls and original covariance/coefficient summaries. Compression is lossless; `rough/compressed-artifacts.json` records compressed and original hashes.
- `rough/reduction/report.json`, `rough/analysis.json`: checked reduction and descriptive empirical error/timing summaries. These are not global accuracy or coverage certificates.
- `rough/cross-batch-method.json`, `rough/cross-batch-analysis.json`: exploratory method frozen after initial inspection but before calculation, adjacent-pair results and input/source hashes. The analysis adds no paths or model evaluations.
- `rough/accounting*`, `rough/node-snapshot.txt`, `rough/logs/`: original cluster evidence. CPU frequency counters are not GPU frequencies.

The two source bundles remain under `/oscar/scratch/afengler/sbi4finance-followup-20261027-f4d3679`, with separate immutable checkouts at `51fec86` and `615a72f`. Locked runtime prefixes were reused only after byte identity checks. Sources, tasks, checkpoints and reference observations were not altered during execution.

## Replay the saved-data reductions

Use a fresh temporary directory; the commands below perform no simulation, fitting or notebook execution. Run from the repository root in the locked core environment. Substitute absolute temporary paths as needed.

First restore the original report bytes and verify their hashes:

```python
import gzip
import hashlib
import json
from pathlib import Path

root = Path("docs/pilot-results/2026-10-27/followup/rough")
restored = Path("/tmp/sbi4finance-followup-replay-groups")
restored.mkdir(exist_ok=False)
manifest = json.loads((root / "compressed-artifacts.json").read_text())
for item in manifest["files"]:
    raw = gzip.decompress((root / item["path"]).read_bytes())
    assert hashlib.sha256(raw).hexdigest() == item["uncompressed_sha256"]
    relative = Path(item["uncompressed_path"]).relative_to("groups")
    target = restored / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
```

Then run the existing reducer with the retained accounting:

```bash
pixi run --locked -e core python scripts/diagnose_rough_controls.py reduce \
  --declaration docs/pilot-results/2026-10-27/followup/rough/declaration.json \
  --groups /tmp/sbi4finance-followup-replay-groups \
  --accounting docs/pilot-results/2026-10-27/followup/rough/accounting.json \
  --out /tmp/sbi4finance-followup-replay-reduction
```

The exploratory reader accepts gzip files directly and retains the frozen method:

```bash
pixi run --locked -e core python scripts/analyze_saved_rough_controls.py \
  --root docs/pilot-results/2026-10-27/followup/rough \
  --method docs/pilot-results/2026-10-27/followup/rough/cross-batch-method.json \
  --out /tmp/sbi4finance-followup-replay-cross-batch.json
```

Stored file paths may differ after compression; original-byte identities and numerical results must match. Preserve the original method and outputs. A replay of the same streams is not new independent evidence.
