# Local backbone transfer manifest

[transfer-manifest.json](transfer-manifest.json) identifies the completed 100k banks and current inference checkpoints from archived run evidence. This is an offline source inventory, not a transfer-success receipt. Remote availability, unknown sizes and null hashes still need authenticated SFTP checks. No remote commands, simulations, fitting or uploads were performed for this inventory.

**Transfer update, 29 October:** [local-receipt.json](local-receipt.json) records the subsequent authenticated retrieval. All three posterior bundles, all three distinct rough pricer bundles, exact selected observations and saved validation panels are now durable local copies with recorded hashes. The complete Heston bank also passes every referenced part/payload hash (184 files, 127,777,134 bytes, plus snapshot/collection metadata). Both full rough banks are still copying; their partial local presence is not a complete-bank success receipt. No notebooks, training, new simulation or MCMC were run. [CPU inference measurements](../laptop-inference-benchmark/README.md) use the retrieved models.

Transfer small inference bundles first, then move the complete banks in the background. The current posterior loader requires only `model.keras`, `task.json` and `training.json` together. A raw simulation bank and optimizer-resume history are unnecessary for sampling from a pretrained posterior. The rough forward-pricer bundles contain `pricer.npz`, `pricer.json` and `task.json`. Heston uses the repository's analytic/Fourier pricing implementation; this inventory does not invent a learned Heston pricer.

| Route | Current posterior | Pricing artifact | Identity caveat |
|---|---|---|---|
| Heston 100k | final-500 export, `8d4a243b…` | Versioned analytic/Fourier pricer | Preserve export metadata; the original best-402 checkpoint is a different artifact. |
| Rough Bergomi, known curve, 100k | final-500, `39f1a819…` | Original `9f0f695b…`; separate full-300 diagnostic `069593fc…` | Transfer both pricers separately. The full-300 candidate has an independent pricing rescore, but no new conditional PyMC target and no automatic promotion. |
| Rough Bergomi, joint curve, 100k | final-500, `19c24d64…` | `be699c31…` | Matches the completed joint100k run and validation evidence. |

The verified Heston export and joint validation panel were reused from their temporary local copies and preserved durably, avoiding duplicate downloads. Heston's three inference files total 12,289,951 bytes; the joint validation panel is 48,718,023 bytes. The archive is `/Users/afengler/Projects/proj_sbi4finance/local-artifacts/tutorial-backbones-20261029/`, outside Git, with original scratch-relative paths under `oscar-scratch/`. Its verification records distinguish completed teaching inputs from the continuing full-bank copy. The single SFTP session queues full SHA verification after copying; an interrupted transfer can resume from the exact referenced-file list.

## Transfer order

1. Preserve the verified local Heston export and reuse the small tutorial cases already in Git. Download the two current rough posterior bundles and their separate pricer bundles.
2. Download the three `snapshot-100k/manifest.json` files and their canonical `collection/collection.json` files. Verify known hashes and enumerate the exact referenced native parts.
3. Preserve optional validation draws and small run metadata needed for local curation; most reports and run JSON are already in the repository.
4. Transfer the referenced collection parts in the background, preserving exact bytes. Verify each native part manifest and payload against the snapshot's `manifest_sha256` and `array_sha256` values.

Use one local archive root outside the Git checkout and retain paths relative to `/oscar/scratch/afengler/`. **The Heston bank's collection is not beside its snapshot**: its canonical source is `sbi4finance-full-core-20261026-wy2_j3o4/runs/heston-development/collection`, while its snapshot lives under `sbi4finance-heston-backbone-20261027/snapshot-100k`. Preserving both source-relative paths keeps the native relative references valid. The two rough banks use sibling `collection/` and `snapshot-100k/` directories.

Each full bank needs its snapshot manifest, canonical collection manifest and all referenced `parts/<native-range>/{manifest.json,arrays.npz}` directories. Snapshot manifests contain references, not embedded data. Do not flatten the directory layout, strip NPZ members or edit manifests in place. Failed attempts, `.tmp` and `.locks` are operational history rather than the completed tutorial payload; keep a separate operational archive if desired.

Heston has 100,000 training, 5,000 validation and 5,000 test surfaces. Each rough route has 100,000 training, 5,000 validation and 7,000 test surfaces. Transferring held-out data changes neither its evaluation policy nor scientific acceptance status.

## Local curation can start immediately

`data/synthetic/full-surface-backbones/artifacts.json` lists the existing small replay bundles. Heston's interior-midpoint replay uses the current final500 model. The rough replay bundles still correspond to the historical known50k and joint20k checkpoints and must retain those labels. New joint100k interior candidate cases 1983 and 1941, plus the separately labelled weak-decay case 1962, are already local in `docs/pilot-results/2026-10-29/joint100k-tradeoffs/`; they are curation inputs, not yet replacements for the notebook replay bundles.

The manifest records exact hashes and local file inventories for these cases and reports. Unknown remote bank sizes remain null rather than estimated. Native loader requirements were checked in `src/sbi4finance/calibration/full_surface.py`, `bank_collection.py` and `pricer_net.py`; completed-run identity sources are listed in the JSON.
