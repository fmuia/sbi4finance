# Heston summary-dimension lesson

From `lab/`, open the standalone notebook:

```sh
pixi run -e core-reference marimo edit notebooks/heston_summary_experiments.py
```

A [saved HTML preview](https://drive.google.com/file/d/154UMN57URs4IcPy23BigGbQB7XquqB4_/view)
is also on Drive; download and open it for a static view. Run the notebook above
for interactive controls.

The lesson uses only `results.json` and the ten recovery PNGs in this directory.
It needs marimo, NumPy and matplotlib, with no simulation-bank or checkpoint
download. Its schematic is rendered by the notebook. The existing SPX notebook
will be extended in a separate PR.

## Results and scope

Nine summary sizes (64, 32, 16, 8, 7, 6, 5, 4, 2) were trained on the same
300,000 noisy option surfaces for 300 epochs, with seed 9011. Final epoch 300 is
primary; the distinct validation minimum is secondary. The compact record keeps
19 endpoints, with each of the nine learning histories stored once. Source,
bank, population, reference and image checksums preserve the evidence bindings.

The fitted models show a parameter-specific breakdown: `c` recovery deteriorates
between 6 and 5 coordinates; physical-parameter recovery deteriorates further at
4. Dimension 5 still has physical-parameter RMSEs similar to the direct baseline.
These one-seed results on reused development cases establish an empirical
comparison, not a sufficient-statistic theorem or scientific acceptance. All
complete conditional-reference panels remain inadequate. Dimensions 6 and 5
trained on Oscar GPUs; the other learned summaries trained on local CPU. The
notebook reports backend metadata and the limits of this comparison.

## Full private experiment archive

[Download the experiment archive](https://drive.google.com/file/d/14So_ApeOVUZYV6J14ZIyV2tKaOljLaZP/view).
The [Drive folder](https://drive.google.com/drive/folders/1TdWIn1Wgirz65ekJ0_bhhh1QDf1lc7GX)
contains `heston-summary-dimensions-20261114.tar.gz`, its `manifest.json`,
`SHA256SUMS`, reproduction README and packaging receipt.

- Archive size: **387,881,072 bytes**.
- Archive SHA-256: `e8f0595e54dfb73357f5d2e4d9e97bfc4fdbfd026eb4321b09ae6211c1c63191`.
- Manifest SHA-256: `498087457733b8ab7fdcaa54a9336038b3140af4b6da304b705b24a38b60743a`.
- Original experiment source: `bea2e26ea7633711c88067f5b3a63771dcf9439a`.

The archive preserves all final and validation-minimum models, raw posterior
draws, full reference chains, baseline draws, source history, producer/runtime
identities and complete experiment records. Every archive member was read back
and checked against its manifest. It includes the original training and analysis
tools; those tools are not needed to open this lesson.

The reusable 300k bank is already in the
[existing actual63 Drive folder](https://drive.google.com/drive/folders/1UYBredhF2hrxoZI-sP7sAawC3V6XQpB8),
so the experiment archive links it instead of duplicating it. Its archive SHA-256
is `a2c23ab962d534e8807de0488c37e93581062bac8d87e81dc32a2f06c3380b3a`.
The extracted bank payload SHA-256 is
`47c7f1f25756162985d437e6a133c2651656e7c506bdc2b8aeb39b33fcb803c4`.

Keep full archives outside the Git checkout. The compact lesson files are enough
for review, discussion and later integration into the SPX tutorial.
