# Saved-checkpoint Heston controls

Job **8031924** completed two read-only controls on the three existing development observations and PyMC references. No parameters were trained, financial prices generated, MCMC rerun, or reserved test outcomes evaluated. Each readout used 10,000 neural draws and seeds `271026 + case`.

The archived model was read with its recorded 100-step, data-end-dense Euler policy and with a tight adaptive Tsit5 override. Both meet the proposed parameter rules on **2/3 cases**. The new 100k estimator was read at the selected epoch 402 and at its already-saved final epoch 500, with the same native sampler. The final checkpoint meets the rules on **3/3**, compared with **1/3** for the selected checkpoint. This identifies checkpoint choice as a material contributor to the reported gap; it does not establish a general replacement-quality result.

## Conditional results

The proposed rules require every parameter mean to be within 0.3 reference SD and its central 90% width ratio to lie in `[0.75, 1.25]`. These are point-estimate development screens; reference and neural Monte Carlo uncertainty and product/refinement acceptance remain unassessed.

| Readout | Case 0 | Case 1 | Case 2 | Cases meeting all proposed rules |
| --- | --- | --- | --- | ---: |
| Archived, recorded Euler | Meets | Meets | Initial-variance mean fails | 2/3 |
| Archived, tight adaptive Tsit5 | Meets | Meets | Initial-variance mean fails | 2/3 |
| New 50k, selected epoch 445 | Initial-variance mean fails | Theta width fails | Initial-variance mean/width fail | 0/3 |
| New 50k, final epoch 500 | Meets | Theta mean fails | Initial-variance width fails | 1/3 |
| New 100k, selected epoch 402 | Initial-variance mean fails | Meets | Initial-variance mean/width fail | 1/3 |
| New 100k, final epoch 500 | Meets | Meets | Meets | 3/3 |

For the new 100k checkpoint:

| Case | Initial-variance mean error, selected → final, in reference SD | 90% width ratio, selected → final |
| --- | ---: | ---: |
| 0 | 0.48216 → 0.03383 | 1.13645 → 0.98370 |
| 1 | 0.06714 → 0.00242 | 1.08891 → 1.04311 |
| 2 | 0.65204 → 0.24561 | 1.35547 → 1.14761 |

The archived adaptive readout increases initial-variance widths by about 3.3–4.7% relative to its recorded Euler readout, but its classification stays 2/3. It does not reproduce the selected-versus-final difference in the new 100k model. These are fresh paired readouts; their raw point estimates need not be identical to earlier readouts on other hardware.

The archived model selected epoch 487 of 500; the new fits selected epochs 445 and 402. Although all used a minimum-validation-loss rule, that rule retained much earlier weights in the 100k fit. The paired controls demonstrate that those earlier weights have worse results on these three observations. They do not prove that a final-epoch rule is generally superior or that checkpoint selection explains every old/new difference.

[Archived readouts](results/old/run.json) and [selected/final readouts](results/new/run.json) retain all four parameter means, spreads, width ratios, paired changes, checkpoint hashes and settings. The final 50k checkpoint also improves some errors but does not resolve its full three-case screen.

## Selection scope and provenance

**The final-epoch checkpoints were inspected after seeing the selected-checkpoint failures. They are post-hoc development candidates.** A passing three-case result does not justify silently replacing the recorded selected model or claiming independent acceptance. The separately budgeted final-100k evaluation uses the unchanged 2,000-case development panel, with its post-hoc status retained.

The final100k checkpoint SHA is `8d4a243ba099524cf4cda40e4f0616e895879f053829a7f60dadbe65b98301db`; the original selected checkpoint and its reports are unchanged. [inputs.json](inputs.json) binds the original models, recovery checkpoints, exact observations/references, source and old-library overlay. All **404 frozen input/source hashes** were rechecked. Nine draw payloads remain on Oscar; [remote-artifacts.json](remote-artifacts.json) records their byte counts and SHA-256 hashes without transferring them.

The allocation consumed **168 seconds × four CPUs = 672 allocated CPU-seconds**, and **168 GPU-seconds**, below its separate **2 CPU-hour / 0.25 GPU-hour ceilings**. The role processes took 63.51 seconds (archive) and 71.40 seconds (new checkpoints); allocation accounting includes setup and process overhead. Do not add job-step records to the top-level allocation again. [Accounting](accounting.psv), [execution ledger](execution.json), [original submission ledger](execution-submitted.json) and [Slurm log](logs/slurm-8031924.out) preserve the exact attempt.

The one-off [readout script](audit_heston_saved_readouts_20261027.py), [wrapper](run_heston_audit_unit_20261027.py) and [input preparation](prepare_heston_readout_audit_20261027.py) are retained for review. [downloaded-files.sha256](downloaded-files.sha256) verifies the small source/report/log copies, including both the original and normalized ledger. Model and draw payloads were not downloaded.
