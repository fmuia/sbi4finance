# sbi4finance

The course, participant notebooks and numerical work for **Amortized Bayesian Calibration of Volatility Models**,
the ICAIF '26 tutorial in Milan, 14–15 November 2026. Inference uses [BayesFlow](https://github.com/bayesflow-org/bayesflow).

The core path follows option quotes through Heston, rough Bergomi with a supplied variance curve, and joint
curve inference. The next task is to turn the completed numerical backbone into a clear lesson: inference,
checks, posterior interpretation and the uncertainty of a financial output.

## Start editing here

| Work | Main editing files | Plan and supporting material |
|---|---|---|
| Agree on the story, scope and division of work | [`tutorial/curriculum/00-overview.md`](tutorial/curriculum/00-overview.md) | Owners, timing and decisions; D21 excludes the returns track |
| Heston inference workflow | [`tutorial/notebooks/slot4_heston_bayesflow.py`](tutorial/notebooks/slot4_heston_bayesflow.py) | [Slot 4 plan](tutorial/curriculum/04-slot4-heston-in-bayesflow.md); course lectures [10](course/10-heston.md), [11](course/11-calibrating-heston.md), [12](course/12-the-amortized-workflow-on-heston.md) |
| Rough Bergomi and the financial output | [`tutorial/notebooks/slot5_calibration.py`](tutorial/notebooks/slot5_calibration.py) | [Slot 5 plan](tutorial/curriculum/05-slot5-rough-bergomi-to-a-reserve.md); lectures [13](course/13-rough-bergomi.md), [14](course/14-why-the-exact-posterior-is-expensive.md), [15](course/15-a-liquid-smile-spy.md) |
| Explain amortization and architecture choices | [`slot2_amortized_bayes.py`](tutorial/notebooks/slot2_amortized_bayes.py), [`slot3_architectures.py`](tutorial/notebooks/slot3_architectures.py) | Their curriculum files; lectures [8](course/08-bayesian-calibration.md), [9](course/09-amortized-inference.md) |
| Review current saved posteriors before changing a lesson | [`tutorial/notebooks/backbone_review.py`](tutorial/notebooks/backbone_review.py) | Six cached synthetic cases, with optional matching native model loading |
| Change a numerical implementation or experiment | [`lab/README.md`](lab/README.md) | Canonical loading, training, evaluation, generation and publication routes |
| Check the evidence behind a claim | [`course/status.md`](course/status.md), [`tutorial/CLAIMS.md`](tutorial/CLAIMS.md) | Current J1/J2/J3 status and separately identified historical studies |

For a local session follow [the tutorial README](tutorial/README.md). For training data and numerical work follow
[the lab README](lab/README.md). [The course README](course/README.md) indexes the theory. Repository rules are in
[AGENTS.md](AGENTS.md).

## Current state and next priorities

All three routes have immutable **100k training banks and completed baseline fits**. The participant package uses
**BayesFlow 2.1** and opens the six cached cases without loading a neural backend. Fresh sampling uses the matching
external checkpoints and BayesFlow's native adaptive FlowMatching sampler. Historical SPY results and archived
2.0 checkpoints retain their original provenance; their neural comparisons await a separately checked update.

The first content review is to reconcile the slot 4/5 curriculum files and notebook prose with this complete-surface,
known-then-joint sequence. They still contain historical SPY/known-curve framing; keep that provenance explicit
and make thin-chain material optional. The cached integration is ready for review; the lessons are not yet fully curated.

J1's joint validation, contraction and observation-dependence checks are complete as development evidence.
**J2's conditional references and J3's product/long-end comparisons remain pending.** Review the cached Heston
lesson, then known-curve and joint-curve rough examples, before commissioning more computation. The
[numerical plan](lab/plans/NUMERICAL-EXPERIMENT-PLAN.md) defines the next work and the dependency checks needed
before trimming. Sparse quote chains and the returns curriculum are outside this core unit; slot 6 remains in preparation.

The [versioned GitHub release](https://github.com/fmuia/sbi4finance/releases/tag/tutorial-backbones-2026-10-29)
and [Google Drive mirror](https://drive.google.com/drive/folders/1Zez6mSu8Y_B7IIjdRYWJlDxdAl49fTLq) provide the
banks, pretrained models and small teaching cases. The [training-data guide](lab/plans/training-data-guide.md)
gives exact archive names, checksums and executable loading/training examples. No cluster connection or full bank
is needed for content review or pretrained laptop inference. GitHub downloads require access to the repository.

## Where things belong

| Area | Purpose |
|---|---|
| `course/` | The sixteen lectures, figures and scripts that draw them; scientific status in `course/status.md` |
| `tutorial/` | The self-contained participant package: notebooks, copied runtime functions and small published assets |
| `lab/` | Simulators, data and estimator workflows, reference studies, checks, cluster tools, tests and numerical evidence |

Edit the explanation in its notebook or lecture. Put numerical work in the lab, then use the existing tools to
publish code and assets into the tutorial. Keep bulky banks, models and full draws outside Git, with immutable
identities and download links. Update the existing plan and status pages rather than adding another handover document.
