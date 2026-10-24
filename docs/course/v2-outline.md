# Version 2: where each section comes from

24 October 2026. [Course index](README.md) · [Plan](v2-plan.md).

"L9 9.4" is version 1's Lecture 9, Section 9.4, at commit 19a6ea4
(`git show 19a6ea4:docs/course/09-calibration-as-estimation.md`). Figures keep their image files in `images/`;
only their numbers change (Figure N.k is the k-th figure of Lecture N). The figure scripts in `scripts/course/`
keep version 1's lecture numbers.

## Part I. Markets and the first model

| Lecture | Sections | New material |
| --- | --- | --- |
| 1. Options, quotes and volatility | 1.1–1.8 = L1 | A thin chain next to SPY: EWI and EWG of 22 October 2026 (`images/v2-thin-chain.png`); "marking a book" points to Lecture 6 |
| 2. No-arbitrage without a model | 2.1–2.11 = L2 | Calls before an ex-dividend date (EWI); why parity cannot pin a rate on a thin chain; why the course uses the carry forward on EWG |
| 3. Brownian motion, SDEs and Itô's lemma | = L3 | None |
| 4. The Black–Scholes model | 4.1–4.11 = L4; 4.12 = L8 8.5 (Black–Scholes part) | A term structure of volatility and the forward start under it |
| 5. Implied volatility and the smile | = L5 | Strips on a thin chain (EWI) |

## Part II. Calibration and Bayes on Black–Scholes

| Lecture | Sections | New material |
| --- | --- | --- |
| 6. Marking a product the smile does not quote | 6.1 = L8 8.1 rewritten; 6.2 = L8 8.2; 6.3 = L8 8.3–8.6 (definitions); 6.4 = L8 8.7 (idea and HKDE example); 6.5 = L8 8.8; 6.6 = L8 8.9; 6.7 = L8 8.10; 6.8 new | 6.1: the marking procedure, step by step, without roles; 6.7: baselines that fix what the quotes do not, with sourced wording; 6.8: the argument of the course |
| 7. Calibration as estimation | 7.1–7.7 = L9 9.1–9.7; 7.8 = L9 9.9; 7.9 = L9 9.10; 7.10 = L9 9.12 | None; L9 9.8 and 9.11 move to Lecture 11 |
| 8. Bayesian calibration | 8.1–8.6 = L10 10.1–10.6; 8.7 = L10 10.9 (method); 8.8 = L10 10.7; 8.9 = L13 13.7 (theory); 8.10 new; 8.11 new; 8.12 = L10 10.12 | 8.9: importance sampling as an exact method; 8.10: pooling across names (`images/v2-pooling.png`); 8.11: where the posterior and the classical answer part |
| 9. Amortized inference | 9.1–9.6 = L12 12.1–12.6; 9.7 = L13 13.3 (theory, toy); 9.8 = L13 13.4 (theory, toy) | None |

## Part III. Heston and the BayesFlow workflow

| Lecture | Sections | New material |
| --- | --- | --- |
| 10. The Heston model | 10.1–10.5 = L6 6.1–6.5; 10.6 = L6 6.6 (first part); 10.7 = L6 6.8; 10.8 = L6 6.9; 10.9 = L8 8.4 (Heston); 10.10 = L8 8.5 (Heston); 10.11 = L6 6.11; 10.12 = L6 6.6 (rest) and 6.7 | Heston's role: the model where every amortized result has an exact check; jumps in one paragraph |
| 11. Calibrating Heston | 11.1 = L6 6.10; 11.2 = L8 8.7 (Heston profile); 11.3 = L10 10.8; 11.4 new; 11.5 = L10 10.10; 11.6 = L9 9.8 (Heston); 11.7 = L9 9.11 | 11.4: on SPY the exact posterior and the classical interval agree (`tutorial_final/checks/heston_checks.json`) |
| 12. The amortized workflow on Heston | 12.1 = L12 12.7 (Heston); 12.2 = L12 12.8–12.9 (Heston); 12.3 = L13 13.2; 12.4–12.8 = L13 13.3–13.8 (Heston); 12.9 new; 12.10 = L12 12.10 | BayesFlow code from `inference.py`; 12.9: changing the prior and the noise without retraining (slot 6) |

## Part IV. Rough Bergomi: the full examples

| Lecture | Sections | New material |
| --- | --- | --- |
| 13. The rough Bergomi model | 13.1–13.15 = L7; with L8 8.3, L10 10.11, the rough part of L8 8.7 | Identification when ξ0 is uncertain or the quotes are few (`tutorial_final/checks/rough_checks.json`) |
| 14. Why the exact posterior is expensive | 14.1–14.3 = L11 11.1–11.3; 14.4 = L11 11.4 rewritten; 14.5 = L11 11.5 | Who needs many posteriors; importance sampling from a Gauss–Newton proposal and its cost |
| 15. A liquid smile: SPY | 15.1 = L12 12.7–12.8 (rough); 15.2 = L13 (rough results); 15.3 = L14 14.4; 15.4 new; 15.5 = L14 14.5; 15.6 new; 15.7 = L14 14.6 | 15.4: posterior against Gauss–Newton on SPY; 15.6: the long end |
| 16. A thin chain: EWG | new; 16.7 = L13 13.9 and L14 14.8 | The thin example, its procedure, the withdrawn first attempt, coverage (S10, S10-E), EWG's posterior (S11), model error; 16.8: the argument claim by claim |

Version 1's Lecture 14 leaves the series. Its status tables (14.1–14.2) become [status.md](status.md); its map to
the tutorial (14.7, 14.9) is in `tutorial_final/curriculum/`.
