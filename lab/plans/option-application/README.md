# Option application on SPX, in three layers

The same account at three depths. Each layer starts from the goals, the audience and the
practitioner's point of view, follows the same ten numbered steps, and ends by returning
to the goals, the audience and the practitioner. Step 6 in one layer is step 6 in the
others, so a reader can go one layer deeper on any step.

| Layer | What it gives | Reading time |
| --- | --- | --- |
| [1. Overview](1-overview.md) | Every step in a few sentences, with the key equations and figures | about 10 minutes |
| [2. Walkthrough](2-walkthrough.md) | Each step with its reasoning, its formulas and the numbers behind each choice | about 30 minutes |
| [3. Full detail](3-full-detail.md) | Every definition, derivation, setting and check; the results, their sources, and what remains | reference |

The [glossary](glossary.md) explains the finance and statistics terms the layers use. The
symbols are those of the tutorial's Heston notebook,
`lab/notebooks/heston_spx_calibration.py`, and every quantity is defined by a formula
where it first appears. The model is Heston; rough Bergomi, the next model, follows the
same steps once its simulations are complete.
Textbook background (Itô calculus, Black–Scholes, Heston's Fourier formula, the rough
Bergomi simulation scheme) is in the [course](../../../course/README.md).

All results are as of 13 November 2026 and are those of the Heston notebook. The figures
in [`images/`](images/) are drawn by
[`figure-sources/make_figures.py`](figure-sources/make_figures.py) from recorded results
([`figure-sources/data.json`](figure-sources/data.json)) and from the notebook's own
numbers ([`figure-sources/notebook.json`](figure-sources/notebook.json), recomputed by
[`figure-sources/notebook_numbers.py`](figure-sources/notebook_numbers.py)), with the path
and SHA-256 of every source; [Layer 3](3-full-detail.md#evidence-and-figures) lists the
sources.

## The ten steps

1. What has a price, and what does not
2. Calibration gives one value and hides the rest
3. The posterior turns one value into a distribution
4. The model
5. Which quotes
6. How quotes differ from the model
7. Computing the posterior: simulation-based inference
8. Checking it
9. From the posterior to the products
10. What the results show, and what they do not
