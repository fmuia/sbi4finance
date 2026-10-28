"""The runtime of the tutorial notebooks: everything they compute live, and the loaders of what they show cached.

The tutorial depends on nothing else in the repository. Each module holds the functions the notebooks need from
the same-named lab module (lab/src/sbi4finance, lab/checks), copied unchanged and under the same names. The few
lines written for the tutorial (paths.py, and reading the tasks the lab stored) are listed in
lab/tests/test_tutorial_parity.py, which fails when a copy differs from the lab's code. The notebooks read their
inputs from tutorial/assets/ only (paths.py).
"""
