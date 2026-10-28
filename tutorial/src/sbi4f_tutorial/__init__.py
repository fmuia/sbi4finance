"""The runtime of the tutorial notebooks: everything they compute live, and the loaders of what they show cached.

The tutorial depends on nothing else in the repository. These modules are small copies of the lab's code
(lab/src/sbi4finance), written to be read; the lab's tests check that the copies return the same numbers.
The notebooks read their inputs from tutorial/assets/ only (paths.py).
"""
