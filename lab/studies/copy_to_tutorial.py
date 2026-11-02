"""Compatibility entry point: numerical code is now edited in tutorial/, never copied.

Use `pixi run --locked -e core runtime-check` to check lab reuse. This historical
command remains read-only, so old automation cannot overwrite tutorial edits.
"""

import argparse

from runtime_boundary import differences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run the read-only runtime boundary check")
    parser.add_argument("--add", nargs="+", metavar="MODULE.NAME",
                        help="retired: add implementations to the tutorial-owned runtime directly")
    args = parser.parse_args()
    if args.add:
        parser.error("--add is retired. Edit tutorial/src/sbi4f_tutorial/ directly, import the implementation "
                     "from the lab when needed, then run pixi run --locked -e core runtime-check.")
    print("copy-to-tutorial is retired; checking tutorial-owned runtime reuse (no files are written).")
    failures = differences()
    if failures:
        raise SystemExit("\n".join(failures))
    print("runtime current; use runtime-check in new workflows")


if __name__ == "__main__":
    main()
