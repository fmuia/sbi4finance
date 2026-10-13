"""Create a new synthetic SVJ teaching bundle with the common inference runner."""
import argparse
from pathlib import Path
from sbi4finance.svj.teaching import write_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/svj"))
    args = parser.parse_args()
    write_bundle(args.out, args.checkpoint)
    print(args.out / "bundle.json")


if __name__ == "__main__":
    main()
