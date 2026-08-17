"""Write one unaligned MovePort segment in its native-rate NPZ schema."""
from __future__ import annotations

import argparse
from pathlib import Path

from posesim.data.moveport import load_native_segment, write_native_segment


def path_component(value):
    if not value or value in (".", "..") or "/" in value or "\\" in value:
        raise argparse.ArgumentTypeError("must be a safe single path component")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/raw/moveport")
    parser.add_argument("--out", required=True)
    parser.add_argument("--subject", type=path_component, required=True)
    parser.add_argument("--activity", type=path_component, required=True)
    parser.add_argument("--segment", type=path_component, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    native = load_native_segment(args.root, args.subject, args.activity, args.segment)
    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    destination = (output / f"{native.subject}_{native.activity}_{native.segment}.npz").resolve()
    if destination.parent != output:
        parser.error("destination must remain under --out")
    if destination.exists() and not args.overwrite:
        parser.error(f"destination exists: {destination}; pass --overwrite to replace it")
    write_native_segment(native, destination)


if __name__ == "__main__":
    main()
