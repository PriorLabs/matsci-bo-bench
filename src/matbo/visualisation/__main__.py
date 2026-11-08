# Copyright (c) Prior Labs GmbH 2026. Licensed under the Apache License, Version 2.0.
"""Render Figures 1-2 and D.1-D.9 from reference_results/ as PDF and PNG."""

import argparse
from pathlib import Path

from . import FIGURES, style


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figures", nargs="*",
                        help="figure numbers: 1 2 D.1 ... D.9 (default: all)")
    parser.add_argument("--out", type=Path, default=Path("figures/paper"))
    args = parser.parse_args()
    unknown = set(args.figures) - FIGURES.keys()
    if unknown:
        parser.error(f"unknown figures: {sorted(unknown)}; choose from {list(FIGURES)}")
    style.plt.switch_backend("Agg")
    style.use()
    for number in args.figures or FIGURES:
        fig = FIGURES[number]()
        for path in style.save(fig, args.out / f"figure_{number.replace('.', '_')}"):
            print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
