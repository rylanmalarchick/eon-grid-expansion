"""Render the outcome-1 hardness figure from S1-proper sweep records.

Run from workspace/:
    python scripts/make_hardness_figure.py \
      --on  experiments/results/s1_proper/*.jsonl \
      --off experiments/results/s1_proper_off/*.jsonl \
      --out-dir experiments/figures
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from eon.viz.hardness import load_records, plot_hardness

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("make_hardness_figure")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--on", nargs="+", required=True)
    parser.add_argument("--off", nargs="*", default=[])
    parser.add_argument("--out-dir", default="experiments/figures")
    args = parser.parse_args()

    records = load_records([Path(p) for p in args.on])
    records += load_records([Path(p) for p in args.off])
    logger.info("%d records", len(records))
    out = plot_hardness(records, Path(args.out_dir) / "hardness_family")
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
