"""Draw the hardness-collapse figure from pre-fix and post-fix Layer A records.

Run from workspace/:
    python scripts/make_collapse_figure.py --before OLD/*.jsonl --after NEW/*.jsonl
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from eon.viz.hardness_collapse import _load, plot_collapse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("make_collapse_figure")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", nargs="+", required=True)
    parser.add_argument("--after", nargs="+", required=True)
    parser.add_argument("--out-dir", default="experiments/figures")
    args = parser.parse_args()

    before = _load([Path(p) for p in args.before])
    after = _load([Path(p) for p in args.after])
    logger.info("%d pre-fix records, %d post-fix", len(before), len(after))
    written = plot_collapse(before, after, Path(args.out_dir) / "hardness_collapse")
    logger.info("wrote %s (+ .pdf)", written)


if __name__ == "__main__":
    main()
