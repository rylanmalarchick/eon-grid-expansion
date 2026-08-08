"""Render the S3 coverage figure from random-feasible control records.

Run from workspace/:
    python scripts/make_coverage_figure.py \
      --records experiments/results/s3_coverage_sweep/*.jsonl \
      --out-dir experiments/figures
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from eon.viz.coverage import load_records, plot_coverage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("make_coverage_figure")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", nargs="+", required=True)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--out-dir", default="experiments/figures")
    args = parser.parse_args()

    records = load_records([Path(p) for p in args.records])
    seeds = {r["instance_id"] for r in records}
    logger.info("%d records across %d instances", len(records), len(seeds))
    out = plot_coverage(records, Path(args.out_dir) / "s3_coverage", depth=args.depth)
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
