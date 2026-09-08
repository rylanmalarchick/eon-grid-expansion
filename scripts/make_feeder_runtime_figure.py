"""Draw the per-feeder Layer A runtime figure from the current records.

Run from workspace/:
    python scripts/make_feeder_runtime_figure.py --records RESULTS/*.jsonl
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from eon.viz.hardness_collapse import _load, plot_feeder_runtimes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("make_feeder_runtime_figure")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", nargs="+", required=True)
    parser.add_argument("--out-dir", default="experiments/figures")
    args = parser.parse_args()

    records = _load([Path(p) for p in args.records])
    logger.info("%d records", len(records))
    written = plot_feeder_runtimes(records, Path(args.out_dir) / "feeder_runtimes")
    logger.info("wrote %s (+ .pdf)", written)


if __name__ == "__main__":
    main()
