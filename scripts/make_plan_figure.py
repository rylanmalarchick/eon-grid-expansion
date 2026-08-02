"""Render the E.ON-facing plan figure from a congestion_metrics.py record.

Run from workspace/:
    python scripts/make_plan_figure.py \
        --metrics experiments/results/congestion_seed7/metrics.json
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from eon.instances.distribution_feeders import load_distribution_feeder
from eon.viz.network_plan import plot_plan_figure

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("make_plan_figure")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--feeder", default="ieee33")
    parser.add_argument("--out-dir", default="experiments/figures")
    args = parser.parse_args()

    net = load_distribution_feeder(args.feeder)
    path = plot_plan_figure(
        Path(args.metrics), net, Path(args.out_dir) / f"plan_{args.feeder}"
    )
    logger.info("wrote %s (+ .pdf)", path)


if __name__ == "__main__":
    main()
