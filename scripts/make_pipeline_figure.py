"""Render the one-page pipeline visual.

Run from workspace/:
    python scripts/make_pipeline_figure.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from eon.viz.pipeline import plot_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("make_pipeline_figure")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="experiments/figures")
    args = parser.parse_args()
    path = plot_pipeline(Path(args.out_dir) / "pipeline")
    logger.info("wrote %s (+ .pdf)", path)


if __name__ == "__main__":
    main()
