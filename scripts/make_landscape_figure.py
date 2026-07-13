"""Render the P3 figures from a qaoa_landscape.py JSONL run.

Run from workspace/:
    python scripts/make_landscape_figure.py \
        --results experiments/results/qaoa_p3/landscape.jsonl \
        [--out-dir experiments/figures]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from eon.viz.landscape import load_records, plot_metric_vs_depth, plot_p1_heatmaps

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("make_landscape_figure")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--out-dir", default="experiments/figures")
    parser.add_argument(
        "--heatmap-instance",
        default="",
        help="Instance id for the p=1 heatmaps (default: first with grids).",
    )
    args = parser.parse_args()

    results, grids = load_records(Path(args.results))
    out_dir = Path(args.out_dir)
    heatmap_instance = args.heatmap_instance or (grids[0]["instance_id"] if grids else "")

    written = []
    if heatmap_instance:
        written.append(
            plot_p1_heatmaps(grids, heatmap_instance, out_dir / "p1_landscapes")
        )
    written.append(
        plot_metric_vs_depth(
            results,
            value_key="best_sample_excess",
            ylabel="best feasible sample, relative excess over exact optimum",
            title="Best sampled energy vs depth (statevector simulation, 1024 shots)",
            out_stem=out_dir / "excess_vs_depth",
            logy=True,
        )
    )
    written.append(
        plot_metric_vs_depth(
            results,
            value_key="feasible_fraction",
            ylabel="feasible sample fraction",
            title="Feasibility of sampled states vs depth (simulation)",
            out_stem=out_dir / "feasible_fraction_vs_depth",
        )
    )
    for path in written:
        logger.info("wrote %s (+ .pdf)", path)


if __name__ == "__main__":
    main()
