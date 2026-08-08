"""Does the constrained mixer's advantage depend on how much of its subspace the shots cover?

One panel, one measure: solution excess against shot coverage. Both arms are
scored on the same penalized vector, so the two curves are directly comparable
and 0 is the exact optimum.

The figure exists to show a NON-result honestly. If the curves lie on top of
each other, the constrained mixer is buying feasibility rather than search, and
a chart makes that far harder to overstate than a table does. Each seed is drawn
as its own marker rather than hidden inside a mean, because the spread between
seeds is the reason we decline to claim the left-hand point.

Coverage runs high-to-low left-to-right, so "further right = harder to sample
exhaustively" reads in the direction of the argument.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Two hues from the categorical trio validated for this project (blue, green,
# amber). Blue and amber were the best-separated pair of that set, so a
# two-series chart uses those. Each series also carries its own marker, so
# identity never rests on hue alone.
_RANDOM = "#eda100"
_COP = "#2a78d6"
_TEXT = "#0b0b0b"
_MUTED = "#52514e"
_GRID = "#e5e4de"
_EDGE = "#c3c2b7"


def load_records(paths: list[Path]) -> list[dict]:
    records: list[dict] = []
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def _series(records: list[dict], key: str) -> dict[int, list[tuple[float, float]]]:
    """weight -> [(coverage, excess), ...] one entry per seed."""
    out: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for record in records:
        excess = record["excess_over_exact"].get(key)
        if excess is None:
            continue
        subspace = record["subspace"]
        out[int(subspace["hamming_weight"])].append(
            (100.0 * float(subspace["expected_coverage_fraction"]), float(excess))
        )
    return out


def plot_coverage(records: list[dict], out_stem: Path, *, depth: int = 2) -> Path:
    if not records:
        raise ValueError("no records; nothing to plot")

    arms = (
        ("random draw from the same subspace", _RANDOM, "s", _series(records, "random_median")),
        (f"constrained QAOA (p={depth})", _COP, "o", _series(records, f"cop_p{depth}")),
    )

    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    seed_count = 0
    for label, color, marker, by_weight in arms:
        if not by_weight:
            continue
        weights = sorted(by_weight)
        means = [
            (
                sum(c for c, _ in by_weight[w]) / len(by_weight[w]),
                sum(e for _, e in by_weight[w]) / len(by_weight[w]),
            )
            for w in weights
        ]
        axis.plot(
            [c for c, _ in means], [e for _, e in means],
            color=color, linewidth=2.0, marker=marker, markersize=8,
            markeredgecolor="#fcfcfb", markeredgewidth=1.0, label=label, zorder=3,
        )
        for w in weights:
            seed_count = max(seed_count, len(by_weight[w]))
            axis.scatter(
                [c for c, _ in by_weight[w]], [e for _, e in by_weight[w]],
                s=26, color=color, alpha=0.45, zorder=2, linewidths=0,
            )

    axis.set_xscale("log")
    axis.invert_xaxis()  # exhaustive sampling on the left, sparse on the right
    axis.set_xlabel(
        "expected shot coverage of the feasible subspace (%, log scale)",
        color=_MUTED, fontsize=9,
    )
    axis.set_ylabel("excess over the exact optimum (0 = optimal)", color=_MUTED, fontsize=9)
    axis.grid(True, color=_GRID, linewidth=0.6, zorder=0)
    axis.set_axisbelow(True)
    axis.tick_params(colors=_MUTED, labelsize=8)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axis.spines[spine].set_color(_EDGE)
    axis.legend(frameon=False, fontsize=8.5, labelcolor=_MUTED, loc="upper left")

    weights = sorted({int(r["subspace"]["hamming_weight"]) for r in records})
    # The title must survive the data changing under it. The curves lie on top
    # of each other everywhere except the sparsest point, where the two seeds
    # disagree with each other -- so the title says exactly that rather than
    # asserting either "no difference" or "an advantage".
    axis.set_title(
        "Indistinguishable from a random draw until the subspace is barely sampled,\n"
        "where the seeds disagree",
        color=_TEXT, fontsize=11, pad=12,
    )
    axis.text(
        0.5, -0.20,
        f"Lines are the mean over {seed_count} seed(s); faint markers are the individual seeds. "
        f"Cardinality weights {', '.join(str(w) for w in weights)} at 1024 shots. "
        "Both arms scored on the same penalized cost vector.",
        transform=axis.transAxes, ha="center", va="top", fontsize=7.4, color=_MUTED,
    )

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    png = out_stem.with_suffix(".png")
    figure.savefig(png, dpi=200, bbox_inches="tight")
    figure.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return png
