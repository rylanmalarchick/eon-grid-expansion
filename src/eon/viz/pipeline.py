"""The one-page pipeline visual required by the deliverable packet.

One flow, left to right: a defensible distribution-planning model produces a
plan; a reduced binary surrogate over that plan's neighborhood is what any
quantum method actually sees; three independent checkers score it; and the
results reaggregate into a feasible plan carrying a certified optimality gap.

Drawn with matplotlib rather than a graph library so the figure has no
dependency the reviewer cannot already run.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

_BLUE = "#2a78d6"
_AQUA = "#1baf7a"
_YELLOW = "#eda100"
_TEXT = "#0b0b0b"
_MUTED = "#52514e"
_EDGE = "#c3c2b7"
_SURFACE = "#fcfcfb"


def _box(
    axis: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    body: str,
    *,
    accent: str,
) -> tuple[float, float]:
    x, y = xy
    axis.add_patch(
        FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.6, edgecolor=accent, facecolor=_SURFACE, zorder=2,
        )
    )
    axis.text(
        x + width / 2, y + height - 0.052, title,
        ha="center", va="top", fontsize=9.5, color=_TEXT, weight="bold", zorder=3,
    )
    axis.text(
        x + width / 2, y + height - 0.115, body,
        ha="center", va="top", fontsize=7.6, color=_MUTED, zorder=3, linespacing=1.5,
    )
    return x + width, y + height / 2


def _arrow(axis: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=17,
            linewidth=1.7, color=_MUTED, zorder=1,
            shrinkA=2, shrinkB=2,
        )
    )


def plot_pipeline(out_stem: Path) -> Path:
    figure, axis = plt.subplots(figsize=(12.6, 5.6))
    axis.set_xlim(0, 1)
    # Trim to the used band so the one-page visual has no dead margin.
    axis.set_ylim(-0.06, 0.90)
    axis.axis("off")

    _box(
        axis, (0.015, 0.345), 0.20, 0.285,
        "Layer A — planning model",
        "LinDistFlow expansion +\nreconfiguration under radiality,\n"
        "multi-scenario, MILP (Gurobi).\nThe credibility-bearing object.",
        accent=_BLUE,
    )
    _box(
        axis, (0.255, 0.345), 0.19, 0.285,
        "Layer B — surrogate",
        "Reduced binary QUBO over the\nplan's neighborhood.\n"
        "NISQ-sized; benchmark only,\nnever 'the' problem.",
        accent=_BLUE,
    )

    checkers = [
        (
            0.485, 0.630, "Classical baselines",
            "Gurobi (gap + time),\nsimulated annealing", _MUTED,
        ),
        (
            0.485, 0.345, "Tensor-network control",
            "Exact contraction (GTN) +\nchi-swept MPS protocol", _AQUA,
        ),
        (
            0.485, 0.060, "Quantum subroutine",
            "Constrained-mixer QAOA vs\npenalty QAOA, simulator", _YELLOW,
        ),
    ]
    for x, y, title, body, accent in checkers:
        _box(axis, (x, y), 0.215, 0.225, title, body, accent=accent)

    _box(
        axis, (0.745, 0.345), 0.24, 0.285,
        "Reaggregation + certificate",
        "Feasible line set rescored on\nLayer A, with a certified\n"
        "lower/upper bound gap\n(classical duality, not advantage).",
        accent=_BLUE,
    )

    _arrow(axis, (0.215, 0.4875), (0.255, 0.4875))
    for _, y, *_ in checkers:
        _arrow(axis, (0.445, 0.4875), (0.485, y + 0.1125))
        _arrow(axis, (0.700, y + 0.1125), (0.745, 0.4875))

    figure.suptitle(
        "Two-layer pipeline: a defensible planning model, a NISQ-sized surrogate, "
        "three independent checkers, one feasible plan",
        fontsize=11.5, color=_TEXT, y=0.985,
    )
    axis.text(
        0.015, -0.035,
        "Hardness is established at Layer A; the surrogate is what a quantum method sees. "
        "Every checker scores the same compiled cost object.",
        ha="left", va="bottom", fontsize=7.4, color=_MUTED,
    )

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    png = out_stem.with_suffix(".png")
    figure.savefig(png, dpi=200, bbox_inches="tight")
    figure.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return png
