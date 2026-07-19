"""P3 figures from scripts/qaoa_landscape.py JSONL output.

Three figures, matplotlib, light surface:
- p=1 (beta, gamma) expected-energy heatmaps, cop vs vanilla, one feeder
  instance (per-panel colorbars: vanilla's landscape is penalty-dominated and
  lives on a different scale -- that asymmetry is part of the result).
- best feasible sampled energy excess vs depth p, per instance, series =
  algorithm (fixed colors: cop blue #2a78d6, vanilla aqua #1baf7a,
  warm-started yellow #eda100 -- identity never re-mapped across panels).
- feasible-sample fraction vs depth p, same layout.

Captions state simulator-only; nothing claims hardware or advantage.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ALGORITHM_COLORS = {
    "cop": "#2a78d6",
    "vanilla": "#1baf7a",
    "vanilla_warm": "#eda100",
}
ALGORITHM_LABELS = {
    "cop": "cop-QAOA (XY ring)",
    "vanilla": "penalty QAOA",
    "vanilla_warm": "penalty QAOA + warm start",
}
_TEXT = "#0b0b0b"
_MUTED = "#52514e"


def load_records(path: Path) -> tuple[list[dict], list[dict]]:
    """Returns (result_records, grid_records)."""
    results: list[dict] = []
    grids: list[dict] = []
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            (grids if "p1_grid" in record else results).append(record)
    return results, grids


def plot_p1_heatmaps(grids: list[dict], instance_id: str, out_stem: Path) -> Path:
    selected = [g for g in grids if g["instance_id"] == instance_id]
    if not selected:
        raise ValueError(f"no p1 grids recorded for {instance_id}")
    fig, axes = plt.subplots(1, len(selected), figsize=(5.2 * len(selected), 4.2))
    axes = np.atleast_1d(axes)
    for axis, grid_record in zip(axes, selected, strict=True):
        grid = grid_record["p1_grid"]
        energies = np.asarray(grid["energies"])
        betas = np.asarray(grid["betas"])
        gammas = np.asarray(grid["gammas"])
        mesh = axis.pcolormesh(gammas, betas, energies, cmap="Blues", shading="nearest")
        best = np.unravel_index(np.argmin(energies), energies.shape)
        axis.plot(
            gammas[best[1]],
            betas[best[0]],
            marker="o",
            markersize=9,
            markerfacecolor="none",
            markeredgecolor=_TEXT,
            markeredgewidth=2,
        )
        axis.set_title(
            ALGORITHM_LABELS.get(grid_record["algorithm"], grid_record["algorithm"]),
            color=_TEXT,
            fontsize=11,
        )
        axis.set_xlabel(r"$\gamma$", color=_TEXT)
        axis.set_ylabel(r"$\beta$", color=_TEXT)
        axis.tick_params(colors=_MUTED, labelsize=8)
        colorbar = fig.colorbar(mesh, ax=axis, shrink=0.85)
        colorbar.set_label("expected energy (p=1)", color=_MUTED, fontsize=8)
        colorbar.ax.tick_params(colors=_MUTED, labelsize=7)
    fig.suptitle(
        f"p=1 QAOA landscapes, {instance_id} (statevector simulation; "
        "circled = grid optimum; scales differ by construction)",
        color=_TEXT,
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, out_stem)


def _per_instance_series(
    results: list[dict], value_key: str
) -> dict[str, dict[str, list[tuple[int, float]]]]:
    series: dict[str, dict[str, list[tuple[int, float]]]] = {}
    for record in results:
        value = record.get(value_key)
        if value is None:
            continue
        series.setdefault(record["instance_id"], {}).setdefault(
            record["algorithm"], []
        ).append((record["p"], float(value)))
    return series


def plot_metric_vs_depth(
    results: list[dict],
    *,
    value_key: str,
    ylabel: str,
    title: str,
    out_stem: Path,
    logy: bool = False,
) -> Path:
    series = _per_instance_series(results, value_key)
    instances = sorted(series)
    fig, axes = plt.subplots(
        1, len(instances), figsize=(3.6 * len(instances), 3.6), sharey=True
    )
    axes = np.atleast_1d(axes)
    # Series can coincide exactly (e.g. two algorithms both at excess 0), so
    # each algorithm gets a fixed x-dodge and its own marker -- identity is
    # never carried by color alone or hidden by overplotting.
    dodge = {"cop": -0.07, "vanilla": 0.0, "vanilla_warm": 0.07}
    markers = {"cop": "o", "vanilla": "s", "vanilla_warm": "^"}
    import matplotlib.transforms as mtransforms

    for axis, instance_id in zip(axes, instances, strict=True):
        all_depths = sorted({p for pts in series[instance_id].values() for p, _ in pts})
        for algorithm in ("cop", "vanilla", "vanilla_warm"):
            points = sorted(series[instance_id].get(algorithm, []))
            if not points:
                continue
            by_depth = dict(points)
            # Never draw a line across a missing depth: a gap means "no
            # feasible sample there", the WORST outcome, not missing-at-random.
            # Split into runs of consecutive present depths.
            segments: list[list[int]] = [[]]
            for depth in all_depths:
                if depth in by_depth:
                    segments[-1].append(depth)
                elif segments[-1]:
                    segments.append([])
            labeled = False
            for segment in segments:
                if not segment:
                    continue
                axis.plot(
                    [p + dodge[algorithm] for p in segment],
                    [by_depth[p] for p in segment],
                    color=ALGORITHM_COLORS[algorithm],
                    linewidth=2,
                    marker=markers[algorithm],
                    markersize=6,
                    label=ALGORITHM_LABELS[algorithm] if not labeled else None,
                )
                labeled = True
            missing = [p for p in all_depths if p not in by_depth]
            if missing:
                blended = mtransforms.blended_transform_factory(
                    axis.transData, axis.transAxes
                )
                axis.scatter(
                    [p + dodge[algorithm] for p in missing],
                    [0.97] * len(missing),
                    transform=blended,
                    marker="x",
                    s=45,
                    color=ALGORITHM_COLORS[algorithm],
                    linewidths=2,
                    zorder=5,
                )
        if logy:
            axis.set_yscale("symlog", linthresh=1e-3)
        axis.set_title(instance_id, color=_TEXT, fontsize=8)
        axis.set_xlabel("depth p", color=_TEXT, fontsize=9)
        depth_ticks = sorted({p for pts in series[instance_id].values() for p, _ in pts})
        axis.set_xticks(depth_ticks)
        axis.set_xlim(min(depth_ticks) - 0.35, max(depth_ticks) + 0.35)
        axis.grid(True, color="#e5e4de", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.tick_params(colors=_MUTED, labelsize=8)
        for spine in axis.spines.values():
            spine.set_color("#c3c2b7")
    axes[0].set_ylabel(ylabel, color=_TEXT, fontsize=8)
    legend_handles = [
        plt.Line2D(
            [], [],
            color=ALGORITHM_COLORS[algorithm],
            linewidth=2,
            marker=markers[algorithm],
            markersize=6,
            label=ALGORITHM_LABELS[algorithm],
        )
        for algorithm in ("cop", "vanilla", "vanilla_warm")
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        fontsize=8,
    )
    fig.suptitle(title, color=_TEXT, fontsize=10, y=1.10)
    fig.text(
        0.01,
        -0.04,
        "x at panel top = no feasible sample at that depth (worst outcome, "
        "not missing data)",
        color=_MUTED,
        fontsize=7,
    )
    fig.tight_layout()
    return _save(fig, out_stem)


def _save(fig: plt.Figure, out_stem: Path) -> Path:
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    png = out_stem.with_suffix(".png")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return png
