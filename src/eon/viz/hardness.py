"""The outcome-1 figure: hard feeder instances across variable counts.

Two panels, one measure each (a MIP gap and a coupling ratio share no axis):

  A  Layer A optimality gap after a 1800 s solve, per instance size, with the
     reconfiguration-OFF control. This is the hardness claim: with the switching
     lever the solver is time-limited at a large gap; without it, the same
     solver closes the same feeder in seconds.
  B  coupling-to-field ratio |J|/|h| of the Layer B surrogate, one point per
     seed. This is the diversity claim, and it is where the family's spread
     shows -- including the jittered seed whose surrogate is nearly decoupled
     under a Layer A problem that is still time-limited.

Panel B carries the more interesting honesty: hardness at Layer A does not imply
hardness at Layer B, and the figure shows both rather than averaging them into
one reassuring curve.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_BLUE = "#2a78d6"      # reconfiguration ON (the hard configuration)
_EXISTING = "#c3c2b7"  # reconfiguration OFF (the control)
_TEXT = "#0b0b0b"
_MUTED = "#52514e"
_GRID = "#e5e4de"

# Fixed categorical order for the per-seed series, validated as a set against
# the light chart surface: lightness band, chroma floor, colour-vision-deficiency
# separation and normal-vision floor all pass. The contrast check warns, which
# obligates visible labels -- so each series is direct-labeled as well as
# legended, and carries its own marker so identity never rests on hue alone.
_SEED_COLORS = ("#2a78d6", "#1baf7a", "#eda100")
_SEED_MARKERS = ("o", "s", "^")


def load_records(paths: list[Path]) -> list[dict]:
    records: list[dict] = []
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def _deduplicate(records: list[dict]) -> list[dict]:
    """Key on the ACTUAL variable count, never the requested neighborhood.

    A neighborhood wider than the candidate set is clamped, so two requested
    sizes can land on one real size; keeping both would draw a duplicate as if
    it were an extra instance.
    """
    best: dict[tuple[int, int, bool], dict] = {}
    for record in records:
        key = (
            int(record["coupling"]["variable_count"]),
            int(record["seed"]),
            bool(record["enable_reconfiguration"]),
        )
        best[key] = record
    return list(best.values())


def _style(axis: plt.Axes, *, title: str, ylabel: str, xlabel: str) -> None:
    axis.set_title(title, color=_TEXT, fontsize=10, pad=10)
    axis.set_ylabel(ylabel, color=_MUTED, fontsize=9)
    axis.set_xlabel(xlabel, color=_MUTED, fontsize=9)
    axis.grid(True, axis="y", color=_GRID, linewidth=0.6, zorder=0)
    axis.set_axisbelow(True)
    axis.tick_params(colors=_MUTED, labelsize=8)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axis.spines[spine].set_color(_EXISTING)


def plot_hardness(records: list[dict], out_stem: Path) -> Path:
    rows = _deduplicate(records)
    on = [r for r in rows if r["enable_reconfiguration"]]
    off = [r for r in rows if not r["enable_reconfiguration"]]
    if not on:
        raise ValueError("no reconfiguration-ON records; nothing to plot")

    figure, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    figure.subplots_adjust(wspace=0.28)

    # --- Panel A: Layer A gap, ON vs OFF ---------------------------------
    for subset, color, label, marker in (
        (off, _EXISTING, "reconfiguration OFF (control)", "s"),
        (on, _BLUE, "reconfiguration ON", "o"),
    ):
        if not subset:
            continue
        xs = [r["coupling"]["variable_count"] for r in subset]
        ys = [r["layer_a"]["mip_gap"] for r in subset]
        left.scatter(
            xs, ys, s=58, color=color, marker=marker, label=label,
            zorder=3, edgecolors="#fcfcfb", linewidths=1.0,
        )
    left.set_ylim(-0.04, 1.0)
    _style(
        left,
        title="Layer A optimality gap after a 1800 s solve",
        ylabel="MIP gap",
        xlabel="surrogate variable count",
    )
    left.legend(frameon=False, fontsize=8, labelcolor=_MUTED, loc="center right")

    # Median solve time per series says what the gap alone does not: the
    # control is not merely tighter, it terminates.
    if off:
        median_off = sorted(r["layer_a"]["runtime_s"] for r in off)[len(off) // 2]
        left.text(
            0.02, 0.06,
            f"control closes in ~{median_off:.0f} s; ON is time-limited at 1800 s",
            transform=left.transAxes, fontsize=7.6, color=_MUTED,
        )

    # --- Panel B: coupling ratio per seed --------------------------------
    by_seed: dict[int, list[dict]] = defaultdict(list)
    for record in on:
        by_seed[int(record["seed"])].append(record)
    label_positions: list[float] = []
    for index, seed in enumerate(sorted(by_seed)):
        subset = sorted(by_seed[seed], key=lambda r: r["coupling"]["variable_count"])
        xs = [r["coupling"]["variable_count"] for r in subset]
        ys = [r["coupling"]["coupling_field_ratio"] for r in subset]
        color = _SEED_COLORS[index % len(_SEED_COLORS)]
        right.plot(
            xs, ys,
            marker=_SEED_MARKERS[index % len(_SEED_MARKERS)],
            markersize=7, linewidth=1.9, color=color,
            label=f"seed {seed}", zorder=3,
            markeredgecolor="#fcfcfb", markeredgewidth=0.9,
        )
        # Direct label at the right end, in text ink rather than the series
        # colour; the marker beside it carries identity. Series that converge
        # (two seeds both near-decoupled) would print their labels on top of
        # each other, so stagger any that land within a few percent of the axis.
        offset = 0.0
        low, high = right.get_ylim()
        # Stagger AWAY from the nearer axis edge, or the displaced label
        # clips off the figure instead of merely overlapping.
        direction = 1.0 if ys[-1] < 0.5 * (low + high) else -1.0
        for placed in label_positions:
            if abs(placed - ys[-1]) < 0.06 * max(high - low, 1.0):
                offset += direction * 11.0
        label_positions.append(ys[-1])
        right.annotate(
            f"seed {seed}",
            xy=(xs[-1], ys[-1]), xytext=(6, offset), textcoords="offset points",
            va="center", fontsize=7.8, color=_MUTED, zorder=4,
        )
    _style(
        right,
        title="Layer B coupling strength |J|/|h| (reconfiguration ON)",
        ylabel="coupling-to-field ratio",
        xlabel="surrogate variable count",
    )
    right.legend(frameon=False, fontsize=8, labelcolor=_MUTED)

    decoupled = [r for r in on if r["coupling"]["coupling_field_ratio"] < 2.0]
    if decoupled:
        right.text(
            0.02, 0.42,
            f"{len(decoupled)} of {len(on)} runs are near-decoupled\n"
            "(|J|/|h| ~ 1) despite a time-limited Layer A",
            transform=right.transAxes, fontsize=7.6, color=_MUTED,
        )

    # Only the sizes actually run are meaningful ticks; the default locator
    # invents 17, 18, 19 where no instance exists.
    sizes = sorted({r["coupling"]["variable_count"] for r in rows})
    for axis in (left, right):
        axis.set_xticks(sizes)
        axis.set_xlim(min(sizes) - 1.5, max(sizes) + 2.5)

    sizes = sorted({r["coupling"]["variable_count"] for r in on})
    figure.suptitle(
        f"IEEE 33 instance family: time-limited at every size "
        f"(n = {', '.join(str(s) for s in sizes)}), and Layer B hardness varies within it",
        color=_TEXT, fontsize=11.5, y=1.02,
    )

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    png = out_stem.with_suffix(".png")
    figure.savefig(png, dpi=200, bbox_inches="tight")
    figure.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return png
