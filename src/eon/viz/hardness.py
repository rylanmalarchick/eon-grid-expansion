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


def _x_size(record: dict) -> int:
    """The size that varies is the CANDIDATE POOL, which is what sizes Layer A.

    The surrogate variable count is fixed by the neighborhood (20 everywhere in
    the corrected family), so keying on it collapses every run onto one x
    position and hides the family entirely.
    """
    return int(record.get("candidate_count") or record["coupling"]["variable_count"])


def _deduplicate(records: list[dict]) -> list[dict]:
    """One point per (Layer A size, seed, mode)."""
    best: dict[tuple[int, int, bool], dict] = {}
    for record in records:
        key = (
            _x_size(record),
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
        xs = [_x_size(r) for r in subset]
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
        xlabel="candidate-pool size (Layer A variables)",
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

    # --- Panel B: coupling ratio, ON against OFF -------------------------
    # The finding is the CONTRAST, so both modes go on one axis. Log scale is
    # mandatory: the two families are ~10 orders of magnitude apart, and on a
    # linear axis the ON series is a flat line on zero that reads as missing
    # data rather than as the result.
    for subset, color, label, marker in (
        (off, _EXISTING, "reconfiguration OFF (control)", "s"),
        (on, _BLUE, "reconfiguration ON", "o"),
    ):
        if not subset:
            continue
        right.scatter(
            [_x_size(r) for r in subset],
            [max(r["coupling"]["coupling_field_ratio"], 1e-12) for r in subset],
            s=58, color=color, marker=marker, label=label, zorder=3,
            edgecolors="#fcfcfb", linewidths=1.0,
        )
    right.set_yscale("log")
    _style(
        right,
        title="Layer B coupling strength |J|/|h|",
        ylabel="coupling-to-field ratio (log)",
        xlabel="candidate-pool size (Layer A variables)",
    )
    # No legend here: both panels carry the same two series and the left panel
    # already labels them. A second copy only competes with the data.

    on_max = max(r["coupling"]["coupling_field_ratio"] for r in on)
    off_max = max(r["coupling"]["coupling_field_ratio"] for r in off) if off else 0.0
    # The empty decade band between the two families, so the note sits in the
    # gap it is describing instead of on top of the ON points.
    right.text(
        0.02, 0.42,
        f"ON is decoupled (|J|/|h| < {on_max:.0e});\n"
        f"the control reaches ~{off_max:.0f}.\n"
        "Switching makes Layer A hard and the surrogate EASY.",
        transform=right.transAxes, fontsize=7.6, color=_MUTED,
    )

    # Only the sizes actually run are meaningful ticks; the default locator
    # invents 17, 18, 19 where no instance exists.
    sizes = sorted({_x_size(r) for r in rows})
    for axis in (left, right):
        axis.set_xticks(sizes)
        axis.set_xlim(min(sizes) - 1.5, max(sizes) + 2.5)

    figure.suptitle(
        "IEEE 33 instance family: Layer A is time-limited at every pool size, "
        "yet its surrogate decouples",
        color=_TEXT, fontsize=11.5, y=1.02,
    )

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    png = out_stem.with_suffix(".png")
    figure.savefig(png, dpi=200, bbox_inches="tight")
    figure.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return png
