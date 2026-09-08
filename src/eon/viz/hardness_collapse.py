"""The hardness we reported was our own arithmetic.

One panel per axis of the claim. Left: Layer A solve outcome before and after
the voltage-units correction, per instance -- the instances that hit the 1800 s
limit now close in seconds. Right: every feeder we tested, so the reader can see
that the corrected IEEE 33 is not an outlier but agrees with IEEE 123 and both
MV Oberrhein feeders.

This figure replaces one that plotted Layer A gap against surrogate coupling.
That panel is gone because the coupling diagnostic it drew is withdrawn, and
because plotting a gap of zero against a gap of zero shows nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_BEFORE = "#b9b7ab"
_AFTER = "#2a78d6"
_TEXT = "#0b0b0b"
_MUTED = "#52514e"
_GRID = "#e5e4de"
_EDGE = "#c3c2b7"
_LIMIT = 1800.0


def _load(paths: list[Path]) -> list[dict]:
    records: list[dict] = []
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def _style(axis, *, title: str, xlabel: str, ylabel: str) -> None:
    axis.set_title(title, color=_TEXT, fontsize=10.5, pad=10)
    axis.set_xlabel(xlabel, color=_MUTED, fontsize=9)
    axis.set_ylabel(ylabel, color=_MUTED, fontsize=9)
    axis.grid(True, axis="y", color=_GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.tick_params(colors=_MUTED, labelsize=8)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axis.spines[spine].set_color(_EDGE)


def plot_collapse(
    before: list[dict], after: list[dict], out_stem: Path
) -> Path:
    if not before or not after:
        raise ValueError("need both the pre-fix and post-fix records")

    def key(record: dict) -> tuple[str, int, int]:
        # The feeder MUST be part of the key: MV Oberrhein also carries pool 24
        # with seeds 7 and 24, so keying on (pool, seed) alone silently
        # overwrote the IEEE 33 rows with Oberrhein's much faster ones.
        return str(record["feeder"]), int(record["candidate_count"]), int(record["seed"])

    pre = {key(r): r for r in before if r.get("enable_reconfiguration")}
    post = {key(r): r for r in after if r.get("enable_reconfiguration")}
    shared = sorted(set(pre) & set(post))
    if not shared:
        raise ValueError("pre-fix and post-fix records share no instance")

    figure, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.3))
    figure.subplots_adjust(wspace=0.3)

    positions = range(len(shared))
    labels = [f"{cc}/{seed}" for _feeder, cc, seed in shared]
    left.bar(
        [p - 0.2 for p in positions],
        [min(pre[k]["layer_a"]["runtime_s"], _LIMIT) for k in shared],
        width=0.4, color=_BEFORE, label="before the units fix", zorder=3,
    )
    left.bar(
        [p + 0.2 for p in positions],
        [post[k]["layer_a"]["runtime_s"] for k in shared],
        width=0.4, color=_AFTER, label="after", zorder=3,
    )
    left.axhline(_LIMIT, color="#a33", linewidth=1.0, linestyle="--", zorder=2)
    left.text(
        len(shared) - 0.5, _LIMIT, " 1800 s limit", color="#a33",
        fontsize=7.5, va="bottom", ha="right",
    )
    left.set_yscale("log")
    left.set_xticks(list(positions))
    left.set_xticklabels(labels, fontsize=7.5)
    _style(
        left,
        title="Layer A solve time, same instances",
        xlabel="candidate pool / seed",
        ylabel="seconds to terminate (log)",
    )
    left.legend(frameon=False, fontsize=8, labelcolor=_MUTED, loc="lower right")

    # Right: every feeder, post-fix, so the corrected IEEE 33 can be seen in company.
    by_feeder: dict[str, list[float]] = {}
    for record in after:
        if not record.get("enable_reconfiguration"):
            continue
        by_feeder.setdefault(str(record["feeder"]), []).append(
            float(record["layer_a"]["runtime_s"])
        )
    names = sorted(by_feeder)
    for index, name in enumerate(names):
        values = by_feeder[name]
        right.scatter(
            [index] * len(values), values, s=54, color=_AFTER, zorder=3,
            edgecolors="#fcfcfb", linewidths=1.0,
        )
    right.axhline(_LIMIT, color="#a33", linewidth=1.0, linestyle="--", zorder=2)
    right.set_yscale("log")
    right.set_xticks(list(range(len(names))))
    right.set_xticklabels([n.replace("mv_oberrhein_", "Oberrhein ") for n in names],
                          fontsize=8)
    right.set_xlim(-0.6, len(names) - 0.4)
    _style(
        right,
        title="Every feeder we tested, after the fix",
        xlabel="",
        ylabel="seconds to terminate (log)",
    )

    figure.suptitle(
        "The hardness we reported was our own arithmetic: corrected, every real "
        "feeder closes in minutes",
        color=_TEXT, fontsize=11.5, y=1.02,
    )
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    png = out_stem.with_suffix(".png")
    figure.savefig(png, dpi=200, bbox_inches="tight")
    figure.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return png


def plot_feeder_runtimes(after: list[dict], out_stem: Path) -> Path:
    """Layer A termination time per feeder, from the current records alone.

    One panel, no comparison against a superseded run. Every point is a solve
    from the shipped artifacts, against the 1800 s limit the challenge's
    "reasonable runtime" criterion is measured at.
    """
    by_feeder: dict[str, list[float]] = {}
    for record in after:
        if not record.get("enable_reconfiguration"):
            continue
        by_feeder.setdefault(str(record["feeder"]), []).append(
            float(record["layer_a"]["runtime_s"])
        )
    if not by_feeder:
        raise ValueError("no reconfiguration-on records to plot")

    figure, axis = plt.subplots(1, 1, figsize=(7.0, 4.0))
    names = sorted(by_feeder)
    for index, name in enumerate(names):
        values = by_feeder[name]
        axis.scatter(
            [index] * len(values), values, s=58, color=_AFTER, zorder=3,
            edgecolors="#fcfcfb", linewidths=1.0,
        )
    axis.axhline(_LIMIT, color="#a33", linewidth=1.0, linestyle="--", zorder=2)
    axis.text(
        len(names) - 0.45, _LIMIT, " 1800 s limit", color="#a33",
        fontsize=8, va="bottom", ha="right",
    )
    axis.set_yscale("log")
    axis.set_xticks(list(range(len(names))))
    axis.set_xticklabels(
        [n.replace("mv_oberrhein_", "MV Oberrhein ").replace("ieee33", "IEEE 33")
         for n in names],
        fontsize=9,
    )
    axis.set_xlim(-0.6, len(names) - 0.4)
    _style(
        axis,
        title="Layer A termination time, every feeder tested",
        xlabel="",
        ylabel="seconds to terminate (log)",
    )
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    png = out_stem.with_suffix(".png")
    figure.savefig(png, dpi=200, bbox_inches="tight")
    figure.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return png
