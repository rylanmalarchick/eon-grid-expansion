"""The E.ON-facing figure: what the plan actually does to the feeder.

Three panels, each one measure (congestion MW and voltage violation pu live on
different scales, so they never share an axis):
  A  IEEE 33 as planned -- existing lines, the lines the plan builds, and the
     switches reconfiguration opens.
  B  scenario-weighted congestion (the D3 primary axis) for status quo /
     reconfiguration only / plan.
  C  scenario-weighted voltage violation for the same three, which is where
     reconfiguration's cost shows up.

Panels B and C carry one measure each, so a single hue is enough and no legend
is needed -- the panel titles name the quantity, and every bar is direct-
labeled.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandapower as pp

from eon.instances.distribution_feeders import bus_coordinates

_BLUE = "#2a78d6"      # the plan / built lines
_RED = "#e34948"       # opened switches (validated pair with _BLUE)
_TEXT = "#0b0b0b"
_MUTED = "#52514e"
_GRID = "#e5e4de"
_EXISTING = "#c3c2b7"


def _bar_panel(
    axis: plt.Axes,
    labels: list[str],
    values: list[float],
    *,
    title: str,
    unit: str,
    highlight: int,
) -> None:
    colors = [_BLUE if index == highlight else _EXISTING for index in range(len(values))]
    bars = axis.bar(labels, values, color=colors, width=0.62, zorder=3)
    span = max(values) if max(values) else 1.0
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + span * 0.03,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=_TEXT,
        )
    axis.set_title(title, color=_TEXT, fontsize=10, pad=10)
    axis.set_ylabel(unit, color=_MUTED, fontsize=9)
    axis.set_ylim(0, span * 1.18)
    axis.grid(True, axis="y", color=_GRID, linewidth=0.6, zorder=0)
    axis.set_axisbelow(True)
    axis.tick_params(colors=_MUTED, labelsize=8)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axis.spines[spine].set_color(_EXISTING)


def plot_plan_figure(
    metrics_path: Path,
    net: pp.pandapowerNet,
    out_stem: Path,
    *,
    feeder_label: str = "IEEE 33",
) -> Path:
    record = json.loads(Path(metrics_path).read_text())
    evaluations = record["evaluations"]
    built = set(record["layer_a"]["selected_candidates"])

    figure = plt.figure(figsize=(13.5, 4.6))
    grid_spec = figure.add_gridspec(1, 3, width_ratios=[1.5, 1, 1], wspace=0.32)

    # --- Panel A: the feeder and the plan ---------------------------------
    axis = figure.add_subplot(grid_spec[0, 0])
    graph = nx.Graph()
    graph.add_nodes_from(int(bus) for bus in net.bus.index)
    for _, row in net.line.iterrows():
        graph.add_edge(int(row.from_bus), int(row.to_bus))
    # Real feeder geometry, not a spring layout: a planner must recognise
    # their own network.
    positions = {
        bus: point for bus, point in bus_coordinates(net).items() if bus in graph
    }

    nx.draw_networkx_edges(
        graph, positions, ax=axis, edge_color=_EXISTING, width=1.6
    )
    nx.draw_networkx_nodes(
        graph, positions, ax=axis, node_size=26, node_color=_EXISTING,
        edgecolors="#fcfcfb", linewidths=0.8,
    )
    # The built lines are candidates, i.e. NOT edges of the existing feeder:
    # draw them as new connections between their endpoints.
    for name in sorted(built):
        parts = name.split("_")
        try:
            from_bus, to_bus = int(parts[-3]), int(parts[-2])
        except (ValueError, IndexError):
            continue
        if from_bus in positions and to_bus in positions:
            axis.plot(
                [positions[from_bus][0], positions[to_bus][0]],
                [positions[from_bus][1], positions[to_bus][1]],
                color=_BLUE, linewidth=2.6, zorder=4,
                solid_capstyle="round",
            )
            axis.scatter(
                [positions[from_bus][0], positions[to_bus][0]],
                [positions[from_bus][1], positions[to_bus][1]],
                s=54, color=_BLUE, zorder=5, edgecolors="#fcfcfb", linewidths=1.0,
            )
    # metadata's closed_line_count is the number of lines held CLOSED (in
    # service); the switches OPENED are the remainder. Inverting these two is
    # how a caption ends up claiming a 33-bus feeder was cut into 32 pieces.
    closed = int(evaluations["plan"].get("closed_line_count") or 0)
    existing = len(net.line)
    opened = max(0, existing - closed)
    axis.set_title(
        f"{feeder_label} with the planned build\n"
        f"{len(built)} new line{'s' if len(built) != 1 else ''} (blue); "
        f"{closed} of {existing} lines closed, "
        f"{opened} open (radial operation)",
        color=_TEXT, fontsize=10, pad=10,
    )
    axis.axis("off")

    # --- Panels B and C: one measure each ---------------------------------
    order = ["status_quo", "reconfiguration_only", "plan"]
    labels = ["status quo", "reconfig.\nonly", "plan"]
    congestion = [
        float(evaluations[key]["aggregate_metrics"]["weighted_congestion_mw"])
        for key in order
    ]
    voltage = [
        float(evaluations[key]["aggregate_metrics"]["weighted_voltage_violation_pu"])
        for key in order
    ]
    _bar_panel(
        figure.add_subplot(grid_spec[0, 1]), labels, congestion,
        title="Congestion (scenario-weighted)", unit="over-limit MW", highlight=2,
    )
    _bar_panel(
        figure.add_subplot(grid_spec[0, 2]), labels, voltage,
        title="Voltage violation (scenario-weighted)", unit="per-unit", highlight=2,
    )

    reduction = (
        (congestion[0] - congestion[2]) / congestion[0] * 100 if congestion[0] else 0.0
    )
    # The title must follow the data. Under the corrected model the plan
    # improves BOTH measures, so the old "trades congestion for voltage"
    # framing is simply false; state whichever the numbers support.
    build_share = (
        (congestion[1] - congestion[2]) / (congestion[0] - congestion[2]) * 100
        if congestion[0] != congestion[2] else 0.0
    )
    both_improve = voltage[2] < voltage[0] and congestion[2] < congestion[0]
    tail = (
        f"-- and {build_share:.0f}% of the relief comes from the new line"
        if both_improve
        else "-- and reconfiguration alone trades congestion for voltage"
    )
    figure.suptitle(
        f"Planned expansion cuts scenario-weighted congestion {reduction:.0f}% {tail}",
        color=_TEXT, fontsize=11.5, y=1.03,
    )
    # Derived, not asserted: this caption claimed "TIME_LIMIT, not proven
    # optimal" for weeks after the solve started closing to optimality.
    layer_a = record.get("layer_a", {})
    status = str(layer_a.get("termination_status", "unknown"))
    gap = layer_a.get("mip_gap")
    provenance = (
        f"Layer A {status}"
        + (f" at gap {gap:.3f}" if isinstance(gap, (int, float)) else "")
        + (
            "; the plan is provably near-optimal"
            if status == "OPTIMAL"
            else "; the plan is an incumbent, not a proven optimum"
        )
    )
    figure.text(
        0.005, -0.06,
        f"IEEE 33, stressed five-point scenarios. {provenance}. Congestion is an "
        "L1 |P|+|Q| excess against defaulted branch ratings -- read the ratio, "
        "not the absolute MW. LinDistFlow fixed-builds evaluation of each "
        "configuration.",
        color=_MUTED, fontsize=7.5,
    )

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    png = out_stem.with_suffix(".png")
    figure.savefig(png, dpi=200, bbox_inches="tight")
    figure.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return png
