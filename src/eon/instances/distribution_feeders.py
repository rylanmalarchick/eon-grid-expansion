from __future__ import annotations

import ast
import json
import warnings
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandapower as pp
import pandapower.networks as pn
from matpowercaseframes import CaseFrames
from pandapower.converter.pypower import from_ppc

IEEE123_MATPOWER_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "ieee123" / "grid_IEEE123_complete.m"
)
SUPPORTED_FEEDERS = frozenset({"ieee33", "ieee123"})


def load_distribution_feeder(name: str) -> pp.pandapowerNet:
    feeder = name.lower().strip()
    if feeder not in SUPPORTED_FEEDERS:
        options = ", ".join(sorted(SUPPORTED_FEEDERS))
        raise ValueError(f"Unsupported feeder '{name}'. Expected one of: {options}.")

    if feeder == "ieee33":
        net = pn.case33bw()
        net.sn_mva = float(net.sn_mva or 1.0)
    else:
        net = _load_ieee123_from_matpower()

    _normalize_network(net, feeder)
    return net


def feeder_graph(net: pp.pandapowerNet) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(int(bus) for bus in net.bus.index)
    for _, row in net.line.iterrows():
        if bool(row.in_service):
            graph.add_edge(int(row.from_bus), int(row.to_bus))
    for _, row in net.trafo.iterrows():
        if bool(row.in_service):
            graph.add_edge(int(row.hv_bus), int(row.lv_bus))
    return graph


def slack_bus_index(net: pp.pandapowerNet) -> int:
    if len(net.ext_grid):
        return int(net.ext_grid.iloc[0].bus)
    if len(net.gen):
        return int(net.gen.iloc[0].bus)
    raise ValueError("Feeder does not expose an ext_grid or generator slack bus.")


def bus_coordinates(net: pp.pandapowerNet) -> dict[int, tuple[float, float]]:
    coords = _coords_from_geo_columns(net)
    if coords:
        return coords

    graph = feeder_graph(net)
    layout = nx.spring_layout(graph, seed=11)
    return {int(bus): (float(point[0]), float(point[1])) for bus, point in layout.items()}


def _normalize_network(net: pp.pandapowerNet, feeder_name: str) -> None:
    if "name" not in net.bus:
        net.bus["name"] = [f"bus_{bus}" for bus in net.bus.index]
    net.bus["name"] = [
        str(name) if name is not None and name == name and str(name).strip() else f"bus_{bus}"
        for bus, name in zip(net.bus.index, net.bus["name"], strict=False)
    ]
    net.bus.index = net.bus.index.astype(int)
    if len(net.line):
        net.line["from_bus"] = net.line["from_bus"].astype(int)
        net.line["to_bus"] = net.line["to_bus"].astype(int)
        net.line["length_km"] = net.line["length_km"].astype(float).clip(lower=0.001)
        current_limits = net.line["max_i_ka"].astype(float)
        current_limits = current_limits.mask(
            (current_limits <= 0.0) | (current_limits > 1_000.0),
            np.nan,
        )
        current_limit_default = current_limits.dropna().median()
        if np.isnan(current_limit_default):
            current_limit_default = 0.4
        net.line["max_i_ka"] = current_limits.fillna(float(current_limit_default))
    if len(net.trafo):
        net.trafo["hv_bus"] = net.trafo["hv_bus"].astype(int)
        net.trafo["lv_bus"] = net.trafo["lv_bus"].astype(int)

    net["eon_metadata"] = {
        "feeder_name": feeder_name,
        "slack_bus": slack_bus_index(net),
        "bus_coordinates": bus_coordinates(net),
    }


def _load_ieee123_from_matpower() -> pp.pandapowerNet:
    if not IEEE123_MATPOWER_PATH.exists():
        raise FileNotFoundError(f"Missing IEEE 123 feeder file: {IEEE123_MATPOWER_PATH}")

    case_frames = CaseFrames(str(IEEE123_MATPOWER_PATH))
    ppc = {
        "version": case_frames.version,
        "baseMVA": float(case_frames.baseMVA),
        "bus": _frame_to_numeric(case_frames.bus),
        "gen": _frame_to_numeric(case_frames.gen),
        "branch": _frame_to_numeric(case_frames.branch),
    }
    ppc["bus"][:, 0] -= 1
    ppc["gen"][:, 0] -= 1
    ppc["branch"][:, 0] -= 1
    ppc["branch"][:, 1] -= 1
    ppc["branch"][ppc["branch"][:, 8] == 0, 8] = 1
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Setting an item of incompatible dtype is deprecated",
            category=FutureWarning,
        )
        net = from_ppc(ppc, f_hz=60)
    net.sn_mva = float(net.sn_mva or ppc["baseMVA"])
    return net


def _frame_to_numeric(frame: Any) -> np.ndarray:
    return frame.map(_safe_numeric).to_numpy(dtype=float)


def _safe_numeric(value: Any) -> float:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if not isinstance(value, str):
        return float(value)
    try:
        return float(value)
    except ValueError:
        node = ast.parse(value, mode="eval")
        allowed_nodes = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Pow,
            ast.USub,
            ast.UAdd,
            ast.Constant,
        )
        for child in ast.walk(node):
            if not isinstance(child, allowed_nodes):
                raise ValueError(
                    f"Unsupported numeric expression '{value}' in IEEE 123 data."
                ) from None
        return float(eval(compile(node, "<safe_numeric>", "eval"), {"__builtins__": {}}, {}))


def _coords_from_geo_columns(net: pp.pandapowerNet) -> dict[int, tuple[float, float]]:
    coords: dict[int, tuple[float, float]] = {}
    if "geo" not in net.bus.columns:
        return coords
    for bus, raw_geo in net.bus["geo"].items():
        if raw_geo is None or (isinstance(raw_geo, float) and np.isnan(raw_geo)):
            continue
        if isinstance(raw_geo, str):
            try:
                payload = json.loads(raw_geo)
            except json.JSONDecodeError:
                continue
        else:
            payload = raw_geo
        coordinates = payload.get("coordinates") if isinstance(payload, dict) else None
        if not coordinates or len(coordinates) < 2:
            continue
        coords[int(bus)] = (float(coordinates[0]), float(coordinates[1]))
    return coords
