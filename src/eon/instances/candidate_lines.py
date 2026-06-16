from __future__ import annotations

import math
import random
from dataclasses import dataclass
from itertools import combinations

import networkx as nx
import numpy as np
import pandapower as pp
import pandas as pd

from eon.instances.distribution_feeders import bus_coordinates, feeder_graph


@dataclass(frozen=True, slots=True)
class CandidateLine:
    name: str
    from_bus: int
    to_bus: int
    family: str
    length_km: float
    r_ohm_per_km: float
    x_ohm_per_km: float
    max_i_ka: float
    build_cost: float


def generate_candidate_lines(
    net: pp.pandapowerNet,
    family: str,
    count: int,
    *,
    seed: int = 7,
    cost_per_km: float = 25_000.0,
    stress_info: dict[int, float] | None = None,
) -> list[CandidateLine]:
    if count <= 0:
        return []

    candidate_family = family.lower().strip()
    valid_families = {
        "random_uniform",
        "distance_weighted",
        "adversarial_long_range",
        "community_bridging",
        "useful_adversarial",
    }
    if candidate_family not in valid_families:
        options = ", ".join(sorted(valid_families))
        raise ValueError(f"Unknown candidate family '{family}'. Expected one of: {options}.")

    graph = feeder_graph(net)
    coords = bus_coordinates(net)
    stats = _reference_line_statistics(net)
    eligible_pairs = _eligible_pairs(graph)
    if not eligible_pairs:
        return []

    ranked_pairs = _rank_pairs(
        graph, coords, eligible_pairs, candidate_family, seed, stress_info=stress_info,
    )
    selected_pairs = ranked_pairs[:count]
    candidates: list[CandidateLine] = []
    for pair_index, (from_bus, to_bus) in enumerate(selected_pairs, start=1):
        hop_length = nx.shortest_path_length(graph, from_bus, to_bus)
        euclidean = _distance(coords[from_bus], coords[to_bus])
        length_km = max(
            0.1,
            euclidean * stats["coord_to_km_scale"],
            hop_length * stats["hop_to_km_scale"],
        )
        build_cost = cost_per_km * length_km * _family_cost_multiplier(candidate_family)
        candidates.append(
            CandidateLine(
                name=f"{candidate_family}_{from_bus}_{to_bus}_{pair_index}",
                from_bus=from_bus,
                to_bus=to_bus,
                family=candidate_family,
                length_km=float(length_km),
                r_ohm_per_km=stats["r_ohm_per_km"],
                x_ohm_per_km=stats["x_ohm_per_km"],
                max_i_ka=stats["max_i_ka"],
                build_cost=float(build_cost),
            )
        )
    return candidates


def _eligible_pairs(graph: nx.Graph) -> list[tuple[int, int]]:
    eligible: list[tuple[int, int]] = []
    nodes = sorted(int(node) for node in graph.nodes)
    for a, b in combinations(nodes, 2):
        if graph.has_edge(a, b):
            continue
        if not nx.has_path(graph, a, b):
            continue
        eligible.append((a, b))
    return eligible


def _rank_pairs(
    graph: nx.Graph,
    coords: dict[int, tuple[float, float]],
    pairs: list[tuple[int, int]],
    family: str,
    seed: int,
    *,
    stress_info: dict[int, float] | None = None,
) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    if family == "random_uniform":
        shuffled = list(pairs)
        rng.shuffle(shuffled)
        return shuffled

    if family == "distance_weighted":
        scored = []
        for pair in pairs:
            length = _distance(coords[pair[0]], coords[pair[1]])
            hop = nx.shortest_path_length(graph, *pair)
            scored.append((abs(length - 0.35 * hop), pair))
        return [pair for _, pair in sorted(scored)]

    if family == "adversarial_long_range":
        scored = []
        leaf_bonus = {node: 1.0 if graph.degree[node] <= 2 else 0.0 for node in graph.nodes}
        for pair in pairs:
            hop = nx.shortest_path_length(graph, *pair)
            length = _distance(coords[pair[0]], coords[pair[1]])
            score = hop + 2.0 * length + leaf_bonus[pair[0]] + leaf_bonus[pair[1]]
            scored.append((-score, pair))
        return [pair for _, pair in sorted(scored)]

    if family == "useful_adversarial":
        bus_stress = stress_info or {}
        scored = []
        for pair in pairs:
            hop = nx.shortest_path_length(graph, *pair)
            length = _distance(coords[pair[0]], coords[pair[1]])
            endpoint_stress = bus_stress.get(pair[0], 0.0) + bus_stress.get(pair[1], 0.0)
            # High stress + long hop = physically useful + topology-disruptive
            score = 3.0 * endpoint_stress + 1.5 * hop + length
            scored.append((-score, pair))
        return [pair for _, pair in sorted(scored)]

    # community_bridging
    communities = list(nx.community.greedy_modularity_communities(graph))
    community_lookup = {
        int(node): community_id
        for community_id, community in enumerate(communities)
        for node in community
    }
    scored = []
    for pair in pairs:
        different_communities = community_lookup[pair[0]] != community_lookup[pair[1]]
        hop = nx.shortest_path_length(graph, *pair)
        length = _distance(coords[pair[0]], coords[pair[1]])
        score = (4.0 if different_communities else 0.0) + hop + length
        scored.append((-score, pair))
    return [pair for _, pair in sorted(scored)]


def _reference_line_statistics(net: pp.pandapowerNet) -> dict[str, float]:
    if len(net.line):
        r_ohm_per_km = _series_median_or_default(net.line["r_ohm_per_km"], 0.5)
        x_ohm_per_km = _series_median_or_default(net.line["x_ohm_per_km"], 0.4)
        max_i_ka = _series_median_or_default(net.line["max_i_ka"], 0.4)
        avg_length = _series_median_or_default(net.line["length_km"], 0.5)
    else:
        r_ohm_per_km = 0.5
        x_ohm_per_km = 0.4
        max_i_ka = 0.4
        avg_length = 0.5
    return {
        "r_ohm_per_km": r_ohm_per_km or 0.5,
        "x_ohm_per_km": x_ohm_per_km or 0.4,
        "max_i_ka": max_i_ka or 0.4,
        "coord_to_km_scale": avg_length or 0.5,
        "hop_to_km_scale": max(0.15, avg_length * 0.75),
    }


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.dist(a, b)


def _family_cost_multiplier(family: str) -> float:
    if family == "community_bridging":
        return 1.2
    if family == "adversarial_long_range":
        return 1.35
    if family == "useful_adversarial":
        return 1.1
    return 1.0


def _finite_or_default(value: float, default: float) -> float:
    numeric = float(value)
    return default if np.isnan(numeric) else numeric


def _series_median_or_default(series: pd.Series, default: float) -> float:
    median = series.replace(0, np.nan).dropna().median()
    return _finite_or_default(median, default)
