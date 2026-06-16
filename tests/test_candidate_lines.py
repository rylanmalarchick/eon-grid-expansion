from eon.instances.candidate_lines import generate_candidate_lines
from eon.instances.distribution_feeders import feeder_graph, load_distribution_feeder


def test_candidate_families_generate_non_edges() -> None:
    net = load_distribution_feeder("ieee33")
    graph = feeder_graph(net)
    for family in [
        "random_uniform",
        "distance_weighted",
        "adversarial_long_range",
        "community_bridging",
        "useful_adversarial",
    ]:
        candidates = generate_candidate_lines(net, family, 3, seed=5)
        assert len(candidates) == 3
        assert len({candidate.name for candidate in candidates}) == 3
        for candidate in candidates:
            assert not graph.has_edge(candidate.from_bus, candidate.to_bus)
            assert candidate.length_km > 0
            assert candidate.build_cost > 0


def test_useful_adversarial_prefers_stressed_buses() -> None:
    net = load_distribution_feeder("ieee33")
    buses = [int(b) for b in net.bus.index]
    # Fake stress concentrated on a few buses
    stress_info = {bus: 0.0 for bus in buses}
    stress_info[buses[-1]] = 10.0
    stress_info[buses[-2]] = 8.0

    candidates = generate_candidate_lines(
        net, "useful_adversarial", 5, seed=5, stress_info=stress_info,
    )
    assert len(candidates) >= 1
    stressed_buses = {buses[-1], buses[-2]}
    # At least one candidate should touch a stressed bus
    touches_stressed = any(
        c.from_bus in stressed_buses or c.to_bus in stressed_buses for c in candidates
    )
    assert touches_stressed
