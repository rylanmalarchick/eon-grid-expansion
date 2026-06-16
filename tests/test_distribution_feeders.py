from eon.instances.distribution_feeders import load_distribution_feeder, slack_bus_index


def test_load_ieee33() -> None:
    net = load_distribution_feeder("ieee33")
    assert len(net.bus) == 33
    assert len(net.line) > 0
    assert slack_bus_index(net) in net.bus.index


def test_load_ieee123() -> None:
    net = load_distribution_feeder("ieee123")
    assert len(net.bus) == 123
    assert len(net.line) >= 100
    assert slack_bus_index(net) in net.bus.index
