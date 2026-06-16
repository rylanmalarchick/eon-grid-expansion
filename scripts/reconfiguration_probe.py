"""D13 probe: does joint expansion+reconfiguration make Layer B couplings
load-bearing?

Builds the Layer B surrogate for the same instance with reconfiguration OFF vs
ON and prints the coupling diagnostics side by side. The number to watch is J/h
(coupling/field ratio) and the effective tree-width: the re-score showed these
are ~0.07 / 0 without reconfiguration on IEEE 123. If reconfiguration lifts
them, D13 is validated.

Run from workspace/:  python scripts/reconfiguration_probe.py [feeder] [family]
"""

from __future__ import annotations

import sys

from eon.formulations.layer_b import build_layer_b_surrogate
from eon.formulations.lindistflow import ExpansionProblemConfig, solve_lindistflow_expansion
from eon.instances.candidate_lines import generate_candidate_lines
from eon.instances.distribution_feeders import load_distribution_feeder
from eon.instances.scenarios import build_scenario_set
from eon.instances.treewidth import coupling_diagnostics

FEEDER = sys.argv[1] if len(sys.argv) > 1 else "ieee33"
FAMILY = sys.argv[2] if len(sys.argv) > 2 else "community_bridging"
CANDIDATE_COUNT = 12
NEIGHBORHOOD_SIZE = 8
PAIR_SAMPLE_LIMIT = 20
MAX_NEW_LINES = 3
SOLVE_TIME_LIMIT = 20.0
EVAL_TIME_LIMIT = 15.0
COST_PER_KM = 10_000.0


def probe(enable_reconfiguration: bool) -> None:
    net = load_distribution_feeder(FEEDER)
    scenarios = build_scenario_set("stressed_five_point")
    candidates = generate_candidate_lines(
        net, FAMILY, CANDIDATE_COUNT, seed=7, cost_per_km=COST_PER_KM
    )
    config = ExpansionProblemConfig(
        max_new_lines=MAX_NEW_LINES,
        time_limit_s=SOLVE_TIME_LIMIT,
        enable_reconfiguration=enable_reconfiguration,
    )
    layer_a = solve_lindistflow_expansion(net, scenarios, candidates, config)
    surrogate = build_layer_b_surrogate(
        net,
        scenarios,
        candidates,
        layer_a,
        config,
        neighborhood_size=min(NEIGHBORHOOD_SIZE, len(candidates)),
        evaluation_time_limit_s=EVAL_TIME_LIMIT,
        pair_sample_limit=PAIR_SAMPLE_LIMIT,
    )
    diag = coupling_diagnostics(surrogate)
    ratio = diag.coupling_field_ratio
    ratio_s = f"{ratio:.3f}" if ratio is not None else "None"
    label = "ON " if enable_reconfiguration else "OFF"
    print(
        f"reconfig {label}: builds={len(layer_a.selected_candidates):>2} "
        f"closed={layer_a.metadata.get('closed_line_count')} "
        f"n={diag.variable_count:>2} struct_tw={diag.structural_treewidth:>2} "
        f"eff_tw={diag.effective_treewidth:>2} J/h={ratio_s:>9} "
        f"eff_edges={diag.effective_edges:>3} mps_easy={diag.mps_easy} "
        f"layer_a_status={layer_a.termination_status}"
    )


def main() -> None:
    print(f"feeder={FEEDER} family={FAMILY} (stressed_five_point)")
    probe(enable_reconfiguration=False)
    probe(enable_reconfiguration=True)


if __name__ == "__main__":
    main()
