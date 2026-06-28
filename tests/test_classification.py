from eon.instances.hardness_classifier import _classify_instance


def _classify(**overrides: object) -> str:
    base: dict[str, object] = {
        "mps_negative": False,
        "tree_negative": False,
        "layer_a_hard": False,
        "nisq_runnable": True,
        "layer_b_validated": True,
        "layer_b_gap_open": False,
    }
    base.update(overrides)
    return _classify_instance(**base)  # type: ignore[arg-type]


def test_hard_requires_strong_control_failure_and_open_gurobi_gap() -> None:
    # The full hardness-scale gate: the strong exact-TN control fails (tree_negative)
    # AND Gurobi cannot close the reduced QUBO.
    assert _classify(mps_negative=True, tree_negative=True, layer_b_gap_open=True) == "hard"
    # Without the open Gurobi gap it is not hard.
    assert _classify(mps_negative=True, tree_negative=True, layer_b_gap_open=False) != "hard"


def test_nisq_path_a_candidate_is_the_n20_verdict() -> None:
    # The 2026-06-28 sweep case: a Gurobi-hard Layer A parent + an MPS-negative,
    # exact-TN-tractable (tree_negative=False), NISQ-runnable reduced QUBO with the
    # reduced-QUBO Gurobi gap closed. The pre-Wave-5 gate buried this as "ambiguous".
    assert (
        _classify(layer_a_hard=True, mps_negative=True, tree_negative=False, nisq_runnable=True)
        == "nisq_path_a_candidate"
    )


def test_candidate_requires_exact_tn_tractable() -> None:
    # tree_negative (the strong exact-TN control failed) is not a clean NISQ candidate;
    # with the Gurobi gap closed it is neither hard nor a candidate -> ambiguous.
    assert (
        _classify(layer_a_hard=True, mps_negative=True, tree_negative=True, nisq_runnable=True)
        == "ambiguous"
    )


def test_not_nisq_runnable_is_not_a_candidate() -> None:
    assert _classify(layer_a_hard=True, mps_negative=True, nisq_runnable=False) == "ambiguous"


def test_layer_a_hard_without_mps_negative_is_ambiguous() -> None:
    assert _classify(layer_a_hard=True, mps_negative=False, tree_negative=True) == "ambiguous"


def test_layer_a_hard_alone_stays_easy_when_quantum_easy() -> None:
    # "easy" is the quantum verdict: a hard Layer A parent with a quantum-easy reduced
    # QUBO (not MPS- or tree-negative) is still "easy" -- layer_a_hard must not narrow it.
    assert _classify(layer_a_hard=True, mps_negative=False, tree_negative=False) == "easy"


def test_mps_negative_alone_is_ambiguous() -> None:
    assert _classify(mps_negative=True, layer_a_hard=False, tree_negative=False) == "ambiguous"


def test_all_negative_signals_off_is_easy() -> None:
    assert _classify() == "easy"
