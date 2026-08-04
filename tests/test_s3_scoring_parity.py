"""Both arms of the S3 control must be scored on the same object.

random_best is drawn from `energies` -- the compiled energy vector, penalties
included. cop_best was taken from `best_sample.objective`, which is the
UNPENALIZED surrogate objective. At the derived weight every sampled state is
feasible, the penalty is zero, and the two agree, so the mismatch was invisible.

Raise the weight above the build budget and every cop sample is infeasible. cop
is then credited with an unpenalized score while random pays the penalty, and
cop "wins" by construction -- it even reports an excess BELOW the exact optimum,
which is impossible against a true minimum.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from eon.instances.external import build_external_surrogate, generate_fused_planted
from eon.quantum.cop_qaoa import run_constrained_qaoa_depths
from eon.quantum.energy import build_energy_vector


@pytest.fixture(scope="module")
def setup():
    # The generator leaves max_new_lines == n, so every weight is feasible and
    # the penalty never fires. Constrain the budget so weights above it are
    # genuinely infeasible, which is the situation the real feeder run is in.
    surrogate = replace(
        build_external_surrogate(generate_fused_planted(12, seed=7, block_size=4, alpha=0.5)),
        max_new_lines=2,
    )
    energies = build_energy_vector(surrogate)
    return surrogate, energies


def _cop_energy_from_counts(counts: dict[str, int], energies: np.ndarray) -> float:
    """Score cop's samples on the SAME vector random is scored on."""
    return min(float(energies[int(bits[::-1], 2)]) for bits in counts)


def test_no_arm_can_beat_the_global_minimum(setup) -> None:
    surrogate, energies = setup
    exact = float(energies.min())
    for weight in (2, 4):
        result = run_constrained_qaoa_depths(
            surrogate, p=1, shots=128, energy_vector=energies, seed=7,
            hamming_weight=weight,
        )[0]
        scored = _cop_energy_from_counts(result.counts, energies)
        assert scored >= exact - 1e-6, (
            f"weight {weight}: cop scored {scored} below the global minimum "
            f"{exact} -- the arms are being scored on different objectives"
        )


def test_the_unpenalized_objective_can_fall_below_the_penalized_minimum(setup) -> None:
    """The negative control: this is the mismatch the test above prevents.

    Above the build budget the unpenalized objective of an infeasible state is
    genuinely lower than the penalized global minimum, so using it as cop's
    score is not a rounding difference -- it inverts the comparison.
    """
    surrogate, energies = setup
    exact = float(energies.min())
    result = run_constrained_qaoa_depths(
        surrogate, p=1, shots=128, energy_vector=energies, seed=7,
        hamming_weight=len(surrogate.variables) - 1,
    )[0]
    assert result.best_sample.objective < exact, (
        "expected the unpenalized objective of a heavily infeasible sample to "
        "sit below the penalized minimum; if it does not, this fixture no "
        "longer exercises the mismatch"
    )
