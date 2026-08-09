"""A multi-seed sweep must refuse to run when its seeds are a no-op.

`seed_diversifies` was added on 2026-08-01 after a 12-config sweep turned out to
contain 3 distinct instances. It was never called. The quantum scripts kept
passing several seeds at rank_jitter=0, and `p3_quadratic` and
`s3_coverage_sweep` still hold records where "seed 7" and "seed 24" share an
exact optimum and their optimized angles to full precision -- one instance,
labelled two. The predicate has to be enforced, not merely available.
"""

from __future__ import annotations

import pytest

from eon.instances.candidate_lines import require_seed_diversification


def test_multi_seed_without_jitter_is_refused() -> None:
    with pytest.raises(ValueError, match="NO-OP"):
        require_seed_diversification([7, 24], "community_bridging", rank_jitter=0.0)


def test_jitter_makes_the_same_sweep_legal() -> None:
    require_seed_diversification([7, 24], "community_bridging", rank_jitter=0.5)


def test_a_single_seed_needs_no_jitter() -> None:
    require_seed_diversification([7], "community_bridging", rank_jitter=0.0)
    require_seed_diversification([7, 7], "community_bridging", rank_jitter=0.0)


def test_random_uniform_diversifies_on_its_own() -> None:
    require_seed_diversification([7, 24], "random_uniform", rank_jitter=0.0)
