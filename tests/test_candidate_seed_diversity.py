"""Seed semantics for candidate families (root-caused 2026-08-01).

The heuristic families are deterministic sorts: the seed is a no-op unless
rank_jitter > 0. These tests pin BOTH halves -- the backwards-compatible
default (so every record produced before 2026-08-01 stays reproducible) and
the opt-in jitter that makes seeds genuinely diversify.
"""

from __future__ import annotations

import pytest

from eon.instances.candidate_lines import generate_candidate_lines, seed_diversifies
from eon.instances.distribution_feeders import load_distribution_feeder

_DETERMINISTIC_FAMILIES = ("community_bridging", "useful_adversarial", "distance_weighted")


def _pairs(net, family: str, seed: int, jitter: float = 0.0) -> tuple:
    candidates = generate_candidate_lines(
        net,
        count=24,
        family=family,
        seed=seed,
        cost_per_km=10_000.0,
        stress_info={},
        rank_jitter=jitter,
    )
    return tuple((c.from_bus, c.to_bus) for c in candidates)


@pytest.mark.parametrize("family", _DETERMINISTIC_FAMILIES)
def test_seed_is_a_noop_without_jitter(family: str) -> None:
    """Documented (not accidental) determinism: seeds must NOT diversify at
    jitter=0, which is what keeps pre-2026-08-01 records reproducible."""
    net = load_distribution_feeder("ieee33")
    variants = {_pairs(net, family, seed) for seed in (7, 24, 42, 99)}
    assert len(variants) == 1
    assert seed_diversifies(family, rank_jitter=0.0) is False


@pytest.mark.parametrize("family", _DETERMINISTIC_FAMILIES)
def test_jitter_makes_seeds_diversify(family: str) -> None:
    net = load_distribution_feeder("ieee33")
    variants = {_pairs(net, family, seed, jitter=0.5) for seed in (7, 24, 42, 99)}
    assert len(variants) >= 3, "jittered seeds must yield distinct candidate sets"
    assert seed_diversifies(family, rank_jitter=0.5) is True


@pytest.mark.parametrize("family", _DETERMINISTIC_FAMILIES)
def test_jitter_is_deterministic_in_seed(family: str) -> None:
    net = load_distribution_feeder("ieee33")
    assert _pairs(net, family, 42, jitter=0.5) == _pairs(net, family, 42, jitter=0.5)


def test_random_uniform_always_diversifies() -> None:
    net = load_distribution_feeder("ieee33")
    variants = {_pairs(net, "random_uniform", seed) for seed in (7, 24, 42, 99)}
    assert len(variants) == 4
    assert seed_diversifies("random_uniform", rank_jitter=0.0) is True


@pytest.mark.parametrize("family", _DETERMINISTIC_FAMILIES)
def test_jitter_preserves_family_character(family: str) -> None:
    """Modest jitter perturbs the ranking's arbitrary tie-breaks, it does not
    replace the heuristic: most of the unjittered selection must survive."""
    net = load_distribution_feeder("ieee33")
    base = set(_pairs(net, family, 7))
    jittered = set(_pairs(net, family, 7, jitter=0.5))
    overlap = len(base & jittered) / len(base)
    assert overlap >= 0.5, f"{family}: jitter destroyed the family ({overlap:.0%} kept)"
