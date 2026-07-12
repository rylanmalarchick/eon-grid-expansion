"""Shared QAOA angle optimization: grid at p=1, INTERP + Nelder-Mead above.

Both QAOA variants use this with the SAME budget so the comparison isolates
the ansatz, not the optimizer. INTERP initialization (linear interpolation of
the depth-(p-1) optimum onto p layers) is the standard heuristic for nested
variational ansaetze; Nelder-Mead is derivative-free (the statevector energy
is exact but not autodifferentiated here)."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

EnergyFn = Callable[[tuple[float, ...], tuple[float, ...]], float]

_GRID_BETAS = tuple(np.linspace(0.1, math.pi / 2, 7))
_GRID_GAMMAS = tuple(np.linspace(0.1, math.pi, 7))


@dataclass(frozen=True, slots=True)
class AngleSchedule:
    p: int
    betas: tuple[float, ...]
    gammas: tuple[float, ...]
    expected_energy: float
    energy_evaluations: int


def _interp(values: tuple[float, ...], target_p: int) -> tuple[float, ...]:
    """Linear INTERP of a depth-(p-1) schedule onto p layers."""
    source = np.asarray(values)
    if len(source) == 1:
        return tuple(float(source[0]) for _ in range(target_p))
    old_grid = np.linspace(0.0, 1.0, len(source))
    new_grid = np.linspace(0.0, 1.0, target_p)
    return tuple(float(v) for v in np.interp(new_grid, old_grid, source))


def optimize_angles(
    energy_fn: EnergyFn,
    p: int,
    *,
    nelder_mead_evals: int = 60,
    grid_betas: tuple[float, ...] = _GRID_BETAS,
    grid_gammas: tuple[float, ...] = _GRID_GAMMAS,
) -> list[AngleSchedule]:
    """Optimize layer angles for depths 1..p; returns one schedule per depth.

    Depth 1: exhaustive grid (the p=1 landscape the figure uses). Depth k>1:
    INTERP from the depth-(k-1) optimum, then Nelder-Mead capped at
    nelder_mead_evals energy evaluations."""
    if p < 1:
        raise ValueError(f"p must be >= 1, got {p}")

    schedules: list[AngleSchedule] = []
    evaluations = 0
    best = (float("inf"), grid_betas[0], grid_gammas[0])
    for beta in grid_betas:
        for gamma in grid_gammas:
            energy = energy_fn((beta,), (gamma,))
            evaluations += 1
            if energy < best[0]:
                best = (energy, beta, gamma)
    schedules.append(
        AngleSchedule(
            p=1,
            betas=(best[1],),
            gammas=(best[2],),
            expected_energy=best[0],
            energy_evaluations=evaluations,
        )
    )

    for depth in range(2, p + 1):
        previous = schedules[-1]
        evaluations = 0
        # Two starts: INTERP, and zero-padding (an identity extra layer, which
        # reproduces the depth-(p-1) state EXACTLY -- this is what makes the
        # final schedule monotone in depth by construction).
        starts = [
            (_interp(previous.betas, depth), _interp(previous.gammas, depth)),
            ((*previous.betas, 0.0), (*previous.gammas, 0.0)),
        ]
        start_energies = []
        for betas0, gammas0 in starts:
            start_energies.append(energy_fn(betas0, gammas0))
            evaluations += 1
        best_start = int(np.argmin(start_energies))
        betas0, gammas0 = starts[best_start]
        betas, gammas = betas0, gammas0
        energy = start_energies[best_start]

        def objective(x: np.ndarray, depth: int = depth) -> float:
            nonlocal evaluations
            evaluations += 1
            return energy_fn(tuple(x[:depth]), tuple(x[depth:]))

        result = minimize(
            objective,
            x0=np.asarray([*betas0, *gammas0]),
            method="Nelder-Mead",
            options={"maxfev": nelder_mead_evals, "xatol": 1e-3, "fatol": 1e-6},
        )
        if float(result.fun) <= energy:
            betas = tuple(float(v) for v in result.x[:depth])
            gammas = tuple(float(v) for v in result.x[depth:])
            energy = float(result.fun)
        schedules.append(
            AngleSchedule(
                p=depth,
                betas=betas,
                gammas=gammas,
                expected_energy=energy,
                energy_evaluations=evaluations,
            )
        )
    return schedules
