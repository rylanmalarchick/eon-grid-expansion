"""Check the proposal's headline numbers against the artifacts that produced them.

Prose drifts when an experiment is superseded. A single-seed MPS probe reported
67 %; the two-seed run at the full bond dimension reported 62.3 % and 82.4 %; the
table was updated and the word "67 %" survived in three other sentences,
including the compliance matrix. Re-reading caught it that time. This does not
rely on re-reading.

The check is deliberately narrow: it verifies that numbers the artifacts pin
down appear in the prose, and that superseded ones do not. It cannot judge
whether a sentence is true.

Run from workspace/:
    python scripts/check_proposal_numbers.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("check_proposal_numbers")


def _load(path: Path) -> list[dict]:
    """A MISSING artifact is a failure, not an empty check.

    This used to return [] so that every downstream loop silently did nothing:
    `--results /nonexistent` reported "every checked number matches its
    artifact" and exited 0. A drift guard that passes when there is nothing to
    check against is worse than no guard, because it reads as verification.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"expected artifact {path} is missing -- cannot verify the numbers "
            "it pins down. Regenerate it (docs/results_index.md names the "
            "script) or point --results at the right tree."
        )
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError(f"artifact {path} is empty -- nothing to check against")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", default="../paper/proposal.md")
    parser.add_argument("--results", default="experiments/results")
    args = parser.parse_args()

    prose = Path(args.proposal).read_text()
    root = Path(args.results)
    failures: list[str] = []
    checked = 0

    # --- scale-tier MPS excess, per seed -------------------------------
    mps = [r for r in _load(root / "scale_mps_rerun" / "n120_160.jsonl")
           if r.get("variable_count") == 120]
    for record in mps:
        pct = f"{100 * record['mps']['excess_over_incumbent']:.1f}"
        checked += 1
        if pct not in prose:
            failures.append(f"MPS excess {pct}% (seed {record['seed']}) is not in the proposal")

    # --- congestion headline -------------------------------------------
    congestion = root / "congestion_seed7"
    congestion_files = sorted(congestion.glob("*.json"))
    if not congestion_files:
        raise FileNotFoundError(
            f"no congestion artifact under {congestion} -- the headline "
            "reduction cannot be verified"
        )
    for candidate in congestion_files:
        record = json.loads(candidate.read_text())
        evaluations = record.get("evaluations", {})
        for label in ("status_quo", "plan"):
            if label in evaluations:
                value = evaluations[label]["aggregate_metrics"]["weighted_congestion_mw"]
                checked += 1
                if f"{value:.2f}" not in prose:
                    failures.append(f"congestion {label} {value:.2f} is not in the proposal")
        break

    # --- superseded values that must NOT reappear ----------------------
    # 67 %: the single-seed chi<=32 MPS probe, replaced by the two-seed chi=64 run.
    for stale, why in (("67 %", "single-seed MPS probe, superseded by the two-seed run"),):
        checked += 1
        if stale in prose:
            failures.append(f'superseded value "{stale}" is back in the proposal ({why})')

    logger.info("%d checks", checked)
    for failure in failures:
        logger.error(failure)
    if failures:
        sys.exit(1)
    logger.info(
        "%d checked number(s) match their artifact. This guard is NARROW: it is "
        "a substring test over the numbers listed above, and covers none of the "
        "S4 transpile/gate-floor figures.",
        checked,
    )


if __name__ == "__main__":
    main()
