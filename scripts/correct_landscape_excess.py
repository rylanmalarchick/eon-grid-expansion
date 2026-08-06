"""Recompute best_sample_excess on landscape records written before the fix.

scripts/qaoa_landscape.py used to compute

    excess = (best.objective - exact_optimum) / scale

where best.objective is the UNPENALIZED surrogate objective and exact_optimum is
the minimum of the PENALIZED vector. Those are different cost functions. Under
the quadratic penalty a feasible sample below the build budget still pays
lambda*(count - K)^2, so the excess came out NEGATIVE -- a sample apparently
better than the optimum -- and it favoured whichever method sits furthest from
the budget. The constrained mixer is locked to one build count, so it was the
method being favoured.

The correction is exact and needs no re-run: the missing term is
lambda*(count - K)^2 with lambda = 50,000 and K = MAX_NEW_LINES, and the build
count is recorded in each sample's selected_candidates. Verified against a fresh
run of the fixed script, which reproduces these values directly.

Run from workspace/:
    python scripts/correct_landscape_excess.py --in RECORDS.jsonl --out CORRECTED.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("correct_landscape_excess")

PENALTY_STRENGTH = 50_000.0
MAX_NEW_LINES = 3
# The lambda/K model below is the FEEDER configuration (qaoa_landscape sets
# MAX_NEW_LINES = 3). The synthetic planted instances are built by a different
# path with a different budget, so applying this model to them produces
# nonsense -- it inflated their excess by four orders of magnitude on the first
# attempt. Correct only what the model demonstrably fits, and refuse the rest.
FEEDER_PREFIX = "ieee33"


def corrected(record: dict) -> tuple[dict, str]:
    """Returns (record, status) where status is corrected | skipped | unchanged."""
    sample = record.get("best_sample")
    if sample is None or record.get("best_sample_excess") is None:
        return record, "unchanged"
    if not record["instance_id"].startswith(FEEDER_PREFIX):
        return record, "skipped"

    builds = len(sample["selected_candidates"])
    penalty = PENALTY_STRENGTH * (builds - MAX_NEW_LINES) ** 2
    penalized = sample["objective"] + penalty
    scale = max(abs(record["exact_optimum"]), 1.0)
    new_excess = (penalized - record["exact_optimum"]) / scale

    # The model must explain the old number exactly: old excess plus the
    # penalty term, over the same scale, is the new one. If it does not, the
    # penalty model is wrong for this record and guessing would be worse than
    # leaving it alone.
    implied = record["best_sample_excess"] + penalty / scale
    if abs(implied - new_excess) > 1e-9 or new_excess < -1e-9:
        return record, "skipped"

    out = dict(record)
    out["best_sample"] = {**sample, "penalized_energy": penalized}
    out["best_sample_excess_uncorrected"] = record["best_sample_excess"]
    out["best_sample_excess"] = new_excess
    out["correction"] = {
        "penalty_strength": PENALTY_STRENGTH,
        "max_new_lines": MAX_NEW_LINES,
        "build_count": builds,
        "penalty_added": penalty,
        "note": "excess now compares penalized energy against the penalized optimum",
    }
    return out, "corrected"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="source", required=True)
    parser.add_argument("--out", dest="target", required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.source).read_text().splitlines() if line.strip()]
    results = [corrected(row) for row in rows]
    fixed = [r for r, _ in results]
    tally = {s: sum(1 for _, x in results if x == s) for s in ("corrected", "skipped", "unchanged")}
    logger.info("corrected %(corrected)d, skipped %(skipped)d, unchanged %(unchanged)d", tally)
    if tally["skipped"]:
        logger.warning(
            "SKIPPED records keep their original excess -- the feeder penalty model "
            "does not fit them, and a wrong correction is worse than none"
        )

    negatives_before = sum(
        1 for r in rows if isinstance(r.get("best_sample_excess"), float)
        and r["best_sample_excess"] < -1e-12
    )
    negatives_after = sum(
        1 for r in fixed if isinstance(r.get("best_sample_excess"), float)
        and r["best_sample_excess"] < -1e-12
    )
    logger.info(
        "%d records; negative excesses %d -> %d",
        len(rows), negatives_before, negatives_after,
    )
    feeder_negatives = sum(
        1 for r in fixed
        if r["instance_id"].startswith(FEEDER_PREFIX)
        and isinstance(r.get("best_sample_excess"), float)
        and r["best_sample_excess"] < -1e-12
    )
    if feeder_negatives:
        raise SystemExit("correction left a negative feeder excess; the penalty model is wrong")

    Path(args.target).write_text("\n".join(json.dumps(r, sort_keys=True) for r in fixed) + "\n")
    logger.info("wrote %s", args.target)


if __name__ == "__main__":
    main()
