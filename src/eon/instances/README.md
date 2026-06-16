# Instances

Scripts to generate and classify hard problem instances.

- `ieee_loader.py` — wrappers around pandapower IEEE cases (case14, case30, case57, case118, case300).
- `candidate_lines.py` — synthetic candidate-line generators (random, distance-weighted, deliberately long-range).
- `scenarios.py` — scenario sampling for load growth / renewable injection.
- `hardness_classifier.py` — Gurobi time-limited runs; writes `instances/hard_instances.jsonl`.
