# Where every reported number comes from

The proposal states that a script in this repository produces each of its
numbers. This table makes that statement checkable. For each headline result it
names the script, the artifact the script writes, and the figure the artifact
feeds.

Paths are relative to `workspace/`. `./scripts/reproduce_headline.sh full`
regenerates the whole set in order.

## Headline results

| Claim in the proposal | Script | Artifact | Figure |
|---|---|---|---|
| Congestion 4.02 → 0.56 MW (86 %), one line, 55 % of relief from the build | `scripts/congestion_metrics.py` | `experiments/results/congestion_seed7_fixed/` | `plan_ieee33.pdf` |
| Worst case over the box 1.87 MW; 0.00 % under-report; random 5-point set misses 67.8 % | `scripts/scenario_box_sweep.py` | `experiments/results/layerA_fixed_20260808T221908Z/ieee33_box.json` | — |
| 9/9 Layer A runs time-limited, gap 0.065–0.301, three candidate-pool sizes | `scripts/reconfiguration_sweep.py` | `experiments/results/layerA_fixed_20260808T221908Z/on_cc{24,30,36}.jsonl` | `hardness_family.pdf` |
| Reconfiguration-off control: optimal in 1.1–5.9 s | `scripts/reconfiguration_sweep.py --no-reconfiguration` | `experiments/results/layerA_fixed_20260808T221908Z/off_cc{24,30,36}.jsonl` | `hardness_family.pdf` |
| **9 of 9** ON surrogates decoupled ($\|J\|/\|h\| \sim 10^{-9}$); OFF control reaches 87 | (same records as above) | `experiments/results/layerA_fixed_20260808T221908Z/` | `hardness_family.pdf` |
| Scale tier: dense glasses open at 1 h, exact TN cannot contract | `scripts/pathb_spine.py` | `experiments/results/pathb_dense_1h/` | — |
| MV Oberrhein (BOTH feeders) closes in 2.3–10.6 s: the real feeder is NOT hard | `scripts/reconfiguration_sweep.py --feeders mv_oberrhein_f1,mv_oberrhein_f2` | `experiments/results/layerA_fixed_20260808T221908Z/mv_oberrhein_f{1,2}_on.jsonl` | — |
| S3 coverage sweep: no consistent cop advantage over 5 subspace sizes | `scripts/random_feasible_control.py --hamming-weights` | `experiments/results/s3_coverage_sweep/` | `s3_coverage.pdf` |
| MPS 62–82 % above the classical incumbent at $n=120$, two seeds, $\chi=64$ | `scripts/scale_mps_sweep.py` | `experiments/results/scale_mps_rerun/n120_160.jsonl` | — |
| MPS within 1.2 % of optimum by $p=2$ at $n=20$ | `scripts/scale_mps_sweep.py` | `experiments/results/mps_higherp/` | — |
| Random-feasible control: optimum in 24/25 repeats; cop does not beat the median | `scripts/random_feasible_control.py` | `experiments/results/s3_control/control.jsonl` | — |
| Transpiled depth 1,118 vs 3,812,541 (≈3,400×) | `scripts/transpile_table.py` | `experiments/results/s4_transpile/table.json` | — |
| Distance above a proven CNOT floor: 4.1× vs 5,918× | `scripts/gate_count_floor.py` | `experiments/results/s4_transpile/gate_floor.json` | — |
| Aer vs exact engine agree to 0.06 σ | `scripts/qiskit_validation.py` | `experiments/results/qiskit_validation/` | — |
| D14 certificate sandwiches the planted optimum | `scripts/qaoa_decomposition_run.py` | `experiments/results/qaoa_decomp_v2/` | — |
| QAOA landscapes, cop vs vanilla vs warm-start | `scripts/qaoa_landscape.py` | `experiments/results/qaoa_p3/` | `p1_landscapes.pdf`, `excess_vs_depth.pdf`, `feasible_fraction_vs_depth.pdf` |
| Fair baseline: cop 0.0000, warm 0.0000, penalty QAOA 0.89-1.52 | `scripts/qaoa_landscape.py --penalty-mode quadratic` | `experiments/results/p3_quadratic/landscape_corrected.jsonl` | — |


## Superseded artifacts

On 2026-08-08 four defects were fixed in the Layer A LinDistFlow model (see the
proposal's *Negative results*). Every artifact derived from Layer A **before**
that date is superseded and must not be quoted:

| Superseded | Replaced by |
|---|---|
| `experiments/results/congestion_seed7.SUPERSEDED_broken_physics/` | `experiments/results/congestion_seed7_fixed/` |
| `experiments/results/s1_proper/`, `experiments/results/s1_proper_off/` | `experiments/results/layerA_fixed_20260808T221908Z/on_cc*.jsonl`, `off_cc*.jsonl` |
| `experiments/results/s2_oberrhein/` | `experiments/results/layerA_fixed_20260808T221908Z/mv_oberrhein_f{1,2}_on.jsonl` |
| `experiments/results/scenario_box/` | `experiments/results/layerA_fixed_20260808T221908Z/ieee33_box.json` |
| `experiments/results/p3_quadratic/`, `experiments/results/s3_coverage_sweep/`, `experiments/results/s3_control/`, `experiments/results/mps_higherp/` | re-run in progress; the claims they supported are withdrawn in the proposal |

They are kept rather than deleted so the retraction stays auditable.

The synthetic tiers (`experiments/results/pathb_dense_1h/`, `experiments/results/scale_mps_rerun/`, `experiments/results/s4_transpile/`,
`experiments/results/qiskit_validation/`) never passed through Layer A and are unaffected.

## Proved, not measured

| Statement | Source | Check |
|---|---|---|
| Planted optimum is a global minimum of the posiform | `lean/PosiformPlanting.lean` | `lean lean/PosiformPlanting.lean` |
| Decomposition lower bound never exceeds any assignment's energy | `lean/DroppedCouplingBound.lean` | `lean lean/DroppedCouplingBound.lean`; brute-forced by `tests/test_dropped_coupling_bound.py` |

Each file ends with an axiom audit and contains no `sorry`.

## Citations

`scripts/audit_citations.py` checks the title of every DOI entry against the
Crossref record. It also lists the keys the proposal cites. The script exits
non-zero on a mismatch.

The arXiv API refuses scripted access from this environment. A person therefore
verifies the arXiv-only entries by hand. The script lists them, so the unchecked
set stays visible.

## Quick mode

`reproduce_headline.sh quick` checks the wiring. It does not reproduce results.
Its time limits are far too short. **Do not quote a quick-mode number.** Only
`full` mode reproduces what the proposal reports.
