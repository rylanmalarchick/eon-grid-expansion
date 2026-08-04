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
| Congestion 26.46 → 1.61 MW (94 %), two lines built | `scripts/congestion_metrics.py` | `experiments/results/congestion_seed7/` | `plan_ieee33.pdf` |
| Worst case over the uncertainty box 4.13 MW; 0.00 % under-report; random 5-point set misses 53.6 % | `scripts/scenario_box_sweep.py` | `experiments/results/scenario_box/ieee33_box.json` | — |
| 12/12 Layer A runs time-limited, gap 0.692–0.751, four variable counts | `scripts/reconfiguration_sweep.py` | `experiments/results/s1_proper/n{16,20,24,32}.jsonl` | `hardness_family.pdf` |
| Reconfiguration-off control: optimal in 2.6–6.1 s | `scripts/reconfiguration_sweep.py --no-reconfiguration` | `experiments/results/s1_proper_off/` | `hardness_family.pdf` |
| 5 of 12 surrogates near-decoupled ($\|J\|/\|h\| \approx 1$) | (same records as above) | `experiments/results/s1_proper/` | `hardness_family.pdf` |
| Scale tier: dense glasses open at 1 h, exact TN cannot contract | `scripts/pathb_spine.py` | `experiments/results/pathb_dense_1h/` | — |
| MPS 62–82 % above the classical incumbent at $n=120$, two seeds, $\chi=64$ | `scripts/scale_mps_sweep.py` | `experiments/results/scale_mps_rerun/n120_160.jsonl` | — |
| MPS within 1.2 % of optimum by $p=2$ at $n=20$ | `scripts/scale_mps_sweep.py` | `experiments/results/mps_higherp/` | — |
| Random-feasible control: optimum in 24/25 repeats; cop does not beat the median | `scripts/random_feasible_control.py` | `experiments/results/s3_control/control.jsonl` | — |
| Transpiled depth 1,118 vs 3,812,541 (≈3,400×) | `scripts/transpile_table.py` | `experiments/results/s4_transpile/table.json` | — |
| Aer vs exact engine agree to 0.06 σ | `scripts/qiskit_validation.py` | `experiments/results/qiskit_validation/` | — |
| D14 certificate sandwiches the planted optimum | `scripts/qaoa_decomposition_run.py` | `experiments/results/qaoa_decomp_v2/` | — |
| QAOA landscapes, cop vs vanilla vs warm-start | `scripts/qaoa_landscape.py` | `experiments/results/qaoa_p3/` | `p1_landscapes.pdf`, `excess_vs_depth.pdf`, `feasible_fraction_vs_depth.pdf` |

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
