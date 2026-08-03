# Instances

This directory holds generated instance files. The generators write here. The
directory is empty in a fresh clone.

## Feeders

`eon.instances.distribution_feeders` loads two networks:

- `ieee33` is the IEEE 33-bus feeder. It is the headline instance for every
  reported result.
- `mv_oberrhein` is a real medium-voltage distribution network from pandapower.
  It splits into two as-operated feeders, `mv_oberrhein_f1` and
  `mv_oberrhein_f2`.

IEEE 123 was dropped. Its Layer A model did not produce hard instances under the
reconfiguration formulation, and the two-tier design replaced it. Synthetic
long-range spin glasses now carry the scale tier.

## Synthetic instances

`eon.instances.external` generates the scale-tier benchmarks. A seed controls
each one, so no file needs to be stored to reproduce a run.

- `generate_posiform_planted` builds a QUBO with a known planted optimum.
- `generate_fused_planted` fuses random spin-glass blocks into a planted
  posiform to raise hardness.
- `generate_longrange_spin_glass` builds the dense instances used at n = 80 to
  160.
