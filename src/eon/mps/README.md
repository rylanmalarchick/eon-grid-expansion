# MPS baseline

- `protocol.py` wires Layer B QUBOs into a JuliQAOA-backed MPS protocol with multiple variable orderings and adaptive `chi` control.
- `juliqaoa_driver.jl` is the Julia workhorse: exact shallow-angle search, MPS evaluation, sampling diagnostics, and entropy summaries.
- `juliqaoa_smoke.py` exposes synthetic hard and easy control instances for end-to-end verification.
