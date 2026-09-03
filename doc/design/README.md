# Design documentation

[ADR-0017](decisions/0017-direct-finite-volume-solver.md) is the normative architecture record for the active solver. It defines one direct staggered finite-volume implementation with FFT, GMG, and optional AMG pressure backends.

The numbered design specifications and ADR-0001 through ADR-0016 are retained as historical records. They describe the superseded semantic-field, interpreter, ownership, LASD, and solver-facade architecture and are not implementation requirements.

New architectural changes should amend or supersede ADR-0017 before changing the corresponding public contract.
