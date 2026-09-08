# Cases

> Run all commands below on a compute node, including configuration checks.
> This case uses schema version 1. Historical outputs cannot be resumed;
> regenerate inputs in the new format. See [verification](../doc/verification.md).

A case is data: physical parameters, finite-volume numerical controls, input
fields, and optional reference evidence. The package's simulation builders, shared runtime, and workflows own composition,
execution, diagnostics, and output effects.

- [`Andren1994`](Andren1994/README.md) is a neutral ABL comparison.
- [`Nieuwstadt1993`](Nieuwstadt1993/README.md) is a shear-free buoyant ABL.
- [`GABLS1`](GABLS1/README.md) is the stable nine-hour intercomparison.
- [`DTU10MWPrecursor`](DTU10MWPrecursor/README.md) contains an FV precursor and
  open-domain DTU 10-MW turbine workflow.
- [`HITSZWindTunnel`](HITSZWindTunnel/README.md) contains FV HITSZ R9 turbine
  workflows and measured reference data.
- [`HITSZLiquidNitrogenJet`](HITSZLiquidNitrogenJet/README.md) contains mapped
  incompressible and low-Mach cryogenic jet cases.

Resolve a case using the unified CLI:

```bash
jaxwind check cases/Andren1994/config.toml
jaxwind check cases/GABLS1/config.toml
jaxwind check \
  cases/DTU10MWPrecursor/fv_workflow.toml
jaxwind check \
  cases/HITSZLiquidNitrogenJet/fv_256.toml
```

See [the explicit continuation workflow](workflows/low_mach_continuation.toml)
for a self-producing new-format checkpoint dependency. Change its small step
counts for production; do not substitute historical checkpoints.
