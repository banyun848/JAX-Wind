# Cases

A case is data: physical parameters, finite-volume numerical controls, input
fields, and optional reference evidence. Applications own composition,
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

Run a case by passing its TOML file to the matching finite-volume application:

```bash
python -m applications.fv_abl cases/Andren1994/config.toml --dry-run
python -m applications.fv_abl cases/GABLS1/config.toml --dry-run
python -m applications.fv_abl.workflow \
  cases/DTU10MWPrecursor/fv_workflow.toml --dry-run
python -m applications.fv_ln2_jet \
  cases/HITSZLiquidNitrogenJet/fv_256.toml --dry-run
```
