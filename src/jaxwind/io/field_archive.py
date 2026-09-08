"""Named scientific fields for explicit initialization and postprocessing."""
from __future__ import annotations
import numpy as np
from .state_fields import state_fields


class FieldArchive(dict):
    @property
    def files(self):
        return list(self)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()


def open_fields(path, **unused):
    fields, header = state_fields(path)
    aliases = {"pressure_dynamic_pa": "pressure", "water_vapor_mass_fraction": "water_vapor",
               "nitrogen_mass_fraction": "nitrogen", "continuity_error_kg_m3_s": "continuity_error"}
    for alias, name in aliases.items():
        if name in fields and not (alias == "pressure_dynamic_pa" and header.get("formulation") == "boussinesq"):
            fields[alias] = fields[name]
    for name, value in list(fields.items()):
        if name.startswith("parcels_"):
            fields["parcel_" + name[len("parcels_"):]] = value
    for axis in ("x", "y", "z"):
        if axis + "_faces" in header.get("mesh", {}):
            faces = np.asarray(header["mesh"][axis + "_faces"])
            fields[axis + "_faces_m"] = faces
            fields[axis + "_centers_m"] = .5 * (faces[:-1] + faces[1:])
    if header.get("mesh"):
        fields["lengths_m"] = np.array([header["mesh"][axis + "_faces"][-1] - header["mesh"][axis + "_faces"][0] for axis in ("x", "y", "z")])
    doc = header.get("resolved_case", {})
    pressure = doc.get("physics", {}).get("thermodynamics", {}).get("pressure_pa", header.get("thermodynamic_pressure_pa"))
    if pressure is not None:
        fields["thermodynamic_pressure_pa"] = np.asarray(pressure)
    return FieldArchive(fields)


def save_transformed_checkpoint(path, target_config, **fields):
    """Persist a transformed state as initialization input, not exact resume."""
    from jaxwind.config.document import load_case
    from jaxwind import AtmosphericSolution, StaggeredVelocity
    from jaxwind.io.checkpoint import save_checkpoint
    case = load_case(target_config)
    vector = lambda name: StaggeredVelocity(*(fields[name + "_" + axis] for axis in ("x", "y", "z")))
    if "pressure_dynamic_pa" in fields:
        from jaxwind.formulations.low_mach_abl import LowMachABLState
        state = LowMachABLState(vector("velocity"), fields["pressure_dynamic_pa"], fields["density"], vector("momentum_tendency"),
                                fields["temperature"], fields["water_vapor_mass_fraction"], fields["nitrogen_mass_fraction"],
                                fields.get("continuity_error_kg_m3_s", np.asarray(0., dtype=fields["density"].dtype)), fields["time"], fields["step"])
        formulation = "low-mach-abl"
    else:
        state = AtmosphericSolution(vector("velocity"), fields["pressure"], vector("momentum_tendency"), fields["scalar"], fields["scalar_tendency"], fields["time"], fields["step"])
        formulation = "boussinesq"
    from jaxwind.config.abl import load_fv_abl
    grid = load_fv_abl(case).physical.physical_grid
    metadata = {"formulation": formulation, "fingerprint": case.fingerprint, "initialization_only": True,
                "resolved_case": case.document, "units": "SI", "mesh": {axis + "_faces": np.asarray(getattr(grid, axis + "_faces")).tolist() for axis in ("x", "y", "z")}}
    if "thermodynamic_pressure_pa" in fields:
        metadata["thermodynamic_pressure_pa"] = float(fields["thermodynamic_pressure_pa"])
    save_checkpoint(path, state, metadata=metadata)
