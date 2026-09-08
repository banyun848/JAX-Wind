"""Small compute-node contracts for state, diagnostics, and stage resume."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from jaxwind.config.document import ResolvedCase, load_case
from jaxwind.config.toml import dumps
from jaxwind.io.checkpoint import checkpoint_metadata
from jaxwind.io.state_fields import state_fields
from jaxwind.runtime.engine import run, resume

ROOT = Path(__file__).resolve().parents[1]


def tiny_case(tmp_path, *, adaptive=False):
    base = load_case(ROOT / "cases/Andren1994/config.toml")
    document = deepcopy(base.document)
    document["mesh"]["cells"] = [8, 8, 8]
    document["case"]["profile_resampling"] = "linear"
    document["numerics"].update(pressure_backend="fft", time_integration="fast-rk3")
    document["time"].update(dt_seconds=.01, steps=6, chunk_steps=2, checkpoint_every_steps=2)
    document["diagnostics"].update(sample_start_step=0, sample_every_steps=2)
    document["output"]["directory"] = str(tmp_path / "run")
    if adaptive:
        document["time"]["cfl"] = .5
    return ResolvedCase(base.source, document)


@pytest.mark.parametrize("adaptive", [False, True])
def test_resume_preserves_complete_state_and_statistics(tmp_path, adaptive):
    case = tiny_case(tmp_path, adaptive=adaptive)
    whole = run(case, output=tmp_path / "whole")
    first = run(case, output=tmp_path / "split", max_steps=2)
    assert first.summary["status"] == "paused"
    final = resume(tmp_path / "split")
    assert final.summary["status"] == "complete"
    expected, _ = state_fields(whole.checkpoint)
    actual, _ = state_fields(final.checkpoint)
    assert expected.keys() == actual.keys()
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)
    assert (whole.output / "profiles.csv").read_text() == (final.output / "profiles.csv").read_text()


def test_existing_output_and_incompatible_resume_are_rejected(tmp_path):
    case = tiny_case(tmp_path)
    result = run(case, max_steps=2)
    with pytest.raises(FileExistsError):
        run(case)
    changed = deepcopy(case.document)
    changed["time"]["dt_seconds"] *= 2
    (result.output / "resolved_case.toml").write_text(dumps(changed))
    with pytest.raises(ValueError, match="configuration differs"):
        resume(result.output)


def test_historical_checkpoint_is_rejected(tmp_path):
    path = tmp_path / "old.npz"
    np.savez(path, velocity_x=np.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match="historical"):
        checkpoint_metadata(path)


def test_cryogenic_resume_preserves_parcels_and_all_histories(tmp_path):
    base = load_case(ROOT / "cases/HITSZLiquidNitrogenJet/fv_256.toml")
    document = deepcopy(base.document)
    document["mesh"]["cells"] = [8, 8, 8]
    document["time"].update(steps=4, chunk_steps=2, checkpoint_every_steps=2)
    document["physics"]["jet"].update(maximum_parcels=16, parcels_per_step=2)
    document["output"]["directory"] = str(tmp_path / "jet")
    case = ResolvedCase(base.source, document)
    whole = run(case, output=tmp_path / "whole")
    run(case, output=tmp_path / "split", max_steps=2)
    split = resume(tmp_path / "split")
    expected, _ = state_fields(whole.checkpoint)
    actual, _ = state_fields(split.checkpoint)
    assert "parcels_mass" in actual and "temperature_tendency" in actual
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)


def test_differentiable_inlet_remains_an_explicit_jax_input():
    import jax
    import jax.numpy as jnp
    from dataclasses import replace
    from jaxwind.config.jet import load_case as load_jet
    from jaxwind.simulation.jet import build_simulation
    native = replace(load_jet(ROOT / "cases/HITSZLiquidNitrogenJet/fv_256_incompressible_rk3_inlet.toml"),
                     cells=(8, 8, 8), maximum_parcels=16, parcels_per_step=2)
    grid, jet, microphysics, initialize, advance, courant, control = build_simulation(native, differentiable_inlet=True)
    def objective(scale):
        current_control = control._replace(speed_scale=scale)
        initial = initialize(current_control)
        final = advance(initial, current_control, 1)
        return jnp.sum(final.velocity.x)
    derivative = jax.grad(objective)(control.speed_scale)
    assert np.isfinite(float(derivative))


def test_workflow_resume_and_coverage(tmp_path):
    from jaxwind.workflows.engine import execute
    case = tiny_case(tmp_path)
    case.document["workflow"].update(warmup_steps=4, precursor_steps=4, main_steps=4,
                                      record_plane=1, chunk_steps=2, output_directory=str(tmp_path / "workflow"))
    path = tmp_path / "case.toml"
    path.write_text(dumps(case.document))
    first = execute(path, max_steps=2)
    assert first["stages"]["warmup"]["status"] == "paused"
    final = execute(path, resume=True)
    assert all(value["status"] == "complete" for value in final["stages"].values())
    assert (tmp_path / "workflow/precursor/inflow/metadata.json").is_file()
    case.document["workflow"]["main_steps"] = 5
    case.document["workflow"]["output_directory"] = str(tmp_path / "invalid")
    path.write_text(dumps(case.document))
    with pytest.raises(ValueError):
        execute(path)


def test_low_mach_continuation_consumes_new_stage_checkpoint(tmp_path):
    from jaxwind.workflows.engine import execute
    path = ROOT / "cases/workflows/low_mach_continuation.toml"
    output = tmp_path / "continuation"
    paused = execute(path, output=output, max_steps=2)
    assert paused["stages"]["warmup"]["status"] == "paused"
    completed = execute(path, output=output, resume=True)
    assert completed["stages"]["warmup"]["step"] == 6
    assert completed["stages"]["continuation"]["step"] == 12
    assert completed["stages"]["continuation"]["status"] == "complete"
    assert checkpoint_metadata(output / "continuation/checkpoint.npz")["initial_step"] == 6
