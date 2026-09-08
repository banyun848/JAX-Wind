from copy import deepcopy
from pathlib import Path
import pytest

from jaxwind.config.document import derive_case, load_case, validate
from jaxwind.config.toml import dumps

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "cases/Andren1994/config.toml"


def test_derivation_preserves_inputs_and_isolates_outputs(tmp_path):
    destination = tmp_path / "fine.toml"
    derive_case(BASE, destination, cells=[80, 64, 32], cfl=.5)
    original, derived = load_case(BASE), load_case(destination)
    assert derived.document["mesh"]["cells"] == [80, 64, 32]
    assert derived.document["time"]["cfl"] == .5
    assert derived.document["case"]["initial_profile"] == original.document["case"]["initial_profile"]
    assert derived.output != original.output
    assert derived.document["physics"] == original.document["physics"]
    with pytest.raises(FileExistsError):
        derive_case(BASE, destination, cfl=.2)


@pytest.mark.parametrize("cfl", [0, -1, True, float("nan"), float("inf")])
def test_invalid_cfl_is_rejected(cfl):
    document = deepcopy(load_case(BASE).document)
    document["time"]["cfl"] = cfl
    with pytest.raises(ValueError):
        validate(document)


def test_fixed_step_formulation_rejects_cfl():
    case = load_case(ROOT / "cases/HITSZLiquidNitrogenJet/fv_256.toml")
    case.document["time"]["cfl"] = .5
    with pytest.raises(ValueError, match="fixed timesteps"):
        validate(case.document)


def test_cycles_and_unknown_sections_are_rejected(tmp_path):
    path = tmp_path / "cycle.toml"
    path.write_text('extends = "cycle.toml"\n')
    with pytest.raises(ValueError, match="cycle"):
        load_case(path)
    document = deepcopy(load_case(BASE).document)
    document["typo"] = {}
    with pytest.raises(ValueError, match="unknown"):
        validate(document)


def test_resolved_document_round_trip(tmp_path):
    case = load_case(BASE)
    path = tmp_path / "resolved.toml"
    path.write_text(dumps(case.document))
    assert load_case(path).fingerprint == case.fingerprint


def test_derived_low_mach_mesh_reaches_source_builder(tmp_path):
    from jaxwind.config.low_mach import load_case as load_low
    from jaxwind.config.stages import load_workflow
    from jaxwind.io.initial_conditions import load_initial_profile
    path = tmp_path / "low.toml"
    derive_case(ROOT / "cases/HITSZWindTunnel/fv_low_mach_warmup_256x64x32_900s.toml", path, cells=[8, 8, 8])
    native = load_low(path)
    physical = load_workflow(native.source_workflow).case.physical
    assert (physical.physical_grid.nx, physical.physical_grid.ny, physical.physical_grid.nz) == (8, 8, 8)
    assert load_initial_profile(physical).shape == (8,)


def test_unknown_time_setting_is_rejected():
    document = deepcopy(load_case(BASE).document)
    document["time"]["cfl_threshold_typo"] = .5
    with pytest.raises(ValueError, match="unknown time"):
        validate(document)


def test_cli_derivation_and_workflow_check(tmp_path, capsys):
    from jaxwind.cli.main import main
    path = tmp_path / "derived.toml"
    assert main(["case", "derive", str(BASE), "--output", str(path), "--cells", "8", "8", "8", "--cfl", "0.5"]) == 0
    assert main(["check", str(path)]) == 0
    assert main(["check", str(ROOT / "cases/workflows/low_mach_continuation.toml")]) == 0
    assert "continuation" in capsys.readouterr().out


def test_workflow_cycles_are_rejected():
    from jaxwind.workflows.engine import _order
    nodes = {"a": {"inputs": {"checkpoint": "@b/checkpoint"}},
             "b": {"inputs": {"checkpoint": "@a/checkpoint"}}}
    with pytest.raises(ValueError, match="cycle"):
        _order(nodes)
