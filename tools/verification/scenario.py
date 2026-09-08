"""Self-contained legacy/candidate trajectories, run only by compute verification."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def fields(state, prefix=""):
    import numpy as np
    if hasattr(state, "_fields"):
        result = {}
        for key in state._fields:
            result.update(fields(getattr(state, key), prefix + key + "_"))
        return result
    return {prefix.rstrip("_"): np.asarray(state)}


def tiny_profile(original, destination):
    """Identical explicit input transformation for both comparison phases."""
    import numpy as np
    text = original.read_text()
    profile = re.search(r'(?m)^initial_profile\s*=\s*"([^"]+)"', text)[1]
    lengths = json.loads(re.search(r'(?m)^lengths_m\s*=\s*(\[[^\]]+\])', text)[1])
    table = np.genfromtxt(original.parent / profile, delimiter=",", names=True)
    source_z = table["z_m"]
    source_faces = np.zeros(len(table) + 1)
    for index, center in enumerate(source_z):
        source_faces[index + 1] = 2 * center - source_faces[index]
    target_faces = np.linspace(0., lengths[2], 9)
    target_z = .5 * (target_faces[:-1] + target_faces[1:])
    columns = [target_z]
    for name in table.dtype.names[1:]:
        if name.startswith("w_upper"):
            values = np.interp(target_faces[1:], source_faces, np.concatenate(([0.], table[name])))
            values[-1] = 0.
        else:
            values = np.interp(target_z, source_z, table[name])
        columns.append(values)
    np.savetxt(destination, np.column_stack(columns), delimiter=",", header=",".join(table.dtype.names), comments="")
    return destination


def legacy_case(source, relative, destination, *, low=False):
    """Make bounded legacy input without depending on candidate loaders."""
    original = source / relative
    text = original.read_text()
    for key in ("initial_profile", "reference_results"):
        text = re.sub(rf'(?m)^({key}\s*=\s*)"([^"]+)"', lambda m: m[1] + json.dumps(str((original.parent / m[2]).resolve())), text)
    text = re.sub(r"(?m)^cells\s*=.*$", "cells = [8, 8, 8]", text)
    text = re.sub(r"(?m)^dt_seconds\s*=.*$", "dt_seconds = 0.01", text)
    text = re.sub(r"(?m)^steps\s*=.*$", "steps = 6", text)
    text = re.sub(r"(?m)^sample_start_step\s*=.*$", "sample_start_step = 0", text)
    text = re.sub(r"(?m)^sample_every_steps\s*=.*$", "sample_every_steps = 2", text)
    text = re.sub(r"(?m)^pressure_backend\s*=.*$", 'pressure_backend = "fft"', text)
    text = re.sub(r"(?m)^time_integration\s*=.*$", 'time_integration = "fast-rk3"', text)
    text = re.sub(r"(?m)^cfl_ceiling\s*=.*\n?", "", text)
    text = re.sub(r"(?m)^chunk_steps\s*=.*$", "chunk_steps = 2", text)
    text = re.sub(r"(?m)^(warmup_steps|precursor_steps|main_steps)\s*=.*$", r"\1 = 6", text)
    text = re.sub(r"(?m)^record_plane\s*=.*$", "record_plane = 1", text)
    text = re.sub(r"(?m)^(precursor_frame_count|main_frame_count)\s*=.*$", r"\1 = 2", text)
    text = re.sub(r"(?m)^output_directory\s*=.*$", "output_directory = " + json.dumps(str(destination.parent / "run")), text)
    if low:
        # Only the shared physical case is used to initialize low-Mach flow.
        text = re.sub(r"(?m)^(precursor_dt_seconds|main_dt_seconds)\s*=.*$", r"\1 = 0.01", text)
    profile = tiny_profile(original, destination.with_suffix(".csv"))
    text = re.sub(r'(?m)^initial_profile\s*=.*$', "initial_profile = " + json.dumps(str(profile)), text)
    destination.write_text(text)
    return destination


def candidate_case(source, relative, output, *, low=False):
    from jaxwind.config.document import load_case, ResolvedCase
    case = load_case(source / relative)
    doc = case.document
    doc["mesh"]["cells"] = [8, 8, 8]
    doc["case"]["initial_profile"] = str(tiny_profile(source / relative, output / "profile.csv"))
    doc["time"].update(dt_seconds=.01, steps=6, chunk_steps=2)
    doc["time"].pop("cfl", None)
    doc["numerics"].update(pressure_backend="fft", time_integration="fast-rk3")
    doc["diagnostics"].update(sample_start_step=0, sample_every_steps=2)
    doc["output"]["directory"] = str(output / "run")
    if "workflow" in doc:
        doc["workflow"].update(warmup_steps=6, precursor_steps=6, main_steps=6, record_plane=1, chunk_steps=2,
                               output_directory=str(output / "workflow"))
        # Match legacy_case: bound existing frame requests without adding new
        # ones, since frame boundaries also split compiled advance blocks.
        for key in ("precursor_frame_count", "main_frame_count"):
            if key in doc["workflow"]:
                doc["workflow"][key] = 2
        if low:
            for key in ("precursor_dt_seconds", "main_dt_seconds"):
                if key in doc["workflow"]:
                    doc["workflow"][key] = .01
    return ResolvedCase(case.source, doc)


def main():
    if os.environ.get("JAXWIND_COMPUTE_CONFIRMED") != "1":
        raise SystemExit("Compute-node confirmation required")
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("baseline", "candidate"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--scenario", required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(source / "src"), str(source)]
    import numpy as np
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", False)
    legacy = args.phase == "baseline"
    report = {"scenario": args.scenario, "phase": args.phase, "jax": jax.__version__,
              "devices": [str(device) for device in jax.devices()], "dtype": "float32",
              "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
              "xla_flags": os.environ.get("XLA_FLAGS", "")}
    report["revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    base_relative = "cases/Andren1994/config.toml"
    if args.scenario in {"boussinesq", "adaptive", "open"}:
        if legacy:
            from applications.fv_abl import workflow as wf
            path = legacy_case(source, base_relative, output / "case.toml")
            if args.scenario == "adaptive":
                path.write_text(path.read_text().replace("[finite_volume]", "[finite_volume]\ncfl_ceiling = 0.5"))
            configured = wf.load_workflow(path)
            state = wf._initial_periodic(configured.case, jax, jnp)
            step, fixed = wf._periodic_advance(configured.case)
            if args.scenario == "adaptive":
                from jaxwind import build_adaptive_atmospheric_run
                advance_native = build_adaptive_atmospheric_run(step, configured.case.physical.physical_grid, cfl_ceiling=.5, maximum_dt=.01)
                advance = lambda current, count: advance_native(current, .06, count)
            else:
                advance = lambda current, count: fixed(current, .01, count)
            if args.scenario == "open":
                configured.options.output_directory.mkdir()
                wf.run_warmup(configured, steps=6)
                wf.run_precursor(configured, steps=6)
                wf.run_main(configured, steps=6)
                state = wf._load_solution(configured.options.output_directory / "main_final.npz", jnp)
        else:
            from jaxwind import build_simulation, RunControls
            case = candidate_case(source, base_relative, output)
            if args.scenario == "adaptive":
                case.document["time"]["cfl"] = .5
            if args.scenario == "open":
                from jaxwind.config.toml import dumps
                from jaxwind.workflows.engine import execute
                from jaxwind.io.state_fields import state_fields
                path = output / "case.toml"
                path.write_text(dumps(case.document))
                workflow = execute(path)
                actual, _ = state_fields(output / "workflow" / "main" / "checkpoint.npz")
                state = None
            else:
                simulation = build_simulation(case)
                state = simulation.initialize()
                advance = lambda current, count: simulation.advance(current, RunControls(count, .06))
    elif args.scenario == "low-mach":
        relative = "cases/HITSZWindTunnel/fv_low_mach_warmup_256x64x32_900s.toml"
        shared = "cases/HITSZWindTunnel/fv_uniform_256x64x32_l24_low_mach_sanity.toml"
        if legacy:
            from applications.fv_low_mach_abl.run import load_case, build_simulation
            # The baseline has a stale attribute access in assembly, not a
            # different pressure algorithm. Supply the intended dtype without
            # editing the baseline checkout or changing any numerical kernel.
            from applications.fv_abl.case import BoussinesqCase
            from types import SimpleNamespace
            if not hasattr(BoussinesqCase, "pressure"):
                BoussinesqCase.pressure = property(lambda physical: SimpleNamespace(dtype=physical.dtype))
                report["baseline_assembly_adapter"] = "physical.pressure.dtype -> physical.dtype"
            native = load_case(source / relative)
            shared_path = legacy_case(source, shared, output / "shared.toml", low=True)
            native = replace(native, source_workflow=shared_path, steps=6, frame_count=2, dt=.01, chunk_steps=2)
            _, grid, state, advance, courant = build_simulation(native)
        else:
            from jaxwind import build_simulation, load_case, RunControls
            from jaxwind.config.toml import dumps
            shared_case = candidate_case(source, shared, output, low=True)
            shared_path = output / "shared.toml"
            shared_path.write_text(dumps(shared_case.document))
            case = load_case(source / relative)
            case.document["initial_conditions"] = {"source_case": str(shared_path), "initial_condition": "configured"}
            case.document["time"].update(dt_seconds=.01, steps=6, chunk_steps=2, frame_count=2)
            simulation = build_simulation(case)
            state = simulation.initialize()
            advance = lambda current, count: simulation.advance(current, RunControls(count, .06))
    else:
        name = "fv_256_incompressible_rk3.toml" if args.scenario == "jet-incompressible" else "fv_256.toml"
        relative = "cases/HITSZLiquidNitrogenJet/" + name
        if legacy:
            from applications.fv_ln2_jet.run import load_case, build_simulation
            native = replace(load_case(source / relative), cells=(8,8,8), steps=6, chunk_steps=2, maximum_parcels=32, parcels_per_step=2)
            grid, jet, microphysics, state, advance, courant = build_simulation(native)
        else:
            from jaxwind import build_simulation, load_case, RunControls
            case = load_case(source / relative)
            case.document["mesh"]["cells"] = [8,8,8]
            case.document["time"].update(steps=6, chunk_steps=2)
            case.document["physics"]["jet"].update(maximum_parcels=32, parcels_per_step=2)
            simulation = build_simulation(case)
            state = simulation.initialize()
            advance = lambda current, count: simulation.advance(current, RunControls(count, .003))
    timings = []
    if args.scenario != "open":
        # Identical block boundaries; first call compiles and is excluded.
        for _ in range(3):
            started = time.perf_counter()
            state = advance(state, 2)
            jax.block_until_ready(state)
            timings.append(time.perf_counter() - started)
        actual = fields(state)
    elif legacy:
        actual = fields(state)
    report["block_seconds"] = timings
    report["steady_seconds"] = sum(timings[1:]) if timings else None
    for name, values in actual.items():
        if not np.all(np.isfinite(values)):
            raise ValueError(f"nonfinite state: {name}")
    np.savez_compressed(output / "state.npz", **actual)
    report["status"] = "captured"
    if args.baseline:
        baseline_report = json.loads((args.baseline / "report.json").read_text())
        seed = report["python_hash_seed"]
        if seed is None or seed == "random" or seed != baseline_report.get("python_hash_seed"):
            raise ValueError("baseline and candidate require the same explicit PYTHONHASHSEED; recapture with run.sh")
        if report["xla_flags"] != baseline_report.get("xla_flags"):
            raise ValueError("baseline and candidate require matching XLA_FLAGS; recapture with run.sh")
        if report["jax"] != baseline_report["jax"] or report["devices"] != baseline_report["devices"]:
            raise ValueError("baseline and candidate require matching JAX versions and devices")
        with np.load(args.baseline / "state.npz", allow_pickle=False) as expected:
            if set(actual) != set(expected.files):
                raise AssertionError("state fields differ between baseline and candidate")
            report["maximum_absolute_error"] = {}
            for name, value in actual.items():
                reference = expected[name]
                if value.dtype.kind in "biu":
                    np.testing.assert_array_equal(value, reference, err_msg=name)
                else:
                    np.testing.assert_allclose(value, reference, rtol=2.e-5, atol=2.e-6, err_msg=name)
                report["maximum_absolute_error"][name] = float(np.max(np.abs(value.astype(float) - reference.astype(float)))) if value.size else 0.
        before, after = baseline_report["steady_seconds"], report["steady_seconds"]
        report["performance_review_required"] = bool(before and after and after > 1.05 * before)
        report["status"] = "passed"
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
