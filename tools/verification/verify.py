"""Compute-node verification coordinator. Each scenario runs in a fresh process."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SCENARIOS = ("boussinesq", "adaptive", "open", "low-mach", "jet-incompressible", "jet-low-mach")


def main():
    if os.environ.get("JAXWIND_COMPUTE_CONFIRMED") != "1":
        raise SystemExit("Use run.sh from a compute node; verification is forbidden on the login node")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("baseline", "candidate"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--platform", choices=("cpu", "cuda", "rocm"), default="cpu")
    parser.add_argument("--extended", action="store_true")
    args = parser.parse_args()
    if args.phase == "candidate" and args.baseline is None:
        parser.error("candidate requires --baseline")
    source, output = args.source.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    # Keep fresh-process compilation reproducible across the module moves.
    environment = dict(os.environ, PYTHONHASHSEED="0", JAX_PLATFORMS=args.platform, PYTHONPATH=os.pathsep.join((str(source / "src"), str(source))))
    if args.platform == "cuda":
        environment["XLA_FLAGS"] = (environment.get("XLA_FLAGS", "") +
                                    " --xla_gpu_deterministic_ops=true").strip()
    report = {"phase": args.phase, "source": str(source), "platform": args.platform,
              "python": sys.executable, "python_hash_seed": environment["PYTHONHASHSEED"], "xla_flags": environment.get("XLA_FLAGS", ""), "started": time.time(), "checks": [], "status": "running"}
    summary = output / "summary.json"

    def check(name, command):
        started = time.perf_counter()
        with (output / (name + ".log")).open("w") as log:
            process = subprocess.run(command, cwd=source, env=environment, stdout=log, stderr=subprocess.STDOUT)
        report["checks"].append({"name": name, "command": command, "returncode": process.returncode,
                                 "seconds": time.perf_counter() - started, "log": name + ".log"})
        summary.write_text(json.dumps(report, indent=2) + "\n")
        return process.returncode == 0

    ok = True
    for scenario in SCENARIOS:
        command = [sys.executable, str(Path(__file__).with_name("scenario.py")), args.phase,
                   "--source", str(source), "--output", str(output / scenario), "--scenario", scenario]
        if args.baseline:
            command += ["--baseline", str(args.baseline.resolve() / scenario)]
        ok = check(scenario, command) and ok
    if args.phase == "candidate":
        tests = ["tests/fv", "tests/physics", "tests/cases/test_hitsz.py", "tests/cases/test_fv_abl_core.py", "tests/cases/test_fv_precursor_workflow.py", "tests/test_case_schema.py",
                 "tests/test_runtime_contract.py", "tests/test_architecture.py", "tests/tools/test_prolong_fv_checkpoint.py"]
        if args.extended:
            tests = ["tests"]
        ok = check("pytest", [sys.executable, "-m", "pytest", "-q", "-o", "python_files=test_*.py", *tests]) and ok
        ok = check("compile", [sys.executable, "-m", "compileall", "-q", "src", "tools"]) and ok
        ok = check("undefined-names", [sys.executable, "-m", "ruff", "check", "--select", "F821,F822", "src", "tools", "tests"]) and ok
        # This subprocess runs outside the checkout, with no source PYTHONPATH,
        # so a missing installation cannot be masked by repository imports.
        outside_env = dict(environment)
        outside_env.pop("PYTHONPATH", None)
        with (output / "installed-cli.log").open("w") as log:
            result = subprocess.run([sys.executable, "-m", "jaxwind", "--help"], cwd=output,
                                    env=outside_env, stdout=log, stderr=subprocess.STDOUT)
        report["checks"].append({"name": "installed-cli", "returncode": result.returncode, "log": "installed-cli.log"})
        ok = result.returncode == 0 and ok
    report["status"] = "passed" if ok else "failed"
    report["finished"] = time.time()
    report["optional_backends"] = "AMG/accelerator coverage requires the corresponding environment; no fallback backend is selected"
    summary.write_text(json.dumps(report, indent=2) + "\n")
    print(summary)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
