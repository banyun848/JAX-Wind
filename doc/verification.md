# Verification on compute nodes

**The refactor has not been executed or numerically verified on the login node.**
All installation, Python, tests, compilation, simulations, and benchmarks below
belong inside a compute allocation. Reading/editing files is the only local
development activity used for this migration.

## Allocate a compute node

Obtain an allocation using your site's normal Slurm account/partition/resource
settings, then enter a compute step with `srun --pty bash`. An `salloc` shell
alone can still be on the login node. The runner requires both `SLURM_JOB_ID`
and `SLURMD_NODENAME`; for another scheduler, pass `--confirm-compute-node`
only after entering an actual compute node.

No commands in this document have been run by the coding agent.

## Prepare source and environment — compute node only

The pre-refactor baseline revision is:

    f7d83625e481f0fcba04e73bfb3482eff043a603

From the candidate checkout, create a separate baseline checkout in a new path:

```bash
git worktree add --detach /path/to/JAX-Wind-baseline \
  f7d83625e481f0fcba04e73bfb3482eff043a603
```

Set explicit absolute paths and select an environment with the appropriate JAX
CPU/CUDA/ROCm build. Use the **same Python/JAX versions and hardware** for both
phases. The scripts never install dependencies or silently select another device.

```bash
export JAXWIND_REPO=/raven/u/limo/JAX-Wind
export JAXWIND_BASE=/path/to/JAX-Wind-baseline
export JAXWIND_PYTHON=/path/to/environment/bin/python

"$JAXWIND_PYTHON" -m pip install -e "$JAXWIND_REPO[dev]"
```

Install your site's accelerator-enabled JAX environment separately if needed.
The optional AMG backend requires its own dependencies; the default trajectory
matrix uses FFT and GMG. Baseline imports are taken from its checkout rather
than the candidate installation.

## Capture baseline and compare candidate — compute node only

Choose new output directories. The runner refuses to overwrite an existing
directory. Substitute `cuda` or `rocm` for `cpu` on an appropriately allocated
and configured accelerator node.

```bash
bash "$JAXWIND_REPO/tools/verification/run.sh" baseline \
  --source "$JAXWIND_BASE" \
  --output /path/to/results/baseline-cpu --platform cpu

bash "$JAXWIND_REPO/tools/verification/run.sh" candidate \
  --source "$JAXWIND_REPO" \
  --baseline /path/to/results/baseline-cpu \
  --output /path/to/results/candidate-cpu --platform cpu
```

Both phases use 8×8×8 meshes and six-step trajectories. Cryogenic examples
use 32 parcel slots and two injected parcels per step. Scenarios cover periodic
Boussinesq flow, adaptive stepping, warmup/precursor/open flow, periodic low-Mach
flow, and incompressible/low-Mach cryogenic jets. Each runs in a fresh process.
Both phases explicitly resample the original tabulated profiles onto the same
small mesh before constructing fields. The baseline low-Mach source contains a
stale `physical.pressure.dtype` access although its case exposes `physical.dtype`.
That scenario supplies a recorded assembly-only property adapter; it does not
edit the baseline checkout or change pressure/integration kernels. The candidate
uses the actual dtype field. This adapter is listed in the baseline report.

Candidate verification also runs numerical, configuration, architecture,
checkpoint/resume, and workflow tests; compiles source; and checks the installed
CLI outside the checkout without a source `PYTHONPATH`. Add `--extended` to run
all tests, including optional reference-analysis and backend-specific tests.
Those may need additional dependencies and declared external inputs.
The candidate also checks undefined names with Ruff. This check, like all
other verification, is run only inside the compute-node script.

## Batch submission template

The template intentionally does not guess your partition, account, memory,
walltime, or GPU resource syntax. Supply these using your site's submission
procedure. It invokes the same guarded runner through a compute step.

```bash
# Submit using the site's permitted submission host; job execution is on compute.
sbatch --account=YOUR_ACCOUNT --partition=YOUR_PARTITION \
  --cpus-per-task=8 --mem=16G --time=01:00:00 \
  "$JAXWIND_REPO/tools/verification/submit.slurm" baseline \
  --source "$JAXWIND_BASE" --output /path/to/results/baseline-cpu --platform cpu
```

After the baseline job succeeds, submit the candidate phase with its baseline
directory. For GPU runs, also request the site's GPU resources and select the
matching platform. Activate the environment before submission or export the
absolute `JAXWIND_PYTHON` path. Scripts do not load site-specific modules.

## Read the report

Start with `summary.json` and the per-check logs. A required failed check causes
a nonzero exit status. Each successful trajectory writes state arrays and a
report containing device/version information, revision, field errors, and
block timings. Failed scenarios retain their traceback in the scenario log.

Integer/boolean fields compare exactly. Float32 trajectory comparisons use
`rtol=2e-5`, `atol=2e-6`; the numerical unit tests retain their own tighter
operator/conservation tolerances. Exact-resume tests compare complete state
arrays and accumulated profiles without tolerance.

The first timed block includes compilation and is excluded from steady timings.
The six-step timing check is a smoke measurement, not a performance conclusion.
A slowdown exceeding 5% sets `performance_review_required`; repeat measurements
on the same otherwise-idle node before treating it as a reproducible regression.
The dedicated profiling tools remain available for production-size investigation.

Send back `summary.json`, the scenario reports, and logs for failed checks.
Do not describe the migration as numerically accepted until these results have
been reviewed. Optional unavailable accelerator/backend coverage must be stated
separately from passed checks.
