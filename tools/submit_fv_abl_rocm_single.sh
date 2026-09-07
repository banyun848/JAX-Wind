#!/usr/bin/env bash
#SBATCH --job-name=jaxwind-fv
#SBATCH --partition=hx1hdnormal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:1
#SBATCH --time=01:00:00
#SBATCH --output=jaxwind-fv-%j.out
#SBATCH --error=jaxwind-fv-%j.err

set -euo pipefail

# Examples (submit from the JAX-Wind repository root):
#
#   sbatch tools/submit_fv_abl_rocm_single.sh
#   sbatch --export=ALL,CASE_DIR=cases/Andren1994 \
#     tools/submit_fv_abl_rocm_single.sh
#   sbatch --export=ALL,CASE_DIR=cases/HITSZWindTunnel,STAGE=all \
#     tools/submit_fv_abl_rocm_single.sh
#   sbatch --export=ALL,CONFIG=cases/HITSZWindTunnel/fv_workflow.toml,OVERWRITE=1 \
#     tools/submit_fv_abl_rocm_single.sh
#
# CONFIG overrides CASE_DIR/CONFIG_NAME. Relative paths are resolved from
# REPO_ROOT, which defaults to the directory where sbatch was invoked.
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
CASE_DIR="${CASE_DIR:-cases/HITSZWindTunnel}"
CONFIG_NAME="${CONFIG_NAME:-fv_workflow.toml}"
CONFIG="${CONFIG:-${CASE_DIR}/${CONFIG_NAME}}"
STAGE="${STAGE:-warmup}"
CONDA_ENV="${CONDA_ENV:-jax060}"
MAX_STEPS="${MAX_STEPS:-}"
OVERWRITE="${OVERWRITE:-0}"
JOB_META_DIR="${JOB_META_DIR:-slurm_runs/${SLURM_JOB_ID}}"

cd "${REPO_ROOT}"

if [[ ! -f "${CONFIG}" ]]; then
    echo "ERROR: configuration not found: ${REPO_ROOT}/${CONFIG}" >&2
    exit 2
fi
case "${STAGE}" in
    warmup|precursor|main|all) ;;
    *)
        echo "ERROR: STAGE must be warmup, precursor, main, or all" >&2
        exit 2
        ;;
esac
if [[ "${OVERWRITE}" != "0" && "${OVERWRITE}" != "1" ]]; then
    echo "ERROR: OVERWRITE must be 0 or 1" >&2
    exit 2
fi
if [[ -n "${MAX_STEPS}" && ! "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: MAX_STEPS must be a positive integer" >&2
    exit 2
fi

module purge
module load compiler/dtk/26.04
module load mpi/openmpi/openmpi-4.1.5-gcc9.3.0

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# DTK supplies the gcvm dependency required by the JAX ROCm plugin.
export LD_LIBRARY_PATH="/public/software/compiler/dtk-26.04/dcc/gcvm/lib:${LD_LIBRARY_PATH:-}"

# The DTK FFT stack can reference OpenMPI symbols even in this single-process
# run, so preload the MPI library from the loaded module.
MPI_LIBDIR="$(mpicc --showme:libdirs | awk '{print $1}')"
export LD_PRELOAD="${MPI_LIBDIR}/libmpi.so${LD_PRELOAD:+:${LD_PRELOAD}}"

# The source checkout remains importable even if it was not installed with
# `python -m pip install -e .` in the selected environment.
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export JAX_PLATFORMS=rocm
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

mkdir -p "${JOB_META_DIR}"

echo "Job ID       : ${SLURM_JOB_ID}"
echo "Node         : $(hostname)"
echo "Repository   : ${REPO_ROOT}"
echo "Configuration: ${CONFIG}"
echo "Stage        : ${STAGE}"
echo "Python       : $(command -v python)"
echo "Conda env    : ${CONDA_ENV}"
echo "Metadata     : ${JOB_META_DIR}"
echo

scontrol show job "${SLURM_JOB_ID}" > "${JOB_META_DIR}/slurm_job.txt"
module list 2> "${JOB_META_DIR}/modules.txt"
conda list > "${JOB_META_DIR}/conda_packages.txt"
(rocm-smi --showproductname --showmeminfo vram || hy-smi || true) \
    > "${JOB_META_DIR}/gpu.txt" 2>&1

# Fail before the expensive simulation if the driver, plugin, or runtime
# libraries are unavailable on the allocated compute node.
python - <<'PY'
import jax

print("JAX version:", jax.__version__)
print("JAX devices:", jax.devices("rocm"))
PY

workflow_args=("${CONFIG}" --stage "${STAGE}")
if [[ -n "${MAX_STEPS}" ]]; then
    workflow_args+=(--max-steps "${MAX_STEPS}")
fi
if [[ "${OVERWRITE}" == "1" ]]; then
    workflow_args+=(--overwrite)
fi

srun \
    --ntasks=1 \
    --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    --tres-per-task=gres/dcu:1 \
    --kill-on-bad-exit=1 \
    python -u -m applications.fv_abl.workflow "${workflow_args[@]}"

echo
echo "Workflow completed successfully: $(date --iso-8601=seconds)"
