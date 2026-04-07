#!/bin/bash
# =============================================================================
# SLURM job script — CMA-ES tile optimization on NTNU Idun (CPU)
# Usage: sbatch idun_optimize.slurm [path/to/experiment]
# =============================================================================

#SBATCH --job-name=esn-optimize
#SBATCH --account=share-ie-idi      # TODO: replace with your Idun account
#SBATCH --partition=CPUQ
#SBATCH --time=1-00:00:00              # 1 day wall time (adjust as needed)
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --constraint=pec6520
#SBATCH --output=slurm-optimize-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leon.hesthaug@gmail.com       # TODO: replace with your email

set -euo pipefail

# --- Environment -------------------------------------------------------------
module purge
module load Python/3.11.3-GCCcore-12.3.0
source /cluster/home/$USER/esn-venv/bin/activate

# --- Working directory --------------------------------------------------------
cd ${SLURM_SUBMIT_DIR}

# --- Experiment path (default or from first argument) -------------------------
EXP_PATH="${1:-./experiments/v2/tiles/sizes/narma10/idun/}"

echo "========================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $(hostname)"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Experiment:    $EXP_PATH"
echo "Started:       $(date)"
echo "========================================"

# --- Prevent BLAS thread contention across workers ----------------------------
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# --- Parallel workers (matches allocated cores) -------------------------------
export ESN_WORKERS=$SLURM_CPUS_PER_TASK

# --- Run optimization ---------------------------------------------------------
python optimize.py "$EXP_PATH"

echo "Finished: $(date)"
