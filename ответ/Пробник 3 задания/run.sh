#!/bin/bash
#SBATCH --job-name=gemm_blas
#SBATCH --partition=rocky
#SBATCH --constraint=type_d
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:30:00
#SBATCH --output=gemm_blas_%j.out

set -e

cd "$SLURM_SUBMIT_DIR"

module load intel/oneAPI_2025_env

echo "========================================"
echo "JOB ID: $SLURM_JOB_ID"
echo "NODE: $SLURM_JOB_NODELIST"
echo "========================================"

lscpu | grep -E "Model name|Socket|Core|Thread"

export OMP_NUM_THREADS=24
export OMP_DYNAMIC=FALSE
export OMP_PLACES=cores
export OMP_PROC_BIND=close

export OPENBLAS_NUM_THREADS=24

export MKL_NUM_THREADS=24
export MKL_DYNAMIC=FALSE
export MKL_THREADING_LAYER=GNU

echo ""
echo "========== OPENBLAS =========="
srun --cpu-bind=cores ./gemm_openblas_omp
echo ""
echo "========== MKL =========="
srun --cpu-bind=cores ./gemm_mkl

echo ""
echo "========== DONE =========="
