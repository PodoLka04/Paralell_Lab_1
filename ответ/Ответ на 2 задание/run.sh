#!/bin/bash
#SBATCH --job-name=gemm4_win
#SBATCH --partition=rocky
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --constraint=type_d
#SBATCH --time=00:25:00
#SBATCH --output=gemm4_%j.out

echo "========================================"
echo "SLURM JOB INFORMATION"
echo "========================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURM_JOB_NODELIST"
echo "CPUs:         $SLURM_CPUS_PER_TASK"
echo "Working dir:  $(pwd)"
echo ""

echo "Files in current dir:"
ls -la gemm4

echo ""
echo "CPU information:"
lscpu | grep -E "Model name|Socket|Core|Thread"
echo ""

echo "========================================"
echo "PROGRAM"
echo "========================================"

export OMP_NUM_THREADS=24
export OMP_PLACES=cores
export OMP_PROC_BIND=close

srun --cpu-bind=cores ./gemm

echo "Exit code: $?"