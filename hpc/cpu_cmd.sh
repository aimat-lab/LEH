#!/bin/bash
# Run one command in the project directory on a CPU node. Usage: sbatch [--dependency=..] hpc/cpu_cmd.sh "command ..."
#SBATCH --job-name=cpu_cmd
#SBATCH --partition=cpu_il,cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000mb
#SBATCH --time=04:00:00
#SBATCH --output=logs/cpu_%j.out
cd /path/to/workspace/transferable_excited_MLIP
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONWARNINGS=ignore
echo "host $(hostname) start $(date) cmd: $1"
eval "$1"
echo "end $(date)"
