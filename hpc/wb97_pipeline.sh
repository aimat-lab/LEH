#!/bin/bash
# Pipeline step for the new dataset (see experiments/wb97/pipeline.py). Usage: sbatch [--dependency=...] hpc/wb97_pipeline.sh [--round N]
#SBATCH --job-name=wb97_pipe
#SBATCH --partition=cpu_il,cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000mb
#SBATCH --time=01:00:00
#SBATCH --output=logs/pipe_%j.out
cd /path/to/workspace/transferable_excited_MLIP
echo "host $(hostname) start $(date) args $@"
.venv/bin/python experiments/wb97/pipeline.py "$@"
echo "end $(date)"
