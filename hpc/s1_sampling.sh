#!/bin/bash
# Model-driven S1/S2 sampling (experiments/dataset/s1_sampling.py), one shard per array task (restartable per chunk).
# Usage: sbatch --array=0-(N-1) hpc/s1_sampling.sh N
#SBATCH --job-name=s1r1_md
#SBATCH --partition=cpu_il,cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32000mb
#SBATCH --time=06:00:00
#SBATCH --output=logs/s1md_%A_%a.out
cd /path/to/workspace/transferable_excited_MLIP
export OMP_NUM_THREADS=16 PYTHONWARNINGS=ignore
echo "host $(hostname) shard $SLURM_ARRAY_TASK_ID/$1 start $(date)"
.venv/bin/python experiments/dataset/s1_sampling.py --data data/wb97_all_100k.pkl --zinc --nshards $1 \
    --shard $SLURM_ARRAY_TASK_ID --threads 16 --chunk 128 --fs 500 --time_limit_h 5.3
echo "end $(date)"
