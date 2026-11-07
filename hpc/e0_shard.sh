#!/bin/bash
# S0 energy + gradient at QCDGE minima (compute_e0.py), one shard.
# Usage: sbatch --array=0-(N-1) hpc/e0_shard.sh TASKS(comma-separated) OUTDIR N WORKERS THREADS TIME_LIMIT_H
#SBATCH --job-name=wb97_e0
#SBATCH --partition=cpu_il,cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64000mb
#SBATCH --time=01:00:00
#SBATCH --output=logs/e0_%A_%a.out
TASKS=$1; OUTDIR=$2; NSH=$3; WORKERS=$4; THREADS=$5; TL=$6
B=/path/to/workspace
W=$B/transferable_excited_MLIP
export TURBODIR=$B/TURBOMOLE_7.5.1
export PATH=$TURBODIR/scripts:$TURBODIR/bin/em64t-unknown-linux-gnu_smp:$PATH
export MKL_NUM_THREADS=1
export TM_SCRATCH=${TMPDIR:-/tmp}/e0_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p $TM_SCRATCH
cd $W
echo "host $(hostname) shard $SLURM_ARRAY_TASK_ID/$NSH start $(date)"
.venv/bin/python experiments/dataset/compute_e0.py --tasks ${TASKS//,/ } --out $OUTDIR --nshards $NSH \
    --shard $SLURM_ARRAY_TASK_ID --workers $WORKERS --threads $THREADS --time_limit_h $TL
echo "end $(date)"
