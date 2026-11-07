#!/bin/bash
# TURBOMOLE single points for one shard of the nonequilibrium dataset.
# Usage: sbatch --array=0-(N-1) hpc/tm_shard.sh N WORKERS THREADS [extra generate.py args]
#SBATCH --job-name=tm_data
#SBATCH --partition=cpu_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=200000mb
#SBATCH --time=03:00:00
#SBATCH --output=logs/tm_%A_%a.out
NSH=$1; WORKERS=$2; THREADS=$3; shift 3
W=/path/to/workspace/transferable_excited_MLIP
export TURBODIR=/path/to/workspace/TURBOMOLE_7.5.1
export PATH=$TURBODIR/scripts:$TURBODIR/bin/em64t-unknown-linux-gnu_smp:$PATH
export TM_SCRATCH=${TMPDIR:-/tmp}/tm_scratch_$SLURM_JOB_ID
mkdir -p $TM_SCRATCH
cd $W
echo "host $(hostname) shard $SLURM_ARRAY_TASK_ID/$NSH workers $WORKERS threads $THREADS start $(date)"
.venv/bin/python experiments/tm_data/generate.py --no_prepare --workers $WORKERS --threads $THREADS \
    --shard $SLURM_ARRAY_TASK_ID --nshards $NSH "$@"
echo "end $(date)"
