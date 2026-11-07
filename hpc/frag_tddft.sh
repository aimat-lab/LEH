#!/bin/bash
# bwUniCluster: TD-wB97X-D/6-31G* references of the fragment figure on the shared pair geometries
# (experiments/frag_tddft/run_pairs.py; first NPAIRS pairs of each kind at 2.5-15 A). Raw TURBOMOLE outputs are kept
# in data/frag_tddft_raw (not node-local scratch), results in experiments/frag_tddft/results_pairs.json.
# Usage: sbatch hpc/frag_tddft.sh [NPAIRS] [WORKERS] [THREADS]
#SBATCH --job-name=frag_tddft
#SBATCH --partition=cpu_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=200000mb
#SBATCH --time=08:00:00
#SBATCH --output=logs/frag_tddft_%j.out
NP=${1:-10}; WORKERS=${2:-32}; THREADS=${3:-4}
W=/path/to/workspace/transferable_excited_MLIP
export TURBODIR=/path/to/workspace/TURBOMOLE_7.5.1
export PATH=$TURBODIR/scripts:$TURBODIR/bin/em64t-unknown-linux-gnu_smp:$PATH
cd $W
echo "host $(hostname) pairs $NP workers $WORKERS threads $THREADS start $(date)"
.venv/bin/python experiments/frag_tddft/run_pairs.py data/frag_tddft_raw --pairs $NP --workers $WORKERS --threads $THREADS
echo "end $(date)"
