#!/bin/bash
# bwUniCluster: TD-wB97X-D/6-31G* excitation vectors (escf, restart from the archived ground states or recomputed with
# the dataset settings) for the LEH eigenvector comparison and the population labels (experiments/eigvec/tddft/
# run_escf.py). Raw outputs kept in OUT.
# Usage: sbatch hpc/eigvec_escf.sh [WORKERS] [THREADS] [PICK] [SHARD k/n] [OUT] [SELECTION] [VARIANT]
#SBATCH --job-name=eigvec_escf
#SBATCH --partition=cpu_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=200000mb
#SBATCH --time=04:00:00
#SBATCH --output=logs/eigvec_escf_%j.out
WORKERS=${1:-32}; THREADS=${2:-4}; PICK=${3:-}; SHARD=${4:-0/1}; OUT=${5:-data/eigvec_tddft_raw}
SEL=${6:-experiments/eigvec/tddft/selection.json}; VARIANT=${7:-}
W=/path/to/workspace/transferable_excited_MLIP
export TURBODIR=/path/to/workspace/TURBOMOLE_7.5.1
export PATH=$TURBODIR/scripts:$TURBODIR/bin/em64t-unknown-linux-gnu_smp:$PATH
cd $W
echo "host $(hostname) workers $WORKERS threads $THREADS pick '$PICK' shard $SHARD out $OUT sel $SEL variant '$VARIANT' start $(date)"
.venv/bin/python experiments/eigvec/tddft/run_escf.py $OUT --selection $SEL --workers $WORKERS --threads $THREADS \
  ${PICK:+--pick $PICK} --shard $SHARD ${VARIANT:+--variant $VARIANT}
echo "end $(date)"
