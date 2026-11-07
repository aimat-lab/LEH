#!/bin/bash
# HoreKa2: evaluate the active-learning convergence criterion (experiments/al/convergence.py) for finished runs,
# without submitting a further sampling round (last round / production models). Submit with a dependency on training:
#   sbatch --dependency=afterany:J1:J2 hpc/hk_al_eval.sh TAG PREV_JSON RUN [RUN ...]
#   e.g. r4 experiments/al/convergence_r3.json experiments/al/runs/sampler_v4_s0 experiments/al/runs/sampler_v4_s1
#SBATCH --job-name=al_eval
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120000mb
#SBATCH --time=06:00:00
#SBATCH --output=logs/al_eval_%j.out
source /path/to/workspace/transferable_excited_MLIP/hpc/hk_env.sh
cd $P
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=32
TAG=$1; PREV=$2; shift 2
echo "host $(hostname) start $(date) tag $TAG prev $PREV runs $*"
for r in "$@"; do
    if [ ! -f $r/summary.json ]; then echo "STOP: $r has no summary.json (training not finished)"; exit 1; fi
done
.venv/bin/python experiments/al/convergence.py --tag $TAG "$@" --prev $PREV || { echo "STOP: convergence.py failed"; exit 1; }
echo "end $(date)"
