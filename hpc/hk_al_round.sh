#!/bin/bash
# HoreKa2: after the sampling models of an active-learning round are trained (experiments/al/README.md, steps 3-4):
# check that both runs finished, evaluate the convergence criterion on the validation trajectories, and submit the
# S1 sampling of the next round only if the criterion is not met. Submit with a dependency on the training jobs:
#   sbatch --dependency=afterany:JOB1:JOB2 hpc/hk_al_round.sh TAG RUN_A RUN_B NEXT_ROUND [PREV_JSON] [FRAC_EXTRA]
#   e.g. TAG=r2 RUN_A=experiments/al/runs/sampler_v2_s0 RUN_B=..._s1 NEXT_ROUND=s1r3 PREV=experiments/al/convergence_r1_benchmark.json
#SBATCH --job-name=al_round
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120000mb
#SBATCH --time=04:00:00
#SBATCH --output=logs/al_round_%j.out
source /path/to/workspace/transferable_excited_MLIP/hpc/hk_env.sh
cd $P
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=32
TAG=$1; A=$2; B=$3; NEXT=$4; PREV=${5:-}; FX=${6:-0.05}
echo "host $(hostname) start $(date) tag $TAG runs $A $B next $NEXT prev $PREV"
for r in $A $B; do
    if [ ! -f $r/summary.json ]; then echo "STOP: $r has no summary.json (training not finished)"; exit 1; fi
done
.venv/bin/python experiments/al/convergence.py --tag $TAG $A $B ${PREV:+--prev $PREV} || { echo "STOP: convergence.py failed"; exit 1; }
if .venv/bin/python -c "import json,sys; sys.exit(0 if json.load(open('experiments/al/convergence_$TAG.json'))['converged'] else 1)"; then
    echo "CONVERGED after $TAG: no further sampling round submitted"
else
    J=$(sbatch --parsable hpc/hk_al_sampling.sh $NEXT $A $B $FX)
    echo "not converged: submitted sampling $NEXT as job $J"
fi
echo "end $(date)"
