#!/bin/bash
# HoreKa2 version of hpc/train_chain.sh (gpu-h100/gpu-h200, 2-day jobs, 16 cores per GPU, up to 4 runs x 4 threads). Each run checkpoints to
# <out>/last.pt and exits with code 3 when its per-job wall budget (WALL_HOURS, default 46.5 h here) is reached; runs
# without summary.json that made progress in this job are resubmitted (at most CHAIN_MAX times, default 12).
# Usage: sbatch hpc/hk_train_chain.sh path/to/job1.yaml [path/to/job2.yaml ...]
#SBATCH --job-name=exmlip_chain
#SBATCH --partition=gpu-h100,gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180000mb
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --output=logs/train_%j.out
source /path/to/workspace/transferable_excited_MLIP/hpc/hk_env.sh; W=$P
cd $W
N=${CHAIN_N:-0}; MAX=${CHAIN_MAX:-12}; WALL=${WALL_HOURS:-46.5}; THREADS=${THREADS:-4}
CFGS=()
for cfg in "$@"; do  # skip finished runs (summary.json), so follow-up jobs with afterany dependencies are harmless
    [ -f $(dirname $cfg)/summary.json ] && echo "already finished: $cfg" || CFGS+=($cfg)
done
[ ${#CFGS[@]} -eq 0 ] && { echo "nothing to do"; exit 0; }
START=$(date +%s)
echo "host $(hostname) gpu $CUDA_VISIBLE_DEVICES start $(date) chain $N/$MAX wall $WALL h"
nvidia-smi --query-gpu=name,memory.total --format=csv
done_resubmit=0
resubmit() {
    [ $done_resubmit -eq 1 ] && return
    done_resubmit=1
    left=()
    for cfg in "${CFGS[@]}"; do
        d=$(dirname $cfg)
        [ -f $d/summary.json ] && continue
        if [ -f $d/last.pt ] && [ $(stat -c %Y $d/last.pt) -gt $START ]; then
            left+=($cfg)
        else
            echo "not resubmitting $cfg (no checkpoint progress in this job; see $d/stdout.log)"
        fi
    done
    if [ ${#left[@]} -gt 0 ]; then
        if [ $N -lt $MAX ]; then
            sbatch --export=ALL,CHAIN_N=$((N + 1)),CHAIN_MAX=$MAX,WALL_HOURS=$WALL,THREADS=$THREADS hpc/hk_train_chain.sh "${left[@]}"
        else
            echo "chain limit reached, not resubmitting: ${left[*]}"
        fi
    fi
}
trap 'echo "USR1 (time limit near) $(date)"; resubmit; kill $(jobs -p) 2>/dev/null; wait; exit 0' USR1
for cfg in "${CFGS[@]}"; do
    d=$(dirname $cfg)
    .venv/bin/python -m exmlip.training.train $cfg --set wall_hours=$WALL threads=$THREADS >> $d/stdout.log 2>&1 &
    echo "started $cfg pid $!"
done
wait
resubmit
echo "end $(date)"
