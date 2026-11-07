#!/bin/bash
# CPU variant of hpc/train_chain.sh for small models when the GPU queue is blocked (4 runs x 16 threads, 36 h jobs). Each run checkpoints to
# <out>/last.pt and exits with code 3 when its per-job wall budget (WALL_HOURS, default 35 h here) is reached; runs
# without summary.json that made progress in this job are resubmitted (at most CHAIN_MAX times, default 12).
# Usage: sbatch hpc/train_cpu.sh path/to/job1.yaml [path/to/job2.yaml ...]
#SBATCH --job-name=exmlip_cpu
#SBATCH --partition=cpu_il,cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=120000mb
#SBATCH --time=36:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --output=logs/train_cpu_%j.out
W=/path/to/workspace/transferable_excited_MLIP
cd $W
N=${CHAIN_N:-0}; MAX=${CHAIN_MAX:-12}; WALL=${WALL_HOURS:-35.0}; THREADS=${THREADS:-16}
CFGS=()
for cfg in "$@"; do  # skip finished runs (summary.json), so follow-up jobs with afterany dependencies are harmless
    [ -f $(dirname $cfg)/summary.json ] && echo "already finished: $cfg" || CFGS+=($cfg)
done
[ ${#CFGS[@]} -eq 0 ] && { echo "nothing to do"; exit 0; }
START=$(date +%s)
echo "host $(hostname) gpu $CUDA_VISIBLE_DEVICES start $(date) chain $N/$MAX wall $WALL h"
echo "cpu training: $THREADS threads per run"
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
            sbatch --export=ALL,CHAIN_N=$((N + 1)),CHAIN_MAX=$MAX,WALL_HOURS=$WALL,THREADS=$THREADS hpc/train_cpu.sh "${left[@]}"
        else
            echo "chain limit reached, not resubmitting: ${left[*]}"
        fi
    fi
}
trap 'echo "USR1 (time limit near) $(date)"; resubmit; kill $(jobs -p) 2>/dev/null; wait; exit 0' USR1
for cfg in "${CFGS[@]}"; do
    d=$(dirname $cfg)
    .venv/bin/python -m exmlip.training.train $cfg --set wall_hours=$WALL threads=$THREADS device=cpu >> $d/stdout.log 2>&1 &
    echo "started $cfg pid $!"
done
wait
resubmit
echo "end $(date)"
