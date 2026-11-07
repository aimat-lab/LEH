#!/bin/bash
# HoreKa2: one X-MACE run continued in 2-day blocks until --epochs is reached (resubmits itself).
# Usage: sbatch hpc/hk_xmace_chain.sh "<xmace_run.py args incl. --run DIR --epochs N>"
#SBATCH --job-name=xmace
#SBATCH --partition=gpu-h100,gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120000mb
#SBATCH --time=48:00:00
#SBATCH --output=logs/xmace_%j.out
source /path/to/workspace/transferable_excited_MLIP/hpc/hk_env.sh; cd $P
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=4
ARGS="$1"
N=${CHAIN_N:-0}; MAX=${CHAIN_MAX:-10}
RUN0=$(echo "$ARGS" | sed -E 's/.*--run ([^ ]+).*/\1/'); BEFORE=$(cat $RUN0/epochs_done.txt 2>/dev/null || echo 0)
echo "host $(hostname) start $(date) args: $ARGS"
# training stops at XM_HOURS; the finalisation that follows (MACE evaluates train/valid/test of the full set) took
# > 1.5 h on the full set, so leave 6 h before the 48 h limit (job 75203 was killed there before epochs_done.txt)
.venv_xmace/bin/python experiments/baselines/xmace_run.py $ARGS --resume --hours ${XM_HOURS:-42}
RUN=$(echo "$ARGS" | sed -E 's/.*--run ([^ ]+).*/\1/'); TARGET=$(echo "$ARGS" | sed -E 's/.*--epochs ([0-9]+).*/\1/')
DONE=$(cat $RUN/epochs_done.txt 2>/dev/null || echo 0)
echo "epochs done $DONE of $TARGET"
# resubmit only after real progress in this job and below the chain limit (a fast failure must not loop)
if [ "$DONE" -lt "$TARGET" ] && [ "$DONE" -gt "$BEFORE" ] && [ $N -lt $MAX ]; then
    sbatch --parsable --export=ALL,CHAIN_N=$((N + 1)),CHAIN_MAX=$MAX hpc/hk_xmace_chain.sh "$ARGS"
elif [ "$DONE" -lt "$TARGET" ]; then
    echo "NOT resubmitting: epochs before $BEFORE, after $DONE, chain $N/$MAX"
fi
echo "end $(date)"
