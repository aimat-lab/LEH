#!/bin/bash
# One X-MACE run continued in 8 h blocks until --epochs is reached (resubmits itself).
# Usage: sbatch hpc/xmace_chain.sh "<xmace_run.py args incl. --run DIR --epochs N>"
#SBATCH --job-name=xmace
#SBATCH --partition=gpu_h100,gpu_a100_il,gpu_h100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000mb
#SBATCH --time=08:00:00
#SBATCH --output=logs/xmace_%j.out
cd /path/to/workspace/transferable_excited_MLIP
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=4
ARGS="$1"
N=${CHAIN_N:-0}; MAX=${CHAIN_MAX:-10}
RUN0=$(echo "$ARGS" | sed -E 's/.*--run ([^ ]+).*/\1/'); BEFORE=$(cat $RUN0/epochs_done.txt 2>/dev/null || echo 0)
echo "host $(hostname) start $(date) args: $ARGS"
.venv_xmace/bin/python experiments/baselines/xmace_run.py $ARGS --resume --hours 7.3
RUN=$(echo "$ARGS" | sed -E 's/.*--run ([^ ]+).*/\1/'); TARGET=$(echo "$ARGS" | sed -E 's/.*--epochs ([0-9]+).*/\1/')
DONE=$(cat $RUN/epochs_done.txt 2>/dev/null || echo 0)
echo "epochs done $DONE of $TARGET"
# resubmit only after real progress in this job and below the chain limit (a fast failure must not loop)
if [ "$DONE" -lt "$TARGET" ] && [ "$DONE" -gt "$BEFORE" ] && [ $N -lt $MAX ]; then
    sbatch --parsable --export=ALL,CHAIN_N=$((N + 1)),CHAIN_MAX=$MAX hpc/xmace_chain.sh "$ARGS"
elif [ "$DONE" -lt "$TARGET" ]; then
    echo "NOT resubmitting: epochs before $BEFORE, after $DONE, chain $N/$MAX"
fi
echo "end $(date)"
