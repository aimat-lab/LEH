#!/bin/bash
# HoreKa2: MS-ANI and the OMNI-P2x architecture (vecmsani) via MLatom on one GPU, same data/split for both.
# Usage: sbatch hpc/hk_msani.sh DATA SPLIT OUTPREFIX   (writes OUTPREFIX_msani/ and OUTPREFIX_vecmsani/)
# Training stops at MLatom's own convergence or after 46 h (then the best saved model predicts).
#SBATCH --job-name=msani
#SBATCH --partition=gpu-h100,gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180000mb
#SBATCH --time=48:00:00
#SBATCH --output=logs/msani_%j.out
source /path/to/workspace/transferable_excited_MLIP/hpc/hk_env.sh
cd $P
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=8
DATA=$1; SPLIT=$2; PRE=$3
echo "host $(hostname) start $(date) data $DATA split $SPLIT"; nvidia-smi --query-gpu=name --format=csv,noheader
for m in msani vecmsani; do
    d=${PRE}_$m; mkdir -p $d
    ( timeout 46h .venv_mlatom/bin/python experiments/baselines/msani_train.py --out $d --max_epochs 3000 --max_hours 45.5 \
          --mode wb97 --model $m --data $DATA --split $SPLIT > $d/stdout.log 2>&1;
      [ -f $d/predictions.pkl ] || .venv_mlatom/bin/python experiments/baselines/msani_train.py --out $d --predict_only \
          --mode wb97 --model $m --data $DATA --split $SPLIT >> $d/stdout.log 2>&1 ) &
done
wait
echo "end $(date)"
