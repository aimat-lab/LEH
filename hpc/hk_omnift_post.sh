#!/bin/bash
# HoreKa2, after the full-set OMNI-P2x fine-tuning (hk_omnift.sh all [TAG]): predictions on the S-free conjugated sets
# (members from runs_lc/suball_omnift<TAG>, output runs_lc/suball_omnift<TAG>_conj), learning curves, size/parity data.
# Usage: sbatch hpc/hk_omnift_post.sh [TAG]
#SBATCH --job-name=omnift_post
#SBATCH --partition=gpu-h100,gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120000mb
#SBATCH --time=03:00:00
#SBATCH --output=logs/omnift_post_%j.out
source /path/to/workspace/transferable_excited_MLIP/hpc/hk_env.sh
cd $P
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=8
L=experiments/wb97/runs_lc; T=${1:-}
[ -f $L/suball_omnift$T/predictions.pkl ] || { echo "full-set fine-tuning has no predictions"; exit 1; }
.venv_mlatom/bin/python experiments/baselines/omnip2x_finetune.py --data data/wb97_all_100k_s1_conj.pkl \
    --split experiments/conj/split_conj_only.json --out $L/suball_omnift$T --predict --pred_out $L/suball_omnift${T}_conj \
    > $L/suball_omnift${T}_conj.log 2>&1 && echo "conj predictions done"
.venv/bin/python experiments/wb97/learning_curves.py
.venv/bin/python experiments/collect_size_parity.py
