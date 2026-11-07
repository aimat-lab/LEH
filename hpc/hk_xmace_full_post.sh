#!/bin/bash
# HoreKa2, after the full-set X-MACE run: predictions on the S-free conjugated sets (export + predict into
# runs_lc/xmace128_suball_conj), force-scaling diagnostic incl. the full-set models, learning curves, size/parity data.
#SBATCH --job-name=xmace_post
#SBATCH --partition=gpu-h100,gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120000mb
#SBATCH --time=03:00:00
#SBATCH --output=logs/xmace_post_%j.out
source /path/to/workspace/transferable_excited_MLIP/hpc/hk_env.sh
cd $P
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=8
L=experiments/wb97/runs_lc; M=experiments/wb97/runs_main; X=$L/xmace128_suball
.venv/bin/python experiments/baselines/xmace_export.py --mode wb97 --data data/wb97_all_100k_s1_conj.pkl \
    --split experiments/conj/split_conj_only.json --out ${X}_conj/export --n_states 4 && \
.venv_xmace/bin/python experiments/baselines/xmace_predict.py $X/xmace.model ${X}_conj/export ${X}_conj --device cuda && echo "conj done"
.venv/bin/python experiments/diag_force_scaling.py $X $L/xmace128_sub30k $M/main_leh_all_s0 $M/main_softmin_all_s0 \
    $L/suball_msani $L/suball_vecmsani $L/suball_omnift $L/suball_omniftA
.venv/bin/python experiments/wb97/learning_curves.py
.venv/bin/python experiments/collect_size_parity.py
