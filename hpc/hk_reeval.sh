#!/bin/bash
# HoreKa2: re-run the evaluations that read baseline predictions (after hpc/hk_msani_repredict.sh): per-size errors +
# parity data (collect_size_parity.py) and the force-scaling diagnostic (S0/S1 force slope per size bin).
#SBATCH --job-name=reeval
#SBATCH --partition=dev-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000mb
#SBATCH --time=00:55:00
#SBATCH --output=logs/reeval_%j.out
source /path/to/workspace/transferable_excited_MLIP/hpc/hk_env.sh
cd $P
M=experiments/wb97/runs_main; L=experiments/wb97/runs_lc
.venv/bin/python experiments/collect_size_parity.py
.venv/bin/python experiments/diag_force_scaling.py $L/xmace128_sub30k $M/main_leh_30k_s1 $M/main_softmin_30k_s0 \
    $L/sub30k_msani $L/sub30k_vecmsani
