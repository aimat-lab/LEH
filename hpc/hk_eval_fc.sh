#!/bin/bash
# HoreKa2: Franck-Condon force test of the frozen benchmark models (experiments/eval_fc.py; reuses stored predictions).
#SBATCH --job-name=eval_fc
#SBATCH --partition=dev-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64000mb
#SBATCH --time=00:55:00
#SBATCH --output=logs/eval_fc_%j.out
source /path/to/workspace/transferable_excited_MLIP/hpc/hk_env.sh
cd $P
M=experiments/wb97/runs_main; L=experiments/wb97/runs_lc
.venv/bin/python experiments/eval_fc.py $M/main_leh_all_s0 $M/main_leh_all_s1 $M/main_leh_L2_all_s0 $M/main_softmin_all_s0 \
    $M/main_softmin_all_s1 $L/xmace128_suball $L/suball_msani $L/suball_vecmsani $L/suball_omnift $L/suball_omniftA \
    $M/main_leh_30k_s1 $L/xmace128_sub30k experiments/wb97/runs_xlab/xlab_leh_30k_s0 experiments/wb97/runs_zeroshot/omnip2x
