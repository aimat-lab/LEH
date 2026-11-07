#!/bin/bash
# HoreKa2, wave 2 (user approved 2026-10-14), on the final dataset data/wb97_all_100k_s1.pkl:
#  - S0 energy-weight check (spec_hp3, 3 runs, 1 GPU)
#  - ablations (spec_abl, 21 runs, 4 per GPU)
#  - baselines on the learning-curve subsets: X-MACE 128 ch (authors' settings, 150 epochs) and MS-ANI + OMNI-P2x
#    architecture (MLatom), sizes 1k/3k/10k/30k/all molecules
# Run on the login node (only writes job.yaml files and submits).
set -e
source /path/to/workspace/transferable_excited_MLIP/hpc/hk_env.sh
cd $P
.venv/bin/python experiments/run_queue.py experiments/wb97/spec_hp3.yaml --dry > /dev/null
.venv/bin/python experiments/run_queue.py experiments/wb97/spec_abl.yaml --dry > /dev/null
echo "hp3 $(sbatch --parsable hpc/hk_train_chain.sh experiments/wb97/runs_hp/hp3_c128_we0_{10,100,300}/job.yaml)"
mapfile -t ABL < <(ls -d experiments/wb97/runs_abl/abl_*/ | sort)
for ((i = 0; i < ${#ABL[@]}; i += 4)); do
    cfgs=(); for d in "${ABL[@]:i:4}"; do cfgs+=("${d}job.yaml"); done
    echo "abl $(sbatch --parsable hpc/hk_train_chain.sh "${cfgs[@]}") ${cfgs[*]}"
done
mkdir -p experiments/wb97/runs_lc
for s in 1k 3k 10k 30k all; do
    SPLIT=experiments/wb97/split_100k_s1_sub$s.json
    echo "xmace $s $(sbatch --parsable hpc/hk_xmace_chain.sh "--mode wb97 --data data/wb97_all_100k_s1.pkl --split $SPLIT --n_states 4 --run experiments/wb97/runs_lc/xmace128_sub$s --channels 128 --epochs 150")"
    echo "msani $s $(sbatch --parsable hpc/hk_msani.sh data/wb97_all_100k_s1.pkl $SPLIT experiments/wb97/runs_lc/sub$s)"
done
