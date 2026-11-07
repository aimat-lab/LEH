#!/bin/bash
# HoreKa2: freeze the v1 benchmark state (experiments/freeze.py): manifest with sha256 + read-only copies in frozen/v1.
# Re-run after the remaining benchmark runs finish (X-MACE full set, OMNI-P2x fine-tuned full set, N x K): files
# already frozen are verified, new ones are added. Usage: sbatch hpc/hk_freeze.sh
#SBATCH --job-name=freeze_v1
#SBATCH --partition=dev-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000mb
#SBATCH --time=00:55:00
#SBATCH --output=logs/freeze_%j.out
source /path/to/workspace/transferable_excited_MLIP/hpc/hk_env.sh
cd $P
W=experiments/wb97
.venv/bin/python experiments/freeze.py --out frozen/v1 --copy --runs \
    "$W/runs_main/*" "$W/runs_abl/*" "$W/runs_hp/hp3_*" "$W/runs_hp3/*" "$W/runs_lc/*" "$W/runs_zeroshot/*" "$W/runs_nk/*" "$W/runs_xlab/*" \
    "experiments/conj/runs_B/*" "experiments/fragments/*" \
    --files data/wb97_all_100k_s1_conj.pkl data/frag_pairs.pkl experiments/conj/split_conj_only.json \
        experiments/conj/split_B.json experiments/conj/split_B_noS.json experiments/conj/split_A_conj.json \
        experiments/conj/conj_info.json experiments/fragments/split_frag.json experiments/wb97/subsets_100k_s1.json \
        experiments/wb97/scaffold_holdout.json experiments/wb97/nk_info.json experiments/dataset/motifs.json \
        experiments/frag_tddft/results.json experiments/frag_tddft/results_pairs.json \
        $W/learning_curves.json $W/size_parity.json $W/parity.npz $W/diag_force_scaling.json $W/results_runs_abl.json \
        $W/summary_runs_abl.json $W/results_runs_hp3.json $W/summary_runs_nk.json $W/results_runs_nk.json \
        $W/summary_runs_xlab.json $W/results_runs_xlab.json $W/fc_eval.json $W/fc_eval_local.json \
        data/wb97/fc_test_labels.pkl data/wb97/tasks/fc_test_index.json experiments/fragments/heads_cache.pkl \
        experiments/frag_tddft/results_pairs.json
du -sh frozen/v1
