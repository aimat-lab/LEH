#!/bin/bash
# HoreKa2: re-predict all MS-ANI / OMNI-P2x-architecture (from scratch) runs with the state-ordering fix of
# msani_train.py (excited-state forces permuted with the sorted energies). Old predictions and the training metadata
# are moved to <run>/superseded_unsorted_forces/; config.yaml, summary.json and train.log (training metadata, which
# --predict_only would overwrite) are restored afterwards. Then the before/after check and the learning curves.
#SBATCH --job-name=msani_repred
#SBATCH --partition=gpu-h100,gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120000mb
#SBATCH --time=04:00:00
#SBATCH --output=logs/msani_repred_%j.out
source /path/to/workspace/transferable_excited_MLIP/hpc/hk_env.sh
cd $P
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=8
L=experiments/wb97/runs_lc
RUNS=$(ls -d $L/sub*_msani $L/sub*_vecmsani $L/suball_msani_conj $L/suball_vecmsani_conj experiments/fragments/sub10k_msani \
       experiments/fragments/sub10k_vecmsani 2>/dev/null)
echo "start $(date)"
for d in $RUNS; do
    [ -e $d/model.pt ] && [ -f $d/predictions.pkl ] && [ -f $d/config.yaml ] || { echo "skip $d"; continue; }
    read HEAD DATA SPLIT < <(.venv/bin/python -c "import yaml,sys; c=yaml.safe_load(open(sys.argv[1])); print(c['head'], c['data'], c['split'])" $d/config.yaml)
    B=$d/superseded_unsorted_forces
    if [ ! -f $B/predictions.pkl ]; then
        mkdir -p $B
        for f in predictions.pkl config.yaml summary.json train.log stdout.log; do [ -f $d/$f ] && cp -p $d/$f $B/; done
    fi
    .venv_mlatom/bin/python experiments/baselines/msani_train.py --out $d --predict_only --mode wb97 --model $HEAD \
        --data $DATA --split $SPLIT > $d/repredict.log 2>&1 && echo "re-predicted $d ($HEAD)" || echo "FAILED $d"
    for f in config.yaml summary.json train.log; do [ -f $B/$f ] && cp -p $B/$f $d/$f; done
done
.venv/bin/python experiments/baselines/msani_order_check.py $RUNS
.venv/bin/python experiments/wb97/learning_curves.py
echo "end $(date)"
