#!/bin/bash
# HoreKa2: predictions of all full-set model-A runs and baselines on the S-free conjugated test sets (dyes, ladders),
# then per-size errors and parity data for the figures (experiments/collect_size_parity.py).
#SBATCH --job-name=conj_pred
#SBATCH --partition=gpu-h100,gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180000mb
#SBATCH --time=06:00:00
#SBATCH --output=logs/conj_pred_%j.out
source /path/to/workspace/transferable_excited_MLIP/hpc/hk_env.sh
cd $P
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=8
D=data/wb97_all_100k_s1_conj.pkl; S=experiments/conj/split_conj_only.json; M=experiments/wb97/runs_main; L=experiments/wb97/runs_lc
echo "start $(date)"
.venv/bin/python experiments/repredict.py $M/main_leh_all_s0 $M/main_leh_all_s1 $M/main_softmin_all_s0 $M/main_softmin_all_s1 \
    $M/main_leh_L2_all_s0 --data $D --split $S --keys test_dye test_lad --out predictions_conj.pkl
for m in msani vecmsani; do
    d=$L/suball_${m}_conj; mkdir -p $d; ln -sf ../suball_$m/model.pt $d/model.pt
    .venv_mlatom/bin/python experiments/baselines/msani_train.py --out $d --predict_only --mode wb97 --model $m \
        --data $D --split $S > $d/stdout.log 2>&1 && echo "predicted $d"
done
for X in $L/xmace128_sub30k $L/xmace128_suball; do      # 'suball' only once its 150 epochs are done
    [ -f $X/predictions.pkl ] || continue
    XC=${X}_conj; mkdir -p $XC
    .venv/bin/python experiments/baselines/xmace_export.py --mode wb97 --data $D --split $S --out $XC/export --n_states 4 && \
    .venv_xmace/bin/python experiments/baselines/xmace_predict.py $X/xmace.model $XC/export $XC --device cuda && echo "predicted $XC"
done
.venv/bin/python experiments/collect_size_parity.py
echo "end $(date)"
