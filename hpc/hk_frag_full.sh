#!/bin/bash
# HoreKa2: full-set models on the fragment pair geometries (data/frag_pairs.pkl): LEH and softmin (2 seeds each, our
# predictor), X-MACE and the OMNI-P2x architecture from scratch (full set). For the LEH-vs-softmin transfer figure
# (dimer spectra vs TD-wB97X-D on the same pairs). Outputs: <run>/predictions_frag.pkl, experiments/fragments/full_*.
#SBATCH --job-name=frag_full
#SBATCH --partition=gpu-h100,gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120000mb
#SBATCH --time=02:00:00
#SBATCH --output=logs/frag_full_%j.out
source /path/to/workspace/transferable_excited_MLIP/hpc/hk_env.sh
cd $P
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=8
D=data/frag_pairs.pkl; S=experiments/fragments/split_frag.json; M=experiments/wb97/runs_main; L=experiments/wb97/runs_lc; F=experiments/fragments
.venv/bin/python experiments/repredict.py $M/main_leh_all_s0 $M/main_leh_all_s1 $M/main_softmin_all_s0 $M/main_softmin_all_s1 \
    --data $D --split $S --keys test_frag --out predictions_frag.pkl && echo "exmlip done"
mkdir -p $F/full_xmace
.venv/bin/python experiments/baselines/xmace_export.py --mode wb97 --data $D --split $S --out $F/full_xmace/export --n_states 4 && \
.venv_xmace/bin/python experiments/baselines/xmace_predict.py $L/xmace128_suball/xmace.model $F/full_xmace/export $F/full_xmace --device cuda && echo "xmace done"
d=$F/full_vecmsani; mkdir -p $d; ln -sf $P/$L/suball_vecmsani/model.pt $d/model.pt
.venv_mlatom/bin/python experiments/baselines/msani_train.py --out $d --predict_only --mode wb97 --model vecmsani --data $D --split $S \
    > $d/stdout.log 2>&1 && echo "vecmsani done"
