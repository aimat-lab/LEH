#!/bin/bash
# HoreKa2: fragment (size-consistency) test of the baselines on the shared pair geometries (experiments/fragments/
# make_pairs.py -> data/frag_pairs.pkl). All models trained on the same 10k molecules as the head ablations:
# X-MACE 128 ch, MS-ANI, OMNI-P2x architecture from scratch, OMNI-P2x fine-tuned, plus pretrained OMNI-P2x.
# Outputs experiments/fragments/<model>/predictions.pkl (test_frag_mono, test_frag_dimer); read by fig_fragments.py.
#SBATCH --job-name=frag_pred
#SBATCH --partition=gpu-h100,gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120000mb
#SBATCH --time=03:00:00
#SBATCH --output=logs/frag_pred_%j.out
source /path/to/workspace/transferable_excited_MLIP/hpc/hk_env.sh
cd $P
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=8
D=data/frag_pairs.pkl; S=experiments/fragments/split_frag.json; L=experiments/wb97/runs_lc; F=experiments/fragments
echo "start $(date)"
mkdir -p $F/xmace128_sub10k
.venv/bin/python experiments/baselines/xmace_export.py --mode wb97 --data $D --split $S --out $F/xmace128_sub10k/export --n_states 4 && \
.venv_xmace/bin/python experiments/baselines/xmace_predict.py $L/xmace128_sub10k/xmace.model $F/xmace128_sub10k/export $F/xmace128_sub10k \
    --device cuda && echo "predicted xmace"
for m in msani vecmsani; do
    d=$F/sub10k_$m; mkdir -p $d; ln -sf $P/$L/sub10k_$m/model.pt $d/model.pt
    .venv_mlatom/bin/python experiments/baselines/msani_train.py --out $d --predict_only --mode wb97 --model $m \
        --data $D --split $S > $d/stdout.log 2>&1 && echo "predicted $m"
done
.venv_mlatom/bin/python experiments/baselines/omnip2x_finetune.py --data $D --split $S --out $L/sub10k_omnift --predict \
    --pred_out $F/sub10k_omnift > $F/omnift_predict.log 2>&1 && echo "predicted omnift"
.venv_mlatom/bin/python experiments/baselines/omnip2x_zeroshot.py --data $D --split $S --out $F/omni_zeroshot \
    --sets test_frag_mono test_frag_dimer > $F/omni_zeroshot.log 2>&1 && echo "predicted omni zero-shot"
echo "end $(date)"
