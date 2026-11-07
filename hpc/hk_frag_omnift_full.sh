#!/bin/bash
# HoreKa2: OMNI-P2x fine-tuned with the authors' protocol on the full training set (runs_lc/suball_omnift, three
# ensemble members) on the fragment pair geometries (data/frag_pairs.pkl), for the separability and transfer figures
# (the 10k fine-tuned model was evaluated in hpc/hk_frag_predict.sh). Output: experiments/fragments/full_omnift/.
#SBATCH --job-name=frag_omnift
#SBATCH --partition=gpu-h100,gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120000mb
#SBATCH --time=01:00:00
#SBATCH --output=logs/frag_omnift_%j.out
source /path/to/workspace/transferable_excited_MLIP/hpc/hk_env.sh
cd $P
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=8
D=data/frag_pairs.pkl; S=experiments/fragments/split_frag.json; L=experiments/wb97/runs_lc; F=experiments/fragments
echo "start $(date)"
mkdir -p $F/full_omnift
.venv_mlatom/bin/python experiments/baselines/omnip2x_finetune.py --data $D --split $S --out $L/suball_omnift --predict \
    --pred_out $F/full_omnift > $F/full_omnift/predict.log 2>&1 && echo "predicted omnift (full)"
echo "end $(date)"
