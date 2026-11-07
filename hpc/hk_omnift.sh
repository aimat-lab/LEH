#!/bin/bash
# HoreKa2: fine-tune OMNI-P2x (3 ensemble members in parallel on one GPU) on one learning-curve subset, then predict.
# Usage: sbatch hpc/hk_omnift.sh SIZE [TAG "extra fine-tuning args"]
#   SIZE in 1k 3k 10k 30k all; output experiments/wb97/runs_lc/subSIZE_omnift<TAG>
#   adapted variant: sbatch hpc/hk_omnift.sh 30k A "--fixed_layers none --gap_coefficient 1.0"
#SBATCH --job-name=omnift
#SBATCH --partition=gpu-h100,gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180000mb
#SBATCH --time=48:00:00
#SBATCH --output=logs/omnift_%j.out
source /path/to/workspace/transferable_excited_MLIP/hpc/hk_env.sh
cd $P
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=4
S=$1; TAG=${2:-}; EXTRA=${3:-}; SPLIT=experiments/wb97/split_100k_s1_sub$S.json; OUT=experiments/wb97/runs_lc/sub${S}_omnift$TAG
mkdir -p $OUT
echo "host $(hostname) size $S tag '$TAG' args '$EXTRA' start $(date)"; nvidia-smi --query-gpu=name --format=csv,noheader
for k in 1 2 3; do
    [ -f $OUT/member$k.done ] && continue
    .venv_mlatom/bin/python experiments/baselines/omnip2x_finetune.py --split $SPLIT --out $OUT --member $k $EXTRA > $OUT/member$k.log 2>&1 &
done
wait
ls $OUT/member{1,2,3}.done && .venv_mlatom/bin/python experiments/baselines/omnip2x_finetune.py --split $SPLIT --out $OUT --predict $EXTRA > $OUT/predict.log 2>&1
echo "end $(date)"
