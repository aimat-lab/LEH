#!/bin/bash
# MS-ANI (MLatom) baselines + matched-data exmlip runs on one GPU.
#SBATCH --job-name=baselines
#SBATCH --partition=gpu_h100,gpu_a100_il,gpu_h100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120000mb
#SBATCH --time=08:00:00
#SBATCH --output=logs/baselines_%j.out
W=/path/to/workspace/transferable_excited_MLIP
cd $W
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=4
echo "host $(hostname) start $(date)"; nvidia-smi --query-gpu=name --format=csv,noheader
ms() {  # name, extra args
    d=$1; shift
    mkdir -p $d
    ( timeout 6.8h .venv_mlatom/bin/python experiments/baselines/msani_train.py --out $d --max_epochs 3000 "$@" > $d/stdout.log 2>&1;
      [ -f $d/predictions.pkl ] || .venv_mlatom/bin/python experiments/baselines/msani_train.py --out $d --predict_only "$@" >> $d/stdout.log 2>&1 ) &
}
ms experiments/wb97/runs/msani --mode wb97 --model msani --data data/wb97_all.pkl --split experiments/wb97/split.json
ms experiments/wb97/runs/vecmsani --mode wb97 --model vecmsani --data data/wb97_all.pkl --split experiments/wb97/split.json
ms experiments/qcdge/runs/msani --mode spectra --model msani --n_states 6 --data data/qcdge_sub.pkl --split experiments/qcdge/split_size.json
for cfg in "$@"; do
    .venv/bin/python -m exmlip.training.train $cfg > $(dirname $cfg)/stdout.log 2>&1 &
done
wait
echo "end $(date)"
