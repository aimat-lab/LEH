#!/bin/bash
# kcist version of the training job (RTX 4090 / RTX 6000 Ada): one run per GPU. Runs sharing a card are 4x slower each in
# float64, and kcist has no partition time limits, so no packing and no chaining: a 24 h limit with wall_hours 23.
# Usage: sbatch -p small hpc/kcist_train.sh path/to/job.yaml
#SBATCH -A <account>
#SBATCH --job-name=exmlip
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 24:00:00
#SBATCH -o logs/train_%j.out
cd ~/transferable_excited_MLIP
cfg=$1; d=$(dirname $cfg)
echo "host $(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader) start $(date) cfg $cfg"
.venv/bin/python -m exmlip.training.train $cfg --set wall_hours=23 threads=4 >> $d/stdout.log 2>&1
echo "exit $? end $(date)"
