#!/bin/bash
# Run several training jobs concurrently on one GPU.
# Usage: sbatch [--partition=...] hpc/train_gpu.sh path/to/job1.yaml path/to/job2.yaml ...
#SBATCH --job-name=exmlip_train
#SBATCH --partition=gpu_h100,gpu_a100_il,gpu_h100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120000mb
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_%j.out
W=/path/to/workspace/transferable_excited_MLIP
cd $W
echo "host $(hostname) gpu $CUDA_VISIBLE_DEVICES start $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv
for cfg in "$@"; do
    d=$(dirname $cfg)
    .venv/bin/python -m exmlip.training.train $cfg > $d/stdout.log 2>&1 &
    echo "started $cfg pid $!"
done
wait
echo "end $(date)"
