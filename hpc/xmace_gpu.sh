#!/bin/bash
# X-MACE baseline runs on one GPU. Usage: sbatch hpc/xmace_gpu.sh "<xmace_run.py args>" ["<args run 2>" ...]
#SBATCH --job-name=xmace
#SBATCH --partition=gpu_h100,gpu_a100_il,gpu_h100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96000mb
#SBATCH --time=08:00:00
#SBATCH --output=logs/xmace_%j.out
cd /path/to/workspace/transferable_excited_MLIP
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=4
echo "host $(hostname) start $(date)"; nvidia-smi --query-gpu=name --format=csv,noheader
for args in "$@"; do
    .venv_xmace/bin/python experiments/baselines/xmace_run.py $args --hours 7.0 &
done
wait
echo "end $(date)"
