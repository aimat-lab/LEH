#!/bin/bash
# HoreKa2: 10-minute GPU test (queue wait, GPU type, and our model's update time if the env already exists).
#SBATCH --job-name=gpu_test
#SBATCH --partition=gpu-h100,gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000mb
#SBATCH --time=00:10:00
#SBATCH --output=logs/gpu_test_%j.out
source /path/to/workspace/transferable_excited_MLIP/hpc/hk_env.sh
cd $P
echo "host $(hostname) partition $SLURM_JOB_PARTITION start $(date)"
squeue -j $SLURM_JOB_ID -h -o "submitted %V started %S"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
if [ -x .venv/bin/python ]; then
    .venv/bin/python -c "import torch, e3nn, mace, rdkit, ase, h5py; print('main', torch.__version__, torch.version.cuda, 'cuda available', torch.cuda.is_available())"
    .venv_xmace/bin/python -c "import importlib, inspect; t = importlib.import_module('mace.tools.train'); import torch; print('xmace', torch.__version__, 'evaluate centres energies:', 'centred_energy' in inspect.getsource(t.evaluate))"
    .venv/bin/python -m pytest -q -p no:warnings tests/ 2>&1 | tail -1
    .venv/bin/python experiments/hk_gpu_bench.py 2>&1 | grep -v -i "warn\|visitor\|cuequi"
else
    echo "env not built yet: no model benchmark"
fi
echo "end $(date)"
