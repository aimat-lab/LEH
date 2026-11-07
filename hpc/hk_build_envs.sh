#!/bin/bash
# HoreKa2: build .venv (main) and .venv_xmace (patched X-MACE) with the exact versions of the bwUniCluster envs.
#SBATCH --job-name=env_build
#SBATCH --partition=dev-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32000mb
#SBATCH --time=00:55:00
#SBATCH --output=logs/env_build_%j.out
source /path/to/workspace/transferable_excited_MLIP/hpc/hk_env.sh
cd $P
echo "host $(hostname) start $(date)"
if ! curl -sI --max-time 20 https://pypi.org/simple/ | head -1 | grep -q 200; then echo "NO_INTERNET on compute node"; exit 2; fi
set -e
TORCH_IDX="--extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match"
uv python install 3.12 3.11
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r envs/req_main.txt $TORCH_IDX
uv venv --python 3.11 .venv_xmace
uv pip install --python .venv_xmace/bin/python -r envs/req_xmace.txt $TORCH_IDX
uv pip install --python .venv_xmace/bin/python --no-deps ./external/X-MACE
echo "== checks"
.venv/bin/python -c "import torch, e3nn, mace, rdkit, ase, h5py; print('main', torch.__version__, torch.version.cuda, e3nn.__version__)"
.venv_xmace/bin/python -c "import importlib, inspect; t = importlib.import_module('mace.tools.train'); import torch; print('xmace', torch.__version__, 'evaluate centres energies:', 'centred_energy' in inspect.getsource(t.evaluate))"
.venv/bin/python -m pytest -q -p no:warnings tests/ 2>&1 | tail -2
du -sh .venv .venv_xmace .uv 2>/dev/null
echo "end $(date)"
