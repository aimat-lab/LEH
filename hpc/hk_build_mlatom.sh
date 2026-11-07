#!/bin/bash
# HoreKa2: build .venv_mlatom (MS-ANI / OMNI-P2x architecture baselines) with the bwUniCluster versions.
#SBATCH --job-name=env_mlatom
#SBATCH --partition=dev-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32000mb
#SBATCH --time=00:55:00
#SBATCH --output=logs/env_mlatom_%j.out
source /path/to/workspace/transferable_excited_MLIP/hpc/hk_env.sh
cd $P
set -e
uv python install 3.12
uv venv --python 3.12 .venv_mlatom
uv pip install --python .venv_mlatom/bin/python -r envs/req_mlatom.txt
.venv_mlatom/bin/python -c "import mlatom, torch, torchani; print('mlatom', mlatom.__version__ if hasattr(mlatom, '__version__') else '?', 'torch', torch.__version__, torch.version.cuda, 'torchani', torchani.__version__)"
du -sh .venv_mlatom
