#!/bin/bash
# Local evaluation of one finished HoreKa2 run (HoreKa2 CPU queue too slow): copy the run, convergence criterion on the
# validation trajectories (tag = run name), eval_models frames + nve_ref on all reference trajectories.
# Usage: bash experiments/al/eval_local.sh RUN_NAME [THREADS]   (run in experiments/al/runs/RUN_NAME on HoreKa2)
set -u
cd "$(dirname "$0")/../.."
R=$1; T=${2:-8}
H=/path/to/workspace/transferable_excited_MLIP
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T
if [ "${NOCOPY:-0}" = 1 ]; then  # run already copied (NOCOPY=1 bash experiments/al/eval_local.sh RUN)
echo "$(date '+%F %T') $R local copy used"
else
echo "$(date '+%F %T') $R copy"
timeout 1800 rsync -a -e "ssh -o ControlPath=$HOME/.ssh/cm-%r@%h:%p -o BatchMode=yes" --include='best.pt' --include='meta.pt' \
  --include='config.yaml' --include='job.yaml' --include='summary.json' --include='train.log' --include='predictions.pkl' \
  --exclude='*' horeka2:$H/experiments/al/runs/$R/ experiments/al/runs/$R/ || { echo "copy failed"; exit 1; }
fi
[ -f experiments/al/runs/$R/summary.json ] || { echo "no summary.json"; exit 1; }
echo "$(date '+%F %T') $R convergence"
.venv/bin/python experiments/al/convergence.py --tag $R experiments/al/runs/$R --prev experiments/al/convergence_r3.json > experiments/al/convergence_$R.log 2>&1
echo "$(date '+%F %T') $R eval_models (exit $?)"
.venv/bin/python experiments/md/eval_models.py experiments/al/runs/$R --what frames nve_ref --threads $T > experiments/md/eval_$R.log 2>&1
echo "$(date '+%F %T') $R done (exit $?)"
