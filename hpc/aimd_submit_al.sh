#!/bin/bash
# bwUniCluster: TD-wB97X-D/6-31G* adiabatic S1 reference trajectories for the active-learning criterion (validation:
# experiments/al/validation_trajectories.json) and additional test trajectories for the paper
# (experiments/al/test_trajectories_extra.json). 1000 steps of 0.5 fs (500 fs) from the GFN2-xTB minimum, 300 K MB
# velocities; restartable, so each trajectory gets two chained 24 h jobs (the second resumes from the checkpoint).
# Usage: bash hpc/aimd_submit_al.sh [validation|test|all]
set -e
cd /path/to/workspace/transferable_excited_MLIP
WHICH=${1:-all}
files=()
[ "$WHICH" != test ] && files+=(experiments/al/validation_trajectories.json)
[ "$WHICH" != validation ] && files+=(experiments/al/test_trajectories_extra.json)
for f in "${files[@]}"; do
    .venv/bin/python -c "import json,sys; [print(v['name'], v['smiles']) for v in json.load(open('$f'))]" |
    while read -r name smi; do
        A=$(sbatch --parsable --job-name=aimd_$name hpc/aimd.sh --name $name --smiles "$smi" --steps 1000 --max_hours 23.5)
        B=$(sbatch --parsable --dependency=afterany:$A --job-name=aimd_$name hpc/aimd.sh --name $name --smiles "$smi" --steps 1000 --max_hours 23.5)
        echo "$name $A $B"
    done
done
