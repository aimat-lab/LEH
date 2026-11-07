#!/bin/bash
# After the sampling array: merge the selected frames into tasks and submit the TD-wB97X-D single points
# (same protocol as all other tranches, 2 threads per calculation) plus a cleanup round for leftovers.
set -e
cd /path/to/workspace/transferable_excited_MLIP
[ -f data/wb97/tasks/s1r1.SUBMITTED ] && { echo "already submitted"; exit 0; }
.venv/bin/python experiments/dataset/s1_sampling.py --merge --out data/wb97/tasks/s1r1.pkl
A=$(sbatch --parsable --job-name=s1r1_tddft --time=08:00:00 --array=0-23%16 \
    hpc/dataset_shard.sh data/wb97/tasks/s1r1.pkl data/wb97/s1r1 24 32 2 6.5)
B=$(sbatch --parsable --dependency=afterany:$A --job-name=s1r1_clean --time=08:00:00 --array=0-5 \
    hpc/dataset_shard.sh data/wb97/tasks/s1r1.pkl data/wb97/s1r1 6 32 2 6.5)
echo "tddft $A cleanup $B" | tee data/wb97/tasks/s1r1.SUBMITTED
