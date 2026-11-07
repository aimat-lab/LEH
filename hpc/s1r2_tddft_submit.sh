#!/bin/bash
# bwUniCluster, after the HoreKa2 sampling job (hpc/hk_s1r2_sampling.sh) and copying data/wb97/tasks/s1r2.pkl (+
# _info.json) here: TD-wB97X-D single points of the round-2 frames (same protocol as all other tranches, 2 threads
# per calculation) plus a cleanup round for leftovers. Raw outputs in data/wb97/s1r2 (archived by compute.py).
set -e
cd /path/to/workspace/transferable_excited_MLIP
[ -f data/wb97/tasks/s1r2.SUBMITTED ] && { echo "already submitted"; exit 0; }
[ -f data/wb97/tasks/s1r2.pkl ] || { echo "data/wb97/tasks/s1r2.pkl missing (copy from HoreKa2)"; exit 1; }
A=$(sbatch --parsable --job-name=s1r2_tddft --time=08:00:00 --array=0-23%16 \
    hpc/dataset_shard.sh data/wb97/tasks/s1r2.pkl data/wb97/s1r2 24 32 2 6.5)
B=$(sbatch --parsable --dependency=afterany:$A --job-name=s1r2_clean --time=08:00:00 --array=0-5 \
    hpc/dataset_shard.sh data/wb97/tasks/s1r2.pkl data/wb97/s1r2 6 32 2 6.5)
echo "tddft $A cleanup $B" | tee data/wb97/tasks/s1r2.SUBMITTED
