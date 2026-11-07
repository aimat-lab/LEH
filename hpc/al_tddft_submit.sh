#!/bin/bash
# bwUniCluster: TD-wB97X-D single points of one active-learning round (after hpc/hk_al_sampling.sh and copying
# data/wb97/tasks/ROUND.pkl (+ _info.json) here); same protocol as all tranches. Usage: bash hpc/al_tddft_submit.sh ROUND
set -e
R=$1
cd /path/to/workspace/transferable_excited_MLIP
[ -f data/wb97/tasks/$R.SUBMITTED ] && { echo "already submitted"; exit 0; }
[ -f data/wb97/tasks/$R.pkl ] || { echo "data/wb97/tasks/$R.pkl missing (copy from HoreKa2)"; exit 1; }
A=$(sbatch --parsable --job-name=${R}_tddft --time=08:00:00 --array=0-23%16 \
    hpc/dataset_shard.sh data/wb97/tasks/$R.pkl data/wb97/$R 24 32 2 6.5)
B=$(sbatch --parsable --dependency=afterany:$A --job-name=${R}_clean --time=08:00:00 --array=0-5 \
    hpc/dataset_shard.sh data/wb97/tasks/$R.pkl data/wb97/$R 6 32 2 6.5)
echo "tddft $A cleanup $B" | tee data/wb97/tasks/$R.SUBMITTED
