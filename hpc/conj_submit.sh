#!/bin/bash
# Conjugated large-molecule dataset (A: dyes, B: ladders). NOT submitted automatically.
# Usage (login node, submission only), once the 100k dataset is done and the user decided:
#   bash hpc/conj_submit.sh [md40|md100] [THREADS]
#     md40  : MD geometry for a stratified 40 % of the dyes (tasks/dyes_md.pkl,     ~5.3k core-h)
#     md100 : MD geometry for every dye                    (tasks/dyes_md_all.pkl, ~13.1k core-h)
#     THREADS: TURBOMOLE threads per calculation (default 4; set from experiments/conj/thread_bench.json)
# Fixed parts: dyes_eq ~13.1k, ladders_eq ~3.0k, ladders_md ~3.0k core-h (estimates incl. 25 % overhead).
# 64-core shards, 8 h jobs, 6.5 h cutoff for starting new molecules, throttled to 16 concurrent shards per array.
set -e
MD=${1:-md100}; NT=${2:-4}; NW=$((64 / NT))
cd /path/to/workspace/transferable_excited_MLIP
case $MD in
    md40) MDTASKS=data/conj/tasks/dyes_md.pkl; MDSH=12 ;;
    md100) MDTASKS=data/conj/tasks/dyes_md_all.pkl; MDSH=28 ;;
    *) echo "first argument must be md40 or md100"; exit 1 ;;
esac
# optional: CONJ_DEP="after:<jobids>" delays the start (e.g. until our GPU training jobs have started)
DEPARG=${CONJ_DEP:+--dependency=$CONJ_DEP}
sub() {  # name tasks out nshards
    sbatch --parsable $DEPARG --job-name=$1 --time=08:00:00 --array=0-$(($4 - 1))%16 \
        hpc/dataset_shard.sh $2 $3 $4 $NW $NT 6.5
}
A=$(sub conj_dyes_eq data/conj/tasks/dyes_eq.pkl data/conj/dyes_eq 28)
B=$(sub conj_dyes_md $MDTASKS data/conj/dyes_md $MDSH)
C=$(sub conj_lad_eq data/conj/tasks/ladders_eq.pkl data/conj/ladders_eq 7)
D=$(sub conj_lad_md data/conj/tasks/ladders_md.pkl data/conj/ladders_md 7)
echo "md=$MD threads=$NT workers=$NW | dyes_eq $A dyes_md $B ladders_eq $C ladders_md $D"
