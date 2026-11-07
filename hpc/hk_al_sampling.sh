#!/bin/bash
# HoreKa2: S1 sampling of one active-learning round (experiments/dataset/s1_sampling.py; procedure in
# experiments/al/README.md). The two sampling models (A propagates, B = committee) are the warm-started models of the
# round. Molecules: motif molecules (experiments/dataset/motifs.json) + a round-specific hash slice of all molecules
# (FRAC_EXTRA) + all ZINC molecules (test only). 1 ps NVE on S1 (80 %) / S2 (20 %); picks: max double-bond twist +
# committee disagreement (no motif: committee + most S1-relaxed), <= 2 per trajectory. Then merge -> tasks file.
# Usage: sbatch hpc/hk_al_sampling.sh ROUND MODEL_A MODEL_B [FRAC_EXTRA]   e.g. s1r3 runs/... runs/... 0.05
#SBATCH --job-name=al_md
#SBATCH --partition=gpu-h100,gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120000mb
#SBATCH --time=12:00:00
#SBATCH --output=logs/al_md_%j.out
source /path/to/workspace/transferable_excited_MLIP/hpc/hk_env.sh
cd $P
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=4
R=$1; A=$2; B=$3; FX=${4:-0.05}; WD=data/wb97_aux/${R}_sampling
ARGS="--data data/wb97_all_100k_s1.pkl --models $A $B --round $R --zinc --motifs experiments/dataset/motifs.json
      --frac_extra $FX --gap_stop 0.2 --picks twist,committee,relaxed --max_picks 2 --fs 1000 --chunk 128
      --threads 4 --device cuda --workdir $WD --nshards 4 --time_limit_h 11"
echo "host $(hostname) round $R start $(date)"; nvidia-smi --query-gpu=name --format=csv,noheader
for s in 0 1 2 3; do
    .venv/bin/python experiments/dataset/s1_sampling.py $ARGS --shard $s > logs/al_md_${SLURM_JOB_ID}_$s.log 2>&1 &
done
wait
.venv/bin/python experiments/dataset/s1_sampling.py --merge --round $R --workdir $WD --out data/wb97/tasks/$R.pkl --models $A $B
echo "end $(date)"
