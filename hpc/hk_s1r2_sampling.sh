#!/bin/bash
# HoreKa2: S1 sampling round 2 (user approved 2026-10-16; experiments/dataset/s1_sampling.py, see its docstring).
# Final full-set LEH seeds (A propagates, B = committee), motif molecules (experiments/dataset/motifs.json, all roles)
# + all ZINC molecules (test only), 1 ps NVE on S1 (80 %) / S2 (20 %), stop at model S1 gap < 0.2 eV, bond change or
# non-finite; picks: max double-bond twist + max committee disagreement (ZINC without motif: most S1-relaxed frame +
# committee). 4 shards in parallel on one GPU, restartable per chunk; then merge -> data/wb97/tasks/s1r2.pkl
# (+ _info.json), to be copied to bwUniCluster for the TDDFT single points.
#SBATCH --job-name=s1r2_md
#SBATCH --partition=gpu-h100,gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120000mb
#SBATCH --time=12:00:00
#SBATCH --output=logs/s1r2_md_%j.out
source /path/to/workspace/transferable_excited_MLIP/hpc/hk_env.sh
cd $P
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=4
M=experiments/wb97/runs_main; WD=data/wb97_aux/s1r2_sampling
ARGS="--data data/wb97_all_100k_s1.pkl --models $M/main_leh_all_s0 $M/main_leh_all_s1 --round s1r2 --zinc
      --motifs experiments/dataset/motifs.json --gap_stop 0.2 --picks twist,committee,relaxed --max_picks 2 --fs 1000 --chunk 128
      --threads 4 --device cuda --workdir $WD --nshards 4 --time_limit_h 11"
echo "host $(hostname) start $(date)"; nvidia-smi --query-gpu=name --format=csv,noheader
for s in 0 1 2 3; do
    .venv/bin/python experiments/dataset/s1_sampling.py $ARGS --shard $s > logs/s1r2_md_${SLURM_JOB_ID}_$s.log 2>&1 &
done
wait
.venv/bin/python experiments/dataset/s1_sampling.py --merge --round s1r2 --workdir $WD --out data/wb97/tasks/s1r2.pkl \
    --models $M/main_leh_all_s0 $M/main_leh_all_s1
echo "end $(date)"
