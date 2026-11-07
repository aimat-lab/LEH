#!/bin/bash
# HoreKa2: analysis of finished run sets (no training): analyze.py + summarize_seeds.py per run directory given,
# then learning_curves.py. Usage: sbatch hpc/hk_analysis.sh experiments/wb97/runs_nk [more run dirs]
#SBATCH --job-name=analysis
#SBATCH --partition=dev-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000mb
#SBATCH --time=00:55:00
#SBATCH --output=logs/analysis_%j.out
source /path/to/workspace/transferable_excited_MLIP/hpc/hk_env.sh
cd $P
export PYTHONWARNINGS=ignore
for R in "$@"; do
    .venv/bin/python experiments/wb97/analyze.py $R > ${R%/*}/analyze_${R##*/}.log 2>&1 && \
    .venv/bin/python experiments/wb97/summarize_seeds.py $R && echo "analysed $R"
done
.venv/bin/python experiments/wb97/learning_curves.py
