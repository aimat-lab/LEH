#!/bin/bash
# bwUniCluster: raw TURBOMOLE archives of the dataset entries -> one tar.gz per group for Zenodo, without MO files
# (release/repack_raw.py). Input release/out_v4/aux/raw_map.tsv (copied from the laptop), output release/raw_zenodo/.
# Usage: sbatch hpc/bwuni_repack_raw.sh
#SBATCH --job-name=repack_raw
#SBATCH --partition=cpu_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32000mb
#SBATCH --time=04:00:00
#SBATCH --output=logs/repack_raw_%j.out
W=/path/to/workspace/transferable_excited_MLIP
cd $W
echo "host $(hostname) start $(date)"
.venv/bin/python release/repack_raw.py release/out_v4/aux/raw_map.tsv release/raw_zenodo --data data --workers 12
echo "end $(date)"
