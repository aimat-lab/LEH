#!/bin/bash
# HoreKa2: run one command (analysis, no training) in the project directory on a dev-cpu node.
# Usage: sbatch hpc/hk_cmd_cpu.sh "command ..."
#SBATCH --job-name=cmd
#SBATCH --partition=dev-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000mb
#SBATCH --time=00:55:00
#SBATCH --output=logs/cmd_%j.out
source /path/to/workspace/transferable_excited_MLIP/hpc/hk_env.sh
cd $P
export PYTHONWARNINGS=ignore
echo "cmd: $1"; eval "$1"
