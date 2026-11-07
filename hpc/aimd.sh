#!/bin/bash
# Reference TURBOMOLE S_n NVE trajectory (restartable). Usage: sbatch hpc/aimd.sh --name NAME --smiles SMI [--steps ..]
#SBATCH --job-name=aimd
#SBATCH --partition=cpu_il,cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32000mb
#SBATCH --time=24:00:00
#SBATCH --output=logs/aimd_%j.out
B=/path/to/workspace
W=$B/transferable_excited_MLIP
export TURBODIR=$B/TURBOMOLE_7.5.1
export XTBPATH=$B/software/xtb/share/xtb
export PATH=$B/software/xtb/bin:$TURBODIR/scripts:$TURBODIR/bin/em64t-unknown-linux-gnu_smp:$PATH
export MKL_NUM_THREADS=1
export TM_SCRATCH=${TMPDIR:-/tmp}/aimd_${SLURM_JOB_ID}
mkdir -p $TM_SCRATCH
cd $W
echo "host $(hostname) start $(date) args $@"
.venv/bin/python experiments/md/aimd_reference.py --threads 16 --max_hours 23.3 "$@"
echo "end $(date)"
