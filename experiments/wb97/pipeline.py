"""Cluster-side pipeline for the new dataset, run as a SLURM job (hpc/wb97_pipeline.sh) after the data shards.

If a tranche still has unfinished tasks and round < max_rounds: resubmit only the incomplete shards plus this pipeline
(afterany dependency) and stop. Otherwise: build the extra-QCDGE-spectra tasks (once), run prepare.py, write the
training job files and submit them in packs of four runs per GPU.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = str(ROOT / ".venv/bin/python")
QCDGE = "/path/to/workspace/qcdge"
# (job name, tasks, output dir, shards, batch script, time limit for submitting new work in h)
TRANCHES = [("wb97_data", "data/wb97/tasks/t0.pkl", "data/wb97/t0", 12, "hpc/dataset_shard.sh", 1.6),
            ("wb97_large", "data/wb97/tasks/large.pkl", "data/wb97/large", 12, "hpc/dataset_shard.sh", 1.6),
            ("wb97_e0", "data/wb97/tasks/pilot.pkl,data/wb97/tasks/t0.pkl,data/wb97/tasks/eqextra.pkl", "data/wb97_eq",
             12, "hpc/e0_shard.sh", 0.75)]

ap = argparse.ArgumentParser()
ap.add_argument("--round", type=int, default=1)
ap.add_argument("--max_rounds", type=int, default=4)
ap.add_argument("--n_extra", type=int, default=20000)
ap.add_argument("--no_train", action="store_true")
ap.add_argument("--stage", default="iter", choices=["iter", "final", "final100"],
                help="iter: 5k dataset + automatic training; final: t1 + second geometries, build data/wb97_all_25k.pkl only")
a = ap.parse_args()


def sh(cmd):
    print("+", cmd, flush=True)
    return subprocess.run(cmd, shell=True, cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


FINAL = [("wb97_t1", "data/wb97/tasks/t1.pkl", "data/wb97/t1", 30, "hpc/dataset_shard.sh", 3.4),
         ("wb97_t0g2", "data/wb97/tasks/t0g2.pkl", "data/wb97/t0g2", 8, "hpc/dataset_shard.sh", 3.4),
         ("wb97_e0", "data/wb97/tasks/pilot.pkl,data/wb97/tasks/t0.pkl,data/wb97/tasks/t1.pkl", "data/wb97_eq",
          12, "hpc/e0_shard.sh", 0.75)]
FINAL100 = FINAL[:2] + [
    ("wb97_t2", "data/wb97/tasks/t2.pkl", "data/wb97/t2", 96, "hpc/dataset_shard.sh", 3.4),
    ("wb97_e0", "data/wb97/tasks/pilot.pkl,data/wb97/tasks/t0.pkl,data/wb97/tasks/t1.pkl,data/wb97/tasks/t2.pkl",
     "data/wb97_eq", 24, "hpc/e0_shard.sh", 0.75)]
if a.stage == "final":
    TRANCHES = FINAL
elif a.stage == "final100":
    TRANCHES = FINAL100
ids = []
for name, tasks, out, nsh, script, tl in TRANCHES:
    r = json.loads(sh(f"{PY} experiments/dataset/remaining.py {tasks} {out} {nsh}"))
    print(name, r, flush=True)
    shards = [str(i) for i, x in enumerate(r["per_shard"]) if x > 0]
    if shards and a.round < a.max_rounds:
        tlim = "--time=04:00:00 " if tl > 2 else ""
        ids.append(sh(f"sbatch --parsable --job-name={name} --partition=cpu_il,cpu {tlim}--array={','.join(shards)} "
                      f"{script} {tasks} {out} {nsh} 16 4 {tl}").split(";")[0])
if ids:
    nxt = sh(f"sbatch --parsable --dependency=afterany:{':'.join(ids)} hpc/wb97_pipeline.sh --round {a.round + 1} "
             f"--max_rounds {a.max_rounds} --n_extra {a.n_extra} --stage {a.stage}" + (" --no_train" if a.no_train else ""))
    print("resubmitted shards", ids, "next pipeline", nxt)
    sys.exit(0)
if a.stage == "final":  # final datasets: all tranches, separate files, training is submitted by hand
    print(sh(f"{PY} experiments/wb97/prepare.py pilot t0 large t1 t0g2 --eq_dir data/wb97_eq --tag _25k --drop_reactive"))
    sys.exit(0)
if a.stage == "final100":
    print(sh(f"{PY} experiments/wb97/prepare.py pilot t0 large t1 t0g2 t2 --eq_dir data/wb97_eq --tag _100k "
             f"--drop_reactive"))
    # user decision 2026-10-10: once the 100k dataset is built, start the conjugated large-molecule TDDFT
    # (dyes with an MD geometry for every dye = md100); threads from experiments/conj/threads.txt
    marker = ROOT / "data/conj/SUBMITTED"
    if not marker.exists():
        nt = (ROOT / "experiments/conj/threads.txt").read_text().strip() or "4"
        # user 2026-10-11: start the conjugated set only once our GPU training has started (CPU usage lowers the
        # fairshare and thereby the GPU priority); SLURM 'after:' = may begin after the listed jobs have begun
        dep = (ROOT / "experiments/conj/start_after_jobs.txt")
        env = f"CONJ_DEP=after:{dep.read_text().strip()} " if dep.exists() and dep.read_text().strip() else ""
        print(sh(f"{env}bash hpc/conj_submit.sh md100 {nt}"))
        marker.write_text("submitted by pipeline --stage final100\n")
    sys.exit(0)
extra = ROOT / "data/wb97/tasks/eqextra.pkl"
if not extra.exists():
    print(sh(f"{PY} experiments/dataset/make_tasks.py --qcdge {QCDGE} --out data/wb97 --start 5048 --n {a.n_extra} "
             f"--name eqextra"))
print(sh(f"{PY} experiments/wb97/prepare.py pilot t0 large --eq_extra {extra} --eq_dir data/wb97_eq"))
if a.no_train:
    sys.exit(0)
jobs = sh(f"{PY} experiments/run_queue.py experiments/wb97/spec.yaml --dry").split()
for k in range(0, len(jobs), 4):
    print(sh(f"sbatch --parsable --job-name=wb97_train hpc/train_gpu.sh {' '.join(jobs[k:k + 4])}"))
if not (ROOT / "experiments/wb97/runs/msani/predictions.pkl").exists():
    print(sh("sbatch --parsable hpc/baselines_gpu.sh"))
if not (ROOT / "experiments/wb97/runs/xmace_128/predictions.pkl").exists():  # X-MACE: S0-S3, authors' and our size
    xa = "--mode wb97 --data data/wb97_all.pkl --split experiments/wb97/split.json --n_states 4"
    print(sh(f'sbatch --parsable hpc/xmace_gpu.sh "{xa} --run experiments/wb97/runs/xmace_128 --channels 128" '
             f'"{xa} --run experiments/wb97/runs/xmace_32 --channels 32"'))
