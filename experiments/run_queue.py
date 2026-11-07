"""Run a list of training jobs (yaml: base config + list of runs with overrides) with P parallel workers.

Finished runs (summary.json present) are skipped, so the queue is restartable.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("spec")
ap.add_argument("--parallel", type=int, default=2)
ap.add_argument("--only", nargs="*")
ap.add_argument("--dry", action="store_true", help="only write job.yaml files and print their paths")
ap.add_argument("--set", nargs="*", default=[], help="key=value overrides applied to every run")
a = ap.parse_args()
spec = yaml.safe_load(open(a.spec))
base = spec["base"]
jobs = []
for run in spec["runs"]:
    name = run.pop("name")
    if a.only and name not in a.only:
        continue
    cfg = {**base, **run}
    cfg["head_kwargs"] = {**base.get("head_kwargs", {}), **run.get("head_kwargs", {})}
    for kv in a.set:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    cfg["out"] = str(Path(spec["outdir"]) / name)
    if (ROOT / cfg["out"] / "summary.json").exists():
        continue
    (ROOT / cfg["out"]).mkdir(parents=True, exist_ok=True)
    p = ROOT / cfg["out"] / "job.yaml"
    yaml.safe_dump(cfg, open(p, "w"))
    jobs.append((name, p))
if a.dry:
    for name, p in jobs:
        print(p.relative_to(ROOT))
    sys.exit(0)
running = []
while jobs or running:
    while jobs and len(running) < a.parallel:
        name, p = jobs.pop(0)
        log = open(p.parent / "stdout.log", "w")
        running.append((name, subprocess.Popen([sys.executable, "-m", "exmlip.training.train", str(p)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)))
        print(time.strftime("%H:%M"), "start", name, flush=True)
    for name, pr in list(running):
        if pr.poll() is not None:
            print(time.strftime("%H:%M"), "done", name, "rc", pr.returncode, flush=True)
            running.remove((name, pr))
    time.sleep(10)
