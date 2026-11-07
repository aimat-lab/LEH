"""Freeze the benchmark state ("v1", user decision 2026-10-16) for reproducibility: all benchmark comparisons
(learning curves, ablations, baselines, N x K, fragments, size/parity, conjugated sets) use this dataset and these
models; new data (S1 sampling rounds >= 2) only enters the production models.

For every run directory given (globs relative to the project root): checkpoints and model files, configs, logs,
summaries and predictions; plus every data/split file referenced in the runs' config.yaml/job.yaml, the extra files
given with --files, and a code snapshot (exmlip, experiments *.py/*.yaml/*.json, hpc, envs, tests, figure scripts'
outputs in paper/figs, STATUS.md). Writes OUT/MANIFEST.json (relative path, bytes, sha256) and, with --copy,
copies everything into OUT/<relative path> (real copies: several files are rewritten in place by later scripts)
and removes write permission. Runs without predictions.pkl (still training) are only listed as unfinished. Re-running with more runs adds to the manifest; frozen files are re-hashed, and a file that
was regenerated since (e.g. learning_curves.json after more runs finished) is frozen again with the previous version
kept under OUT/_history/ and listed in the manifest's history.
Usage: python experiments/freeze.py --out frozen/v1 --runs 'experiments/wb97/runs_main/*' ... [--files ...] [--copy]
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import stat
import tarfile
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUN_FILES = ["best.pt", "meta.pt", "last.pt", "config.yaml", "job.yaml", "summary.json", "train.log", "*.model",
             "model.pt", "member*.pt", "member*.done", "epochs_done.txt", "predictions*.pkl", "export/index.json",
             "stdout.log", "*.log"]
CODE = ["exmlip", "experiments", "hpc", "envs", "tests", "paper/figs", "STATUS.md"]
CODE_EXT = {".py", ".yaml", ".yml", ".json", ".sh", ".txt", ".md", ".csv", ".png", ".pdf", ".toml"}

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="frozen/v1")
ap.add_argument("--runs", nargs="*", default=[])
ap.add_argument("--files", nargs="*", default=[])
ap.add_argument("--copy", action="store_true")
a = ap.parse_args()
OUT = ROOT / a.out
OUT.mkdir(parents=True, exist_ok=True)
man_f = OUT / "MANIFEST.json"
man = json.load(open(man_f)) if man_f.exists() else dict(created=time.strftime("%Y-%m-%d %H:%M"), files={}, runs={})


def sha(p, bs=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while (b := f.read(bs)):
            h.update(b)
    return h.hexdigest()


def add(rel):
    rel = str(rel)
    if Path(rel).is_absolute() or ".." in Path(rel).parts:  # OUT / rel would point at the original file
        raise SystemExit(f"{rel}: only paths inside the project (relative to {ROOT}) can be frozen")
    p = ROOT / rel
    if not p.is_file():
        return
    d = sha(p)
    old = man["files"].get(rel)
    q = OUT / rel
    if old and old["sha256"] == d and (q.exists() or not a.copy):
        return
    if old and old["sha256"] != d:  # regenerated (e.g. result files after more runs finished): keep the old copy
        print(f"changed since frozen: {rel} ({old['sha256'][:12]} -> {d[:12]}); previous version kept in _history/")
        h = OUT / "_history" / f"{rel}.{old['sha256'][:12]}"
        if q.exists():
            h.parent.mkdir(parents=True, exist_ok=True)
            os.replace(q, h)
        man.setdefault("history", {}).setdefault(rel, []).append(dict(old, superseded=time.strftime("%Y-%m-%d %H:%M"),
                                                                      path=str(h.relative_to(OUT))))
    man["files"][rel] = dict(bytes=p.stat().st_size, sha256=d, frozen=time.strftime("%Y-%m-%d %H:%M"))
    if a.copy and not q.exists():
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, q)
        os.chmod(q, stat.S_IRUSR | stat.S_IRGRP)


refs = set(a.files)
for pat in a.runs:
    for rd in sorted(glob.glob(str(ROOT / pat))):
        rd = Path(rd)
        if not rd.is_dir():
            continue
        rel = rd.relative_to(ROOT)
        done = (rd / "predictions.pkl").exists()
        if not done:  # still training (files keep changing): only recorded, frozen by a later pass
            man["runs"].setdefault(str(rel), dict(n_files=0, finished=False))
            continue
        files = sorted({f for g in RUN_FILES for f in glob.glob(str(rd / g))})
        for f in files:
            add(Path(f).relative_to(ROOT))
        for c in ("config.yaml", "job.yaml"):
            if (rd / c).exists():
                cfg = yaml.safe_load(open(rd / c)) or {}
                refs.update(str(cfg[k]) for k in ("data", "split") if cfg.get(k))
        man["runs"][str(rel)] = dict(n_files=len(files), finished=True)
for r in sorted(refs):
    r = Path(r)
    add(r.relative_to(ROOT) if r.is_absolute() and ROOT in r.parents else r)

# code snapshot (small text/figure files only; large data and run outputs are covered above)
tar = OUT / f"code_{time.strftime('%Y%m%d_%H%M%S')}.tar.gz"
with tarfile.open(tar, "w:gz") as t:
    for c in CODE:
        for p in sorted((ROOT / c).rglob("*") if (ROOT / c).is_dir() else [ROOT / c]):
            if p.is_file() and p.suffix in CODE_EXT and "__pycache__" not in p.parts and "runs" not in p.parts[-3:-1] \
                    and p.stat().st_size < 20e6:
                t.add(p, arcname=str(p.relative_to(ROOT)))
man.setdefault("code_snapshots", []).append(dict(file=tar.name, sha256=sha(tar)))
json.dump(man, open(man_f, "w"), indent=1, sort_keys=True)
n = len(man["files"]); gb = sum(v["bytes"] for v in man["files"].values()) / 1e9
print(f"manifest: {n} files, {gb:.1f} GB, {len(man['runs'])} runs ({sum(not v['finished'] for v in man['runs'].values())} unfinished)")
