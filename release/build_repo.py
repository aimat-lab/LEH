"""Assemble the public code repository (github.com/aimat-lab/LEH) from the project folder into release/LEH/.

Copies the package, tests, environment files, experiment/analysis/figure scripts, cluster scripts, the run
configurations and summaries of all models reported in the paper (checkpoints go to Zenodo), and the release scripts;
replaces cluster paths and account names by placeholders. Leaves out data, checkpoints, predictions, logs, internal
notes, pilot runs that are not in the paper, and files above MAX_BYTES. Files written by hand in release/LEH (README.md,
LICENSE, pyproject.toml, .gitignore, exmlip/data/load_release.py, hpc/README.md) are not touched.
Usage: python release/build_repo.py [--dry]
"""
import argparse
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "release/LEH"
MAX_BYTES = 5_000_000
DIRS = ["exmlip", "experiments", "hpc", "tests", "envs"]
EXT = {".py", ".sh", ".yaml", ".yml", ".md", ".txt", ".json", ".patch"}
EXTRA = ["release/export_dataset.py", "release/export_dataset_v1.py", "release/repack_raw.py", "release/zenodo_upload.py",
         "release/README_dataset.md", "release/build_repo.py", "external/xmace_exmlip_fix.patch"]
SKIP_PREFIX = ["experiments/smoke", "experiments/wb97/runs/", "experiments/tm/", "experiments/tm_data",
               "experiments/qcdge/runs/", "experiments/al/finish_", "experiments/eigvec/tddft/raw_index.txt"]
RUN_FILES = {"config.yaml", "job.yaml", "summary.json"}  # inside run directories
SUBS = [(r"/pfs/work9/workspace/scratch/ka_bp4935-aimat04/transferable_excited_MLIP/", "", (".yaml", ".json")),
        (r"/hfs2/work/workspace/scratch/ka_bp4935-aimat1/transferable_excited_MLIP/", "", (".yaml", ".json")),
        (r"/pfs/work9/workspace/scratch/ka_bp4935-aimat04", "/path/to/workspace", None),
        (r"/hfs2/work/workspace/scratch/ka_bp4935-aimat1", "/path/to/workspace", None),
        (r"cd /home/pascal/todel/transferable_excited_MLIP", 'cd "$(dirname "$0")/../.."', None),
        (r"-u ka_bp4935", "-u $USER", None),
        (r"#SBATCH -A aimat\b", "#SBATCH -A <account>", None)]
LEFTOVER = re.compile(r"ka_bp4935|/home/pascal|aimat0\d|aimat1\b")


def selected():
    files = [p for d in DIRS for p in (ROOT / d).rglob("*") if p.is_file()] + [ROOT / e for e in EXTRA]
    for p in sorted(files):
        rel = p.relative_to(ROOT).as_posix()
        if "__pycache__" in rel or p.suffix not in EXT or any(rel.startswith(s) for s in SKIP_PREFIX):
            continue
        if "/runs" in rel and (p.name not in RUN_FILES or "/failed_" in rel):  # failed starts are not models
            continue
        if p.stat().st_size > MAX_BYTES:
            print("too large, skipped:", rel, p.stat().st_size); continue
        yield p, rel


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true"); a = ap.parse_args()
    n, size, left = 0, 0, []
    for p, rel in selected():
        txt = p.read_text(errors="strict")
        for pat, rep, only in SUBS:
            if only is None or p.suffix in only:
                txt = re.sub(pat, lambda m: rep, txt)
        if LEFTOVER.search(txt):
            left.append(rel)
        n += 1; size += len(txt.encode())
        if not a.dry:
            dst = OUT / rel; dst.parent.mkdir(parents=True, exist_ok=True); dst.write_text(txt)
            shutil.copymode(p, dst)
    print(f"{n} files, {size / 1e6:.1f} MB")
    if left:
        print("check remaining cluster-specific strings in:", *left, sep="\n  ")


if __name__ == "__main__":
    main()
