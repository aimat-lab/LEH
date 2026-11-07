"""Pack the raw TURBOMOLE archives of the dataset entries into one tar.gz per group for Zenodo, without MO files.

Input: release/out_v4/aux/raw_map.tsv from release/export_dataset.py, one line per geometry:
  <archive path relative to the project data/ directory> <group> <folder name (dataset entry id)>
Each per-geometry archive (sp/, md/) becomes the folder <folder>/ in raw_turbomole_<group>.tar.gz. Members named mos
(MO coefficients, about two thirds of every archive) are skipped. One process per group. Writes per group
raw_turbomole_<group>.tar.gz and a summary (archives, missing, member counts, bytes, sha256) to repack_summary.json.
Run inside a job (bwUni: hpc/bwuni_repack_raw.sh), not on a login node:
  python release/repack_raw.py release/out_v4/aux/raw_map.tsv release/raw_zenodo --data data --workers 12
"""
import argparse
import hashlib
import io
import json
import tarfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

SKIP = {"mos", "alpha", "beta"}


def pack(job):
    group, rows, data, out = job
    dst = Path(out) / f"raw_turbomole_{group}.tar.gz"
    tmp = dst.with_suffix(".tmp")
    names, missing, n_in = Counter(), [], 0
    with tarfile.open(tmp, "w:gz", compresslevel=6) as tout:
        for path, folder in rows:
            src = Path(data) / path
            if not src.exists():
                missing.append(path); continue
            n_in += src.stat().st_size
            with tarfile.open(src, "r:gz") as tin:
                for mem in tin:
                    base = mem.name.rsplit("/", 1)[-1]
                    if not mem.isfile() or base in SKIP:
                        continue
                    data_bytes = tin.extractfile(mem).read()
                    info = tarfile.TarInfo(f"{folder}/{mem.name.lstrip('./')}")
                    info.size, info.mtime, info.mode = len(data_bytes), mem.mtime, 0o644
                    tout.addfile(info, io.BytesIO(data_bytes))
                    names[mem.name.lstrip("./")] += 1
    tmp.rename(dst)
    h = hashlib.sha256()
    with open(dst, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return group, dict(archives=len(rows) - len(missing), missing=len(missing), missing_examples=missing[:5],
                       members=dict(names.most_common()), bytes_in=n_in, bytes_out=dst.stat().st_size, sha256=h.hexdigest())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_map"); ap.add_argument("out")
    ap.add_argument("--data", default="data", help="directory the archive paths are relative to")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--groups", nargs="*")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    rows = defaultdict(list)
    for line in open(a.raw_map):
        path, group, folder = line.rstrip("\n").split("\t")
        rows[group].append((path, folder))
    groups = sorted(rows, key=lambda g: -len(rows[g]))  # largest first
    if a.groups:
        groups = [g for g in groups if g in a.groups]
    print({g: len(rows[g]) for g in groups}, flush=True)
    summary = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for g, s in ex.map(pack, [(g, rows[g], a.data, a.out) for g in groups]):
            summary[g] = s
            print(g, {k: v for k, v in s.items() if k != "members"}, flush=True)
    sp = Path(a.out) / "repack_summary.json"
    old = json.load(open(sp)) if sp.exists() else {}
    json.dump({**old, **summary}, open(sp, "w"), indent=1)


if __name__ == "__main__":
    main()
