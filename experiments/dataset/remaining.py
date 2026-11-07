"""Remaining (neither finished nor failed) tasks per shard. Usage: remaining.py TASKS[,TASKS2,...] OUTDIR NSHARDS -> JSON.

Only tasks with a usable geometry for the given computation are counted; duplicate ids across task files once."""
import json
import pickle
import sys
import zlib
from pathlib import Path

tasks, out, nsh = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
done = {p.name[:-4] for p in out.glob("*.npz")} | {p.name[:-12] for p in out.glob("*.failed.json")} if out.exists() else set()
rem, seen = [0] * nsh, set()
for tf in tasks.split(","):
    for t in pickle.load(open(tf, "rb")):
        if t["id"] in seen:
            continue
        seen.add(t["id"])
        if t["id"] not in done:
            rem[zlib.crc32(t["id"].encode()) % nsh] += 1
print(json.dumps(dict(total=sum(rem), per_shard=rem, done=len(done))))
