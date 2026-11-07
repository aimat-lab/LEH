"""Tasks for an additional off-equilibrium geometry of molecules that already have one (new task id '<id>_g2', so
compute.py draws a different temperature and MD frame; 'mol' keeps the molecule grouping for the splits)."""
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
srcs, name = sys.argv[1:-1], sys.argv[-1]
tasks, seen = [], set()
for f in srcs:
    for t in pickle.load(open(ROOT / f, "rb")):
        if t["id"] in seen:
            continue
        seen.add(t["id"])
        tasks.append({**t, "id": t["id"] + "_g2", "mol": t["id"]})
pickle.dump(tasks, open(ROOT / "data/wb97/tasks" / f"{name}.pkl", "wb"))
print(name, len(tasks))
