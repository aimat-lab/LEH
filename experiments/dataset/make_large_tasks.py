"""Size-extrapolation test molecules: neutral closed-shell CHNOF molecules with 11-20 heavy atoms from ZINC250k.

Two tasks per molecule: the GFN2-xTB minimum ('eq') and one MD-distorted geometry ('md'); both start from SMILES.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Descriptors import NumRadicalElectrons

ROOT = Path(__file__).resolve().parents[2]
df = pd.read_csv(ROOT / "data/zinc250k.csv")
rng = np.random.default_rng(7)
per_size = int(sys.argv[1]) if len(sys.argv) > 1 else 50
rows = []
for smi in df["smiles"].str.strip():
    m = Chem.MolFromSmiles(smi)
    if m is None:
        continue
    ha = m.GetNumHeavyAtoms()
    if not 11 <= ha <= 20:
        continue
    if any(a.GetSymbol() not in "CNOF" or a.GetFormalCharge() != 0 for a in m.GetAtoms()):
        continue
    if NumRadicalElectrons(m) != 0:
        continue
    rows.append((Chem.MolToSmiles(m), ha))
cand = pd.DataFrame(rows, columns=["smiles", "ha"]).drop_duplicates("smiles")
print("candidates per heavy-atom count", cand.groupby("ha").size().to_dict())
sel = pd.concat([g.sample(min(per_size, len(g)), random_state=int(rng.integers(1e9))) for _, g in cand.groupby("ha")])
tasks = []
for i, (smi, ha) in enumerate(zip(sel.smiles, sel.ha)):
    for mode in ["eq", "md"]:
        tasks.append(dict(id=f"Z{i:04d}_{mode}", smiles=smi, mode=mode, source="ZINC", n_heavy=int(ha),
                          gaps_eq=np.full(10, np.nan)))
pickle.dump(tasks, open(ROOT / "data/wb97/tasks/large.pkl", "wb"))
sel.to_csv(ROOT / "data/wb97/tasks/large_molecules.csv", index=False)
print("molecules", len(sel), "tasks", len(tasks))
