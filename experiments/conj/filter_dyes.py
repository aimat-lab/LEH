"""Filter the Joung et al. (Sci. Data 2020) chromophore database for the conjugated large-molecule dataset (part A).

Keeps neutral, closed-shell, single-component molecules with 11-30 heavy atoms, elements within ALLOWED, and a
conjugated system of >= MIN_CONJ atoms. Reports counts with and without sulfur and what was excluded.
Usage: python -I filter_dyes.py CSV OUT_CSV [joung|beard]
"""
import collections
import sys

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Descriptors import NumRadicalElectrons

RDLogger.DisableLog("rdApp.*")
ALLOWED = {"H", "C", "N", "O", "F", "S"}
MIN_CONJ = 8


def largest_conjugated_system(m):
    parent = list(range(m.GetNumAtoms()))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    inconj = set()
    for b in m.GetBonds():
        if b.GetIsConjugated():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            inconj |= {i, j}
            parent[find(i)] = find(j)
    sizes = collections.Counter(find(i) for i in inconj)
    return max(sizes.values()) if sizes else 0


fmt = sys.argv[3] if len(sys.argv) > 3 else "joung"
if fmt == "beard":  # Beard et al. Sci. Data 2019, paper_allDB.csv: header has unquoted commas -> explicit names
    df = pd.read_csv(sys.argv[1], header=None, skiprows=1,
                     names=["Chromophore", "lam_sTDA", "f_sTDA", "lam_TDDFT", "f_TDDFT", "Absorption max (nm)", "eps", "Solvent"])
else:
    df = pd.read_csv(sys.argv[1])
reasons = collections.Counter()
other_el = collections.Counter()
rows = {}
for smi, lam, solv in zip(df["Chromophore"], df["Absorption max (nm)"], df["Solvent"]):
    m = Chem.MolFromSmiles(str(smi))
    if m is None:
        reasons["unparseable"] += 1; continue
    if len(Chem.GetMolFrags(m)) > 1:
        reasons["multi-component"] += 1; continue
    if Chem.GetFormalCharge(m) != 0:  # net charge; nitro groups etc. carry formal charges but are neutral
        reasons["charged"] += 1; continue
    if NumRadicalElectrons(m) != 0:
        reasons["radical"] += 1; continue
    els = {a.GetSymbol() for a in m.GetAtoms()}
    if not els <= ALLOWED:
        reasons["other elements"] += 1
        for e in els - ALLOWED:
            other_el[e] += 1
        continue
    ha = m.GetNumHeavyAtoms()
    if not 11 <= ha <= 30:
        reasons["heavy atoms outside 11-30"] += 1; continue
    nc = largest_conjugated_system(m)
    if nc < MIN_CONJ:
        reasons[f"conjugated system < {MIN_CONJ}"] += 1; continue
    can = Chem.MolToSmiles(m)
    r = rows.setdefault(can, dict(smiles=can, ha=ha, n_conj=nc, has_S="S" in els,
                                  n_atoms_H=Chem.AddHs(m).GetNumAtoms(), lam_abs=[], solvents=set()))
    if pd.notna(lam):
        r["lam_abs"].append(float(lam))
    r["solvents"].add(str(solv))
out = pd.DataFrame([dict(smiles=r["smiles"], ha=r["ha"], n_conj=r["n_conj"], has_S=r["has_S"], n_atoms=r["n_atoms_H"],
                         zwitterion=any(a.GetFormalCharge() != 0 for a in Chem.MolFromSmiles(r["smiles"]).GetAtoms()),
                         lam_abs_median=(pd.Series(r["lam_abs"]).median() if r["lam_abs"] else None),
                         n_solvents=len(r["solvents"])) for r in rows.values()])
out.to_csv(sys.argv[2], index=False)
print("records", len(df), "| unique kept", len(out))
print("excluded records by first failing criterion:", dict(reasons))
print("records excluded for other elements, by element:", dict(other_el.most_common(10)))
print("kept with S", int(out.has_S.sum()), "without S", int((~out.has_S).sum()))
print("kept by heavy atoms (no S / with S):")
for lo, hi in [(11, 15), (16, 20), (21, 25), (26, 30)]:
    s = out[(out.ha >= lo) & (out.ha <= hi)]
    print(f"  {lo}-{hi}: {int((~s.has_S).sum())} / {int(s.has_S.sum())}")
print("with experimental absorption maximum:", int(out.lam_abs_median.notna().sum()))
