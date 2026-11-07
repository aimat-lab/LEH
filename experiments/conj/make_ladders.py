"""Conjugation ladders (part B): homologous series of increasing conjugation length up to MAX_HA heavy atoms,
with donor/acceptor end groups. Output experiments/conj/ladders.csv (smiles, family, n, ends, ha, has_S).

Hold-out convention (decided up front): within each (family, ends) series the two longest members are test-only.
"""
import itertools
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[2]
MAX_HA = 30
# end groups as SMILES substituents ('' = H); nitro written neutral-overall
ENDS = {"H": "[H]", "Me": "C", "NH2": "N", "NMe2": "N(C)C", "OMe": "OC", "F": "F", "CN": "C#N", "CHO": "C=O",
        "NO2": "[N+](=O)[O-]", "COOMe": "C(=O)OC"}  # SMILES of the group with the attachment atom first
DONORS, ACCEPTORS = ["H", "Me", "NH2", "NMe2", "OMe"], ["H", "F", "CN", "CHO", "NO2", "COOMe"]
END_PAIRS = sorted({tuple(sorted(p)) for p in itertools.product(DONORS, ACCEPTORS)} | {("H", "H"), ("NMe2", "NMe2"),
                                                                                     ("CHO", "CHO"), ("CN", "CN")})


def attach(core, a, b):
    """core has two '[*]' attachment points; groups are joined with RDKit molzip (no string surgery)."""
    c = core.replace("[*]", "[*:1]", 1).replace("[*]", "[*:2]", 1)
    m = Chem.MolFromSmiles(c)
    ga = Chem.MolFromSmiles("[*:1]" + ENDS[a])
    gb = Chem.MolFromSmiles("[*:2]" + ENDS[b])
    if m is None or ga is None or gb is None:
        return None
    z = Chem.molzip(Chem.molzip(m, ga), gb)
    z = Chem.RemoveHs(z)
    Chem.SanitizeMol(z)
    return Chem.MolToSmiles(z)


SERIES = {  # family: (core builder n -> SMILES with two [*], n range)
    "polyene": (lambda n: "[*]" + "/C=C" * n + "/[*]", range(1, 15)),  # all-trans
    "diphenylpolyene": (lambda n: "[*]c1ccc(cc1)" + "/C=C" * n + "/c1ccc(cc1)[*]", range(1, 9)),
    "oligophenylene": (lambda n: "[*]" + "c1ccc(cc1)" * (n - 1) + "c1ccc(cc1)[*]", range(1, 6)),
    "phenylenevinylene": (lambda n: "[*]c1ccc(cc1)" + "/C=C/c1ccc(cc1)" * n + "[*]", range(1, 4)),
    "oligofuran": (lambda n: "[*]" + "c1ccc(o1)" * (n - 1) + "c1ccc(o1)[*]", range(1, 7)),
    "oligopyrrole": (lambda n: "[*]" + "c1ccc([nH]1)" * (n - 1) + "c1ccc([nH]1)[*]", range(1, 7)),
    "oligothiophene": (lambda n: "[*]" + "c1ccc(s1)" * (n - 1) + "c1ccc(s1)[*]", range(1, 7)),
    "azobenzene": (lambda n: "[*]c1ccc(cc1)" + "/N=N/c1ccc(cc1)" * n + "[*]", range(1, 3)),  # trans
}
FIXED = {  # small hand-picked fused/polycyclic and classic dyes (family, smiles)
    "acene": ["c1ccc2ccccc2c1", "c1ccc2cc3ccccc3cc2c1", "c1ccc2cc3cc4ccccc4cc3cc2c1"],
    "phenacene": ["c1ccc2c(c1)ccc1ccccc12", "c1ccc2c(c1)ccc1c2ccc2ccccc21", "c1ccc2c(c1)ccc1c2ccc2c1ccc1ccccc12"],
    "aza_aromatic": ["c1ccc2ncccc2c1", "c1ccc2nc3ccccc3cc2c1", "c1ccc2nc3ccccc3nc2c1", "c1cnc2c(c1)ccc1cccnc12",
                     "c1ccc2c(c1)[nH]c1ccccc12", "c1ccc2c(c1)oc1ccccc12", "c1ccc2c(c1)sc1ccccc12"],
    "quinone": ["O=C1C=CC(=O)C=C1", "O=C1C=CC(=O)c2ccccc21", "O=C1c2ccccc2C(=O)c2ccccc21"],
    "classic_dye": ["O=C1/C(=C2/Nc3ccccc3C2=O)Nc2ccccc21", "O=C1/C(=C2/Sc3ccccc3C2=O)Sc2ccccc21",
                    "c1cc2cc3ccc(cc4ccc(cc5ccc(cc1n2)[nH]5)n4)[nH]3", "O=c1ccc2ccccc2o1", "CN(C)c1ccc2ccc(=O)oc2c1"],
}
rows = []
for fam, (core, ns) in SERIES.items():
    for a, b in END_PAIRS:
        for n in ns:
            smi = attach(core(n), a, b)
            m = Chem.MolFromSmiles(smi) if smi else None
            if m is None:
                print("failed", fam, n, a, b); continue
            ha = m.GetNumHeavyAtoms()
            if ha > MAX_HA or ha < 4:
                continue
            rows.append(dict(smiles=Chem.MolToSmiles(m), family=fam, n=n, ends=f"{a}/{b}", ha=ha,
                             has_S=any(x.GetSymbol() == "S" for x in m.GetAtoms())))
for fam, smis in FIXED.items():
    for k, s in enumerate(smis):
        m = Chem.MolFromSmiles(s)
        assert m is not None, s
        rows.append(dict(smiles=Chem.MolToSmiles(m), family=fam, n=k + 1, ends="-", ha=m.GetNumHeavyAtoms(),
                         has_S=any(x.GetSymbol() == "S" for x in m.GetAtoms())))
df = pd.DataFrame(rows).drop_duplicates("smiles")
# hold-out flag: two longest members of every (family, ends) series with >= 4 members
df["holdout_longest"] = False
for (fam, ends), g in df.groupby(["family", "ends"]):
    if len(g) >= 4 and ends != "-":
        df.loc[g.sort_values("n").index[-2:], "holdout_longest"] = True
df.to_csv(ROOT / "experiments/conj/ladders.csv", index=False)
print("molecules", len(df), "| with S", int(df.has_S.sum()), "| hold-out (longest two)", int(df.holdout_longest.sum()))
print(df.groupby("family").agg(n=("smiles", "size"), ha_min=("ha", "min"), ha_max=("ha", "max")).to_string())
