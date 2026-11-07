"""Supplementary figures: random selections of the molecules of the training set and of the test and extension sets.

molecules_train: QCDGE-derived training molecules with <= 8, 9, and 10 heavy atoms, training molecules with S1- or
S2-sampled geometries, and the molecules of the seven validation and five test reference trajectories (all, named).
molecules_test: molecules with Bemis-Murcko scaffolds absent from training, larger molecules (11-20 heavy atoms), dyes
(training and test scaffolds), and one member of each of the 13 ladder series with hydrogen end groups and two or
three repeat units (named).
Random selections with a fixed seed from the released dataset (data/wb97_prod_v4.pkl, split_prod_v4.json), sorted by
the number of heavy atoms. Structures: SMILES where the dataset has them (larger molecules, dyes, ladders), otherwise
bonds perceived with RDKit from the minimum geometry (QCDGE molecules, reference trajectories), drawn in 2D with
hydrogens implicit and the same bond length within a panel (larger molecules scaled down to fit their cell).
Output: paper/figs/molecules_{train,test}.{pdf,png,csv}.
"""
import csv
import io
import json
import pickle
from pathlib import Path

import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDepictor, rdDetermineBonds
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry import Point3D

import figstyle
from figstyle import plt

RDLogger.DisableLog("rdApp.*")
rdDepictor.SetPreferCoordGen(True)
figstyle.setup()
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/figs"
PT = Chem.GetPeriodicTable()
TRAJ = [("styrene_s1", "styrene"), ("azobenzene_s1", "azobenzene"), ("cyclohexenone_s1", "cyclohexenone"),
        ("hexatriene_s1", "hexatriene"), ("cytosine_s1", "cytosine"), ("benzaldehyde_s1", "benzaldehyde"),
        ("cinnamaldehyde_s1", "cinnamaldehyde"), ("acrolein_s1", "acrolein"), ("uracil_s1", "uracil"),
        ("stilbene_s1", r"$\it{trans}$-stilbene"), ("hbdi_s1", "HBDI"), ("dmabn_s1", "DMABN")]  # 7 validation, 5 test
# ladder panel: hydrogen end groups and three repeat units (two for azobenzenes, the longest in the dataset, and for
# phenylenevinylenes, whose only three-unit member has 30 heavy atoms); the five core series ("-" ends) by their third
# member (tetracene, picene, anthraquinone, phenazine, porphine)
LAD_N = {"azobenzene": 2, "phenylenevinylene": 2}
LAD_NAME = {"polyene": "polyene", "diphenylpolyene": "diphenylpolyene", "oligophenylene": "oligophenylene",
            "phenylenevinylene": "phenylenevinylene", "oligofuran": "oligofuran", "oligopyrrole": "oligopyrrole",
            "oligothiophene": "oligothiophene", "azobenzene": "azobenzene", "acene": "acene", "phenacene": "phenacene",
            "aza_aromatic": "aza-aromatic", "quinone": "quinone", "classic_dye": "classic dye"}


def from_xyz(Z, R):
    xyz = f"{len(Z)}\n\n" + "\n".join(f"{PT.GetElementSymbol(int(z))} {x:.6f} {y:.6f} {w:.6f}" for z, (x, y, w) in zip(Z, R))
    m = Chem.MolFromXYZBlock(xyz)
    rdDetermineBonds.DetermineBonds(m, charge=0)
    Chem.AssignStereochemistryFrom3D(m)
    return Chem.MolToSmiles(Chem.RemoveHs(m))


def smiles_of(m):
    try:
        return m.get("smiles") or from_xyz(m["Z"], m["R"])
    except Exception:  # bond orders not assignable (rare): skip the molecule
        return None


def draw(smi, size, bond_px, horizontal=False):
    """horizontal (ladder series): drop hydrogens that only define double-bond stereo (the end hydrogens of the polyene
    SMILES) and rotate the 2D layout so that its long axis is horizontal."""
    mol = Chem.MolFromSmiles(smi)
    if horizontal:
        hp = Chem.RemoveHsParameters(); hp.removeDefiningBondStereo = True
        mol = Chem.RemoveHs(mol, hp)
    rdDepictor.Compute2DCoords(mol)
    if horizontal:
        conf = mol.GetConformer()
        P = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y] for i in range(mol.GetNumAtoms())])
        P -= P.mean(0)
        Vt = np.linalg.svd(P, full_matrices=False)[2]
        if np.linalg.det(Vt) < 0:  # rotation, not reflection
            Vt[1] *= -1
        for i, (x, y) in enumerate(P @ Vt.T):
            conf.SetAtomPosition(i, Point3D(float(x), float(y), 0.0))
    dr = rdMolDraw2D.MolDraw2DCairo(*size)
    o = dr.drawOptions()
    o.fixedBondLength = bond_px  # same scale for all molecules of a figure; larger ones are scaled down to fit
    o.bondLineWidth = 2; o.clearBackground = True; o.padding = 0.05; o.baseFontSize = 0.8
    o.additionalAtomLabelPadding = 0.1
    dr.DrawMolecule(mol); dr.FinishDrawing()
    return plt.imread(io.BytesIO(dr.GetDrawingText()), format="png")


def pick(entries, n, rng):
    """n random molecules with a drawable structure, sorted by heavy-atom count."""
    out = []
    for i in rng.permutation(len(entries)):
        smi = smiles_of(entries[i])
        if smi:
            out.append((entries[i], smi))
        if len(out) == n:
            break
    return sorted(out, key=lambda t: (t[0]["n_heavy"], t[0]["mol"]))


def figure(name, panels, figsize):
    """panels: list of (label, [(entry or None, smiles, caption)], ncols, height ratio, canvas size px, bond length px[,
    horizontal]);
    canvas and bond length per panel so that typical molecules of the panel fill most of their cell."""
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(len(panels), 1, figure=fig, height_ratios=[p[3] for p in panels], hspace=0.1)
    labels, rows = [], []
    for k, (lab, mols, ncols, _, size, bond_px, *opt) in enumerate(panels):
        nrows = int(np.ceil(len(mols) / ncols))
        sub = GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=gs[k], wspace=0.02, hspace=0.3 if mols[0][2] else 0.02)
        for j, (e, smi, cap) in enumerate(mols):
            ax = fig.add_subplot(sub[j // ncols, j % ncols])
            ax.imshow(draw(smi, size, bond_px, horizontal=bool(opt and opt[0]))); ax.set_axis_off()
            if cap:
                ax.text(0.5, -0.02, cap, transform=ax.transAxes, ha="center", va="top", fontsize=6)
            if j == 0:
                labels.append((ax, lab))
            rows.append(dict(panel=lab, id=e["mol"] if e else cap, n_heavy=Chem.MolFromSmiles(smi).GetNumAtoms(),
                             smiles=smi))
    figstyle.panel_labels(fig, labels, dy_pt=0.0)
    out = OUT / name
    fig.savefig(out.with_suffix(".pdf")); fig.savefig(out.with_suffix(".png"))
    with open(out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("written", out.with_suffix(".pdf"), len(rows), "molecules")


def main():
    pm = pickle.load(open(ROOT / "data/wb97_prod_v4.pkl", "rb"))
    ps = json.load(open(ROOT / "experiments/al/split_prod_v4.json"))
    train = ps["train"]
    qmin = {pm[i]["mol"]: pm[i] for i in train if pm[i]["kind"] == "qcdge_eq"}
    rng = np.random.default_rng(0)
    small = [m for m in qmin.values() if m["n_heavy"] <= 8]
    nine = [m for m in qmin.values() if m["n_heavy"] == 9]
    ten = [m for m in qmin.values() if m["n_heavy"] == 10]
    exc = sorted({pm[i]["mol"] for i in train if pm[i]["kind"] == "excmd"} & set(qmin))
    exc = [qmin[k] for k in exc]  # structure from the minimum of the molecule (the sampled geometries are distorted)
    traj = []
    for t, nm in TRAJ:
        d = np.load(ROOT / f"experiments/md/reference/{t}/traj.npz")
        traj.append((None, from_xyz(d["Z"], d["R"][0]), nm))
    nc = 8
    cap0 = lambda sel: [(e, s, "") for e, s in sel]
    sm = ((260, 190), 34)  # QCDGE-sized molecules (<= 10 heavy atoms)
    figure("molecules_train", [
        ("a", cap0(pick(small, 2 * nc, rng)), nc, 1.0, *sm), ("b", cap0(pick(nine, 2 * nc, rng)), nc, 1.0, *sm),
        ("c", cap0(pick(ten, 2 * nc, rng)), nc, 1.0, *sm), ("d", cap0(pick(exc, 2 * nc, rng)), nc, 1.0, *sm),
        ("e", traj, 6, 1.5, (330, 230), 34)], figsize=(7.2, 6.6))
    scaf = [pm[i] for i in ps["test_scaffold_eq"]]
    large = [pm[i] for i in ps["test_large_eq"]]
    conj = [m for m in pm if m["kind"] == "eq" and m["source"] in ("DYE", "LAD")]
    dyes = [m for m in conj if m["source"] == "DYE"]
    lad = []
    for fam in sorted({m["family"] for m in conj if m["source"] == "LAD"}):
        members = [m for m in conj if m["source"] == "LAD" and m["family"] == fam]
        rng.permutation(len(members))  # draw of the earlier random ladder panel, keeps panels a-c unchanged
        n = LAD_N.get(fam, 3)
        m = next(x for x in members if x["n"] == n and x["ends"] in ("H/H", "-"))
        lad.append((m, m["smiles"], LAD_NAME[fam]))
    lad = sorted(lad, key=lambda t: (t[0]["n_heavy"], t[0]["mol"]))
    del pm
    figure("molecules_test", [
        ("a", cap0(pick(scaf, 16, rng)), 8, 0.8, *sm), ("b", cap0(pick(large, 12, rng)), 6, 1.0, (330, 230), 26),
        ("c", cap0(pick(dyes, 12, rng)), 6, 1.0, (330, 230), 28), ("d", lad, 5, 1.6, (400, 220), 30, True)],
        figsize=(7.2, 6.6))


if __name__ == "__main__":
    main()
