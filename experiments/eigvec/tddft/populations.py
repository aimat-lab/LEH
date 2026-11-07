"""Per-atom populations of TURBOMOLE TD-DFT excitations from the files of an escf run (no integral code needed).

AO overlap: the mos file holds all MOs (square C with C^T S C = 1), so S = (C C^T)^-1. AO -> atom map: TURBOMOLE orders
AOs atom by atom in coord order; the number of AOs per atom follows from the basis file (spherical functions). Both are
checked against the Mulliken charges of the archived ridft run with $pop (mulliken_check).

Excitation vectors: sing_a of an RPA (rpas) run. Per root n, the 1-particle transition density in the Loewdin basis is
T = S^1/2 C_occ X C_vir^T S^1/2, with X the (X+Y) vector normalised to sum X^2 = 1. Following Plasser's fragment
analysis, Omega_AB = sum_{mu in A, nu in B} T_mu,nu^2 (sum 1), hole_A = sum_B Omega_AB, electron_B = sum_A Omega_AB,
and the excitation population p_A = (hole_A + electron_A) / 2, the counterpart of the per-atom weight of an LEH
eigenvector. Transition charges q_A = sum_{mu in A} T_mu,mu. pop_X: p_A from the amplitudes X = ((X+Y) + (X-Y)) / 2.
Checked on test geometries: sing_a holds X+Y then X-Y per root, occupied index slow (largest |X_ia|^2 equal to the
dominant contributions printed by escf), (X+Y).(X-Y) = 1, Mulliken charges reproduced to 5e-6.
"""
import re
from pathlib import Path

import numpy as np

NSPH = {"s": 1, "p": 3, "d": 5, "f": 7, "g": 9}


def read_mos(path):
    txt = Path(path).read_text()
    blocks = re.split(r"\n\s*\d+\s+a\s+eigenvalue=", txt)[1:]
    eps, cols = [], []
    for b in blocks:
        head, body = b.split("\n", 1)
        eps.append(float(head.split()[0].replace("D", "E")))
        nsaos = int(re.search(r"nsaos=\s*(\d+)", head).group(1))
        body = body.split("$end")[0]
        nums = re.findall(r"-?\d?\.\d{14}D[+-]\d\d", body)
        assert len(nums) == nsaos, (len(nums), nsaos)
        cols.append([float(x.replace("D", "E")) for x in nums])
    return np.array(eps), np.array(cols).T  # C[ao, mo]


def basis_sizes(basis_path):
    """Number of spherical AOs per element symbol from a TURBOMOLE basis file."""
    out, el = {}, None
    for l in Path(basis_path).read_text().splitlines():
        m = re.match(r"^([a-z]{1,2})\s+\S+", l)
        if m:
            el = m.group(1)
            out[el] = 0
            continue
        m = re.match(r"^\s+\d+\s+([spdfg])\s*$", l)
        if m and el:
            out[el] += NSPH[m.group(1)]
    return out


def read_coord(path):
    rows = []
    for l in Path(path).read_text().split("$coord")[1].split("$")[0].strip().splitlines():
        x, y, z, s = l.split()[:4]
        rows.append((float(x), float(y), float(z), s.lower()))
    return np.array([r[:3] for r in rows]) * 0.529177210903, [r[3] for r in rows]


def ao_atom(syms, sizes):
    return np.repeat(np.arange(len(syms)), [sizes[s] for s in syms])


def n_occ(control_text):
    return int(re.search(r"\$closed shells\s*\n\s*a\s+1-(\d+)", control_text).group(1))


def read_sing_a(path, nocc, nvir):
    """Vectors of sing_a: list of (eigenvalue, array [k*nocc*nvir]) with k = 1 (TDA) or 2 (RPA: X+Y, X-Y)."""
    txt = Path(path).read_text()
    parts = re.split(r"\n\s*\d+\s+eigenvalue\s*=", txt)
    out = []
    for b in parts[1:]:
        head, body = b.split("\n", 1)
        nums = re.findall(r"-?\d?\.\d{14}D[+-]\d\d", body.split("$end")[0])
        out.append((float(head.split()[0].replace("D", "E")), np.array([float(x.replace("D", "E")) for x in nums])))
    return out


def sqrtm_sym(S):
    e, U = np.linalg.eigh(S)
    return (U * np.sqrt(e)) @ U.T


def analyse(d, nroots=10):
    """d: directory with control, coord, basis, mos, sing_a. Returns dict of per-root, per-atom arrays."""
    d = Path(d)
    ctl = (d / "control").read_text()
    no = n_occ(ctl)
    eps, C = read_mos(d / "mos")
    nao, nmo = C.shape
    assert nao == nmo, "linear dependencies removed: S cannot be recovered from C"
    S = np.linalg.inv(C @ C.T)
    R, syms = read_coord(d / "coord")
    sizes = basis_sizes(d / "basis")
    amap = ao_atom(syms, sizes)
    assert len(amap) == nao, (len(amap), nao)
    nat = len(syms)
    Sh = sqrtm_sym(S)
    CL = Sh @ C  # MOs in the Loewdin basis (orthonormal columns)
    Co, Cv = CL[:, :no], CL[:, no:]
    nv = nmo - no
    vecs = read_sing_a(d / "sing_a", no, nv)
    k = len(vecs[0][1]) // (no * nv)
    assert k * no * nv == len(vecs[0][1]), (len(vecs[0][1]), no, nv)
    P = np.zeros((nat, nao))
    P[amap, np.arange(nao)] = 1.0
    res = dict(omega=[], hole=[], elec=[], qtr=[], exc=[], norm_xy=[], xy_overlap=[], pop_X=[], R=R, syms=syms,
               s_diag_dev=float(np.abs(np.diag(S) - 1).max()))
    for e, v in vecs[:nroots]:
        Xp = v[: no * nv].reshape(no, nv)  # (X+Y), occupied index slow (checked against escf dominant contributions)
        if k == 2:
            Xm = v[no * nv:].reshape(no, nv)
            res["xy_overlap"].append(float((Xp * Xm).sum()))
            Xa = (Xp + Xm) / 2  # excitation amplitudes X alone (robustness check of the X+Y populations)
            Xa = Xa / np.linalg.norm(Xa)
            OmX = P @ ((Co @ Xa @ Cv.T) ** 2) @ P.T
            res["pop_X"].append((OmX.sum(1) + OmX.sum(0)) / 2)
        X = Xp / np.linalg.norm(Xp)
        T = Co @ X @ Cv.T
        Om = P @ (T ** 2) @ P.T
        res["omega"].append(Om)
        res["hole"].append(Om.sum(1))
        res["elec"].append(Om.sum(0))
        res["qtr"].append(P @ np.diag(T))
        res["exc"].append(e)
        res["norm_xy"].append(float(np.linalg.norm(Xp)))
    for q in ("omega", "hole", "elec", "qtr", "exc", "pop_X"):
        res[q] = np.array(res[q])
    res["pop"] = (res["hole"] + res["elec"]) / 2
    return res


def mulliken_check(d, ridft_out):
    """Max |Mulliken charge (from S and the occupied MOs) - TURBOMOLE Mulliken charge| for the AO -> atom map check."""
    d = Path(d)
    ctl = (d / "control").read_text()
    no = n_occ(ctl)
    _, C = read_mos(d / "mos")
    S = np.linalg.inv(C @ C.T)
    R, syms = read_coord(d / "coord")
    amap = ao_atom(syms, basis_sizes(d / "basis"))
    D = 2 * C[:, :no] @ C[:, :no].T
    gross = np.bincount(amap, weights=(D * S).sum(1), minlength=len(syms))
    Zval = np.array([{"h": 1, "c": 6, "n": 7, "o": 8, "f": 9, "s": 16}[s] for s in syms])
    q = Zval - gross
    txt = Path(ridft_out).read_text()
    i = txt.find("atomic populations from total density:")
    if i < 0:
        return np.nan
    lines = txt[i:].splitlines()
    h = next(k for k, l in enumerate(lines) if l.strip().startswith("atom") and "charge" in l)
    ref = np.array([float(lines[h + 1 + a].split()[1]) for a in range(len(syms))])
    return float(np.abs(q - ref).max())
