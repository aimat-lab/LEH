"""TURBOMOLE driver: TDA-PBE0/def2-SVP, E0/F0, N singlet roots, analytic S1..Sq gradients.

One egrad run per excited state ($exopt n), energy-ordered roots in C1 symmetry.
"""
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

HARTREE_EV = 27.211386245988
BOHR_A = 0.529177210903
SYM = {1: "h", 6: "c", 7: "n", 8: "o", 9: "f", 16: "s"}
ZOF = {v: k for k, v in SYM.items()}

DEFINE_IN = """

a coord
*
no
b all {basis}
*
eht
y
0
y
dft
on
func {func}
grid {grid}
q
ri
on
m 1000
q
scf
conv
7
iter
300

ex
{extype}
q
a {nroots}
q
q
*
"""


def tm_env(nthreads):
    env = dict(os.environ)
    env.update(PARA_ARCH="SMP", PARNODES=str(nthreads), OMP_NUM_THREADS=str(nthreads))
    tdir = env["TURBODIR"]
    env["PATH"] = f"{tdir}/bin/em64t-unknown-linux-gnu_smp:{tdir}/scripts:" + env["PATH"]
    return env


def write_coord(path, Z, R_ang):
    with open(path, "w") as f:
        f.write("$coord\n")
        for z, r in zip(Z, np.asarray(R_ang) / BOHR_A):
            f.write(f"{r[0]:20.14f} {r[1]:20.14f} {r[2]:20.14f}  {SYM[int(z)]}\n")
        f.write("$end\n")


def run(cmd, cwd, env, log):
    with open(Path(cwd) / log, "w") as f:
        p = subprocess.run(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT, shell=isinstance(cmd, str))
    return p.returncode


LEVELS = {
    "pbe0_svp_tda": dict(func="pbe0", basis="def2-SVP", extype="ciss", grid="m4"),
    "wb97xd_631gs_rpa": dict(func="wb97x-d", basis="6-31G*", extype="rpas", grid="m4"),
    "wb97xd_631gs_tda": dict(func="wb97x-d", basis="6-31G*", extype="ciss", grid="m4"),
}


def make_template(tdir, Z, R_ang, nroots=8, nthreads=4, level="pbe0_svp_tda"):
    tdir = Path(tdir)
    tdir.mkdir(parents=True, exist_ok=True)
    write_coord(tdir / "coord", Z, R_ang)
    env = tm_env(nthreads)
    with open(tdir / "define.in", "w") as f:
        f.write(DEFINE_IN.format(nroots=nroots, **LEVELS[level]))
    subprocess.run("define < define.in > define.out 2>&1", cwd=tdir, env=env, shell=True)
    ctl = (tdir / "control").read_text()
    assert "$scfinstab" in ctl and "$dft" in ctl, "define failed in " + str(tdir)
    ctl = ctl.replace("$dft\n", "$dft\n   weight derivatives\n", 1)
    ctl = re.sub(r"#?\$maxcor.*\n", "", ctl)
    ctl = ctl.replace("$end", "$maxcor 500 MiB per_core\n$rpaconv 6\n$denconv 1d-7\n$end")
    (tdir / "control").write_text(ctl)
    run("ridft", tdir, env, "ridft.out")
    assert "convergence criteria satisfied" in (tdir / "ridft.out").read_text()
    return tdir


def _set_roots_and_state(ctl, nroots, state):
    ctl = re.sub(r"(\$soes\n\s*a\s+)\d+", lambda m: m.group(1) + str(nroots), ctl)
    ctl = re.sub(r"\$exopt.*\n", "", ctl)
    if state:
        ctl = ctl.replace("$end", f"$exopt {state}\n$end")
    return ctl


def parse_gradient(path, natoms):
    lines = Path(path).read_text().splitlines()
    i = max(k for k, l in enumerate(lines) if "cycle =" in l)
    m = re.search(r"energy =\s*(-?[\d.]+)", lines[i])
    e = float(m.group(1))
    g = [[float(x.replace("D", "E")) for x in l.split()] for l in lines[i + 1 + natoms : i + 1 + 2 * natoms]]
    return e, np.array(g)


def parse_exc(text):
    return [float(x) for x in re.findall(r"Excitation energy:\s+(-?[\d.Ee+-]+)", text)]


def parse_root_properties(text):
    """Oscillator strengths (length representation) and electric transition dipoles (length rep., a.u.) per root."""
    blocks = re.split(r"\n\s*Excitation energy:\s", text)[1:]
    osc, tdip = [], []
    for b in blocks:
        m = re.search(r"Oscillator strength:.*?length representation:\s+(-?[\d.Ee+-]+)", b, re.S)
        osc.append(float(m.group(1)) if m else np.nan)
        m = re.search(r"Electric transition dipole moment \(length rep\.\):\s*\n\s*x\s+(\S+).*?\n\s*y\s+(\S+).*?\n"
                      r"\s*z\s+(\S+)", b, re.S)
        tdip.append([float(m.group(i)) for i in (1, 2, 3)] if m else [np.nan] * 3)
    return np.array(osc), np.array(tdip).reshape(-1, 3)


def parse_mulliken(text, natoms):
    """Mulliken atomic charges from a ridft output with $pop (NaN if absent)."""
    i = text.find("atomic populations from total density:")
    if i < 0:
        return np.full(natoms, np.nan)
    lines = text[i:].splitlines()
    h = next(k for k, l in enumerate(lines) if l.strip().startswith("atom") and "charge" in l)
    return np.array([float(lines[h + 1 + a].split()[1]) for a in range(natoms)])


def parse_mo_energies(mos_path):
    """All MO eigenvalues (Hartree) from a TURBOMOLE mos file."""
    txt = Path(mos_path).read_text()
    return np.array([float(x.replace("D", "E")) for x in re.findall(r"eigenvalue=\s*(\S+)", txt)])


def parse_n_occ(control_text):
    m = re.search(r"\$closed shells\s*\n\s*a\s+1-(\d+)", control_text)
    return int(m.group(1)) if m else -1


def single_point(template, workdir, Z, R_ang, nroots=8, nexgrad=3, nthreads=4, keep=False):
    """Compute E0, F0, nroots excitation energies and nexgrad excited forces. Returns dict (atomic units).

    Also returns Mulliken charges, all MO energies (Hartree), the number of doubly occupied MOs, and for the
    nroots roots the oscillator strengths and electric transition dipoles (length representation, a.u.).
    """
    t0 = time.time()
    wd = Path(workdir)
    if wd.exists():
        shutil.rmtree(wd)
    wd.mkdir(parents=True)
    for f in ["control", "basis", "auxbasis", "mos"]:
        shutil.copy(Path(template) / f, wd / f)
    write_coord(wd / "coord", Z, R_ang)
    env = tm_env(nthreads)
    n = len(Z)
    flags = {}
    ctl0 = (wd / "control").read_text()
    if "$pop" not in ctl0:  # Mulliken population analysis in the SCF run only
        (wd / "control").write_text(ctl0.replace("$end", "$pop\n$end"))
    run("ridft", wd, env, "ridft.out")
    txt = (wd / "ridft.out").read_text()
    flags["scf_converged"] = "convergence criteria satisfied" in txt
    if not flags["scf_converged"]:
        return dict(ok=False, flags=flags)
    E0 = float(re.findall(r"total energy\s*=\s*(-?[\d.]+)", txt)[-1])
    run("rdgrad", wd, env, "rdgrad.out")
    e0g, g0 = parse_gradient(wd / "gradient", n)
    flags["e0_consistent"] = abs(e0g - E0) < 1e-6
    extra = dict(mulliken=parse_mulliken(txt, n), mo_energies=parse_mo_energies(wd / "mos"))
    exc, gex, eex = None, [], []
    base = re.sub(r"\$pop\s*\n", "", (wd / "control").read_text())
    extra["n_occ"] = parse_n_occ(base)
    for s in range(1, nexgrad + 1):
        (wd / "gradient").unlink(missing_ok=True)
        nr = nroots if s == 1 else max(s + 2, 5)
        (wd / "control").write_text(_set_roots_and_state(base, nr, s))
        run("egrad", wd, env, f"egrad{s}.out")
        out = (wd / f"egrad{s}.out").read_text()
        ok = "egrad : all done" in out and (wd / "gradient").exists()
        if not ok and "No convergence within" in out:
            # the default 35 Davidson iterations are occasionally too few for the highest requested root (long
            # polyenes at their minimum); retry once with a larger limit, keeping the failed output
            (wd / f"egrad{s}.out").rename(wd / f"egrad{s}_noconv.out")
            ctl = (wd / "control").read_text()
            ctl = re.sub(r"\$escfiterlimit\s+\d+\s*\n", "", ctl).replace("$end", "$escfiterlimit 200\n$end")
            (wd / "control").write_text(ctl)
            (wd / "gradient").unlink(missing_ok=True)
            run("egrad", wd, env, f"egrad{s}.out")
            out = (wd / f"egrad{s}.out").read_text()
            ok = "egrad : all done" in out and (wd / "gradient").exists()
            flags[f"egrad{s}_iterlimit_retry"] = True
        flags[f"egrad{s}_ok"] = ok
        if not ok:
            gex.append(np.full((n, 3), np.nan)); eex.append(np.nan)
            if s == 1:
                return dict(ok=False, flags=flags)
            continue
        roots = parse_exc(out)
        if s == 1:
            exc = roots[:nroots]
            osc, tdip = parse_root_properties(out)
            extra["osc_strength"], extra["tdip"] = osc[:nroots], tdip[:nroots]
        else:
            # root consistency: the s-th root with fewer requested roots must match the 8-root run
            flags[f"root{s}_dev_hartree"] = abs(roots[s - 1] - exc[s - 1])
        e, g = parse_gradient(wd / "gradient", n)
        flags[f"egrad{s}_energy_dev"] = abs(e - (E0 + roots[s - 1]))
        gex.append(g); eex.append(e)
    if not keep:
        shutil.rmtree(wd)
    return dict(ok=True, flags=flags, E0=E0, grad0=g0, exc=np.array(exc), gradex=np.stack(gex),
                time=time.time() - t0, **extra)
