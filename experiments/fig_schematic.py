"""Concept figure (main text, Fig. 1), deliberately simple.

(a) LEH: a MACE encoder gives atom features h_i; onsite blocks (one latent channel per atom) and pair blocks for atom
    pairs within r_c form a symmetric matrix H; its sorted eigenvalues are the excitation energies S_1, S_2, ...;
    excited-state energies and forces follow from E_S0 + lambda_n.
No panel titles (Nature style; the caption describes the panels).
(b) Size consistency: for two molecules further apart than r_c all coupling blocks vanish, H is block diagonal and its
    spectrum is exactly the union of the two molecules' spectra.
(c) Data: QCDGE-derived molecules at equilibrium, thermal xTB-MD geometries and model-driven S1 sampling, labelled
    with TD-wB97X-D (energies and forces of S0-S3, ten singlets); the S1-sampling loop is repeated (active learning).
Output: paper/figs/schematic.{pdf,png}.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import figstyle as fs  # noqa
import matplotlib.pyplot as plt  # noqa
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa

fs.setup()
BLUE = fs.MODEL_COLOR["leh"]; LIGHT = "#bcd3f2"; B2 = fs.SLOTS[7]; LIGHT2 = "#f3c2c1"
fig = plt.figure(figsize=(7.0, 3.0))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 70); ax.set_ylim(0, 30); ax.axis("off")


def arrow(x0, y0, x1, y1, text=None, dy=1.2):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, color=fs.INK2, lw=1.0))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, text, ha="center", va="bottom", fontsize=6.5, color=fs.INK2)


def molecule(x0, y0, n=5, col=BLUE, scale=1.6, label=None):
    pts = np.array([[x0 + i * scale, y0 + (0.55 * scale if i % 2 else 0)] for i in range(n)])
    for i in range(n - 1):
        ax.plot(pts[i:i + 2, 0], pts[i:i + 2, 1], color=fs.INK2, lw=1.2, zorder=1)
    for p in pts:
        ax.add_patch(Circle(p, 0.45, color=col, zorder=2))
    if label:
        ax.text(pts[:, 0].mean(), y0 - 1.6, label, ha="center", fontsize=7, color=fs.INK)
    return pts


def matrix(x0, y0, size, blocks, cell=1.0):
    """blocks: list of (i0, n, onsite colour, coupling colour); coupling within a block for |i-j| <= 2."""
    ax.add_patch(Rectangle((x0, y0), size * cell, size * cell, fill=False, ec=fs.INK2, lw=0.8))
    for i0, n, c_on, c_pair in blocks:
        for i in range(n):
            for j in range(n):
                if i == j or abs(i - j) <= 2:
                    c = c_on if i == j else c_pair
                    ax.add_patch(Rectangle((x0 + (i0 + j) * cell, y0 + (size - 1 - i0 - i) * cell), cell, cell, color=c, lw=0))


def levels(x0, y0, vals, cols, w=3.0, labels=None):
    for k, (v, c) in enumerate(zip(vals, cols)):
        ax.plot([x0, x0 + w], [y0 + v, y0 + v], color=c, lw=1.8)
        if labels:
            ax.text(x0 + w + 0.4, y0 + v, labels[k], va="center", fontsize=6.5, color=fs.INK)


# (a) LEH
ax.text(0.5, 28.6, "a", fontsize=fs.PANEL_LABEL_SIZE, fontweight="bold", color=fs.INK)
pts = molecule(1.5, 19.5)
ax.add_patch(Circle(pts[2], 3.4, fill=False, ec=fs.MUTED, lw=0.8, ls=(0, (2, 2))))
ax.text(pts[2][0], pts[2][1] - 4.6, "r$_c$ = 5 Å", ha="center", fontsize=6.5, color=fs.INK2)
arrow(9.3, 20.5, 13.2, 20.5)
ax.text(11.25, 18.6, "MACE\n$\\mathbf{h}_i$", ha="center", va="top", fontsize=6.5, color=fs.INK2)
matrix(13.8, 16.5, 5, [(0, 5, BLUE, LIGHT)], cell=1.25)
ax.text(16.9, 24.4, "$\\mathbf{H}$: onsite $\\varepsilon(\\mathbf{h}_i)$, pairs $h_{ij}$ within r$_c$", ha="center",
        fontsize=6.3, color=fs.INK2)
arrow(20.8, 19.6, 24.2, 19.6, "eigenvalues", dy=0.5)
levels(24.8, 15.8, [1.0, 2.4, 3.8, 5.4], [BLUE] * 4, labels=["S$_1$", "S$_2$", "S$_3$", "…"])
ax.text(1.0, 13.6, "$E_{S_n}=E_{S_0}+\\lambda_n$,   $\\mathbf{F}_{S_n}=\\mathbf{F}_{S_0}-\\nabla\\lambda_n$",
        fontsize=6.5, color=fs.INK)

# (b) separated fragments
ax.text(33.5, 28.6, "b", fontsize=fs.PANEL_LABEL_SIZE, fontweight="bold", color=fs.INK)
molecule(34.2, 22.0, n=3, col=BLUE, scale=1.3, label="A")
molecule(41.0, 22.0, n=3, col=B2, scale=1.3, label="B")
ax.annotate("", xy=(40.6, 21.0), xytext=(37.6, 21.0), arrowprops=dict(arrowstyle="<->", color=fs.MUTED, lw=0.8))
ax.text(39.1, 19.2, "> r$_c$", ha="center", fontsize=6.5, color=fs.INK2)
matrix(47.2, 19.0, 6, [(0, 3, BLUE, LIGHT), (3, 3, B2, LIGHT2)], cell=1.0)
ax.text(50.2, 25.6, "block diagonal", ha="center", fontsize=6.3, color=fs.INK2)
arrow(53.8, 22.0, 56.6, 22.0)
levels(57.2, 18.8, [1.0, 1.6, 2.5, 3.1, 4.2, 5.0], [B2, BLUE, BLUE, B2, B2, BLUE])
ax.text(61.0, 21.6, "spec(A+B)\n= spec(A) ∪ spec(B)", fontsize=6.5, color=fs.INK, va="center")

# (c) data and active learning
ax.text(0.5, 10.6, "c", fontsize=fs.PANEL_LABEL_SIZE, fontweight="bold", color=fs.INK)
boxes = [("76.6k training molecules\n(QCDGE-derived,\n≤ 10 heavy atoms)", 1.0, 14.0), ("equilibrium +\nthermal (xTB MD)", 17.5, 11.0),
         ("model-driven\nS$_1$/S$_2$ dynamics", 31.5, 11.5), ("TD-DFT labels", 46.0, 9.5), ("train models", 58.5, 9.5)]
for i, (t, x, w) in enumerate(boxes):
    ax.add_patch(FancyBboxPatch((x, 1.6), w, 4.6, boxstyle="round,pad=0.25", fc="white", ec=BLUE if i >= 2 else fs.INK2, lw=0.9))
    ax.text(x + w / 2, 3.9, t, ha="center", va="center", fontsize=6.3, color=fs.INK)
    if i:
        prev_end = boxes[i - 1][1] + boxes[i - 1][2]
        arrow(prev_end + 0.5, 3.9, x - 0.5, 3.9)
ax.add_patch(FancyArrowPatch((63.2, 6.5), (37.2, 6.5), connectionstyle="arc3,rad=0.12", arrowstyle="-|>",
                             mutation_scale=9, color=BLUE, lw=1.0))
ax.text(50.2, 8.4, "active learning: twisted and low-gap S$_1$ regions", ha="center", fontsize=6.3, color=fs.INK)
out = ROOT / "paper/figs/schematic"
fig.savefig(out.with_suffix(".pdf")); fig.savefig(out.with_suffix(".png"))
print("written", out)
