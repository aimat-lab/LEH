"""Shared figure style for the paper (static, light mode). Reference categorical palette in fixed slot order;
each model keeps its color in every figure."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
# One hue per model family, fixed across all figures; variants of a family differ by line style / marker.
# Validated (dataviz validate_palette.py, light): LEH blue, softmin aqua, X-MACE violet, MS-ANI magenta, OMNI green.
MODEL_COLOR = {"latent_k1": SLOTS[0], "direct_sum": SLOTS[1], "direct_softmin": SLOTS[2], "direct_mean": SLOTS[3],
               "latent_diag": SLOTS[0], "latent_k2_heavy": SLOTS[0], "reference": "#0b0b0b",
               "leh": SLOTS[0], "leh_L2": SLOTS[0], "softmin": SLOTS[2], "xmace": SLOTS[6], "msani": SLOTS[4],
               "vecmsani": SLOTS[5], "omnift": SLOTS[5], "omniftA": SLOTS[5], "omni_zeroshot": SLOTS[5]}
LABEL = {"latent_k1": "LEH", "direct_sum": "Direct-sum", "direct_softmin": "Direct-softmin",
         "direct_mean": "Direct-mean", "latent_diag": "LEH, diagonal", "latent_k2_heavy": "LEH, k=2 heavy",
         "reference": "TDDFT", "leh": "LEH", "leh_L2": "LEH, L=2", "softmin": "Direct-softmin", "xmace": "X-MACE",
         "msani": "MS-ANI", "vecmsani": "OMNI-P2x arch., scratch", "omnift": "OMNI-P2x ft, authors' protocol",
         "omniftA": "OMNI-P2x ft, adapted",  # all layers trainable, gap weight 1 (details in the captions)
         "omni_zeroshot": "OMNI-P2x pretrained"}
SHORT = {"omnift": "OMNI-P2x ft (authors)", "omniftA": "OMNI-P2x ft (adapted)", "vecmsani": "OMNI-P2x arch. (scratch)",
         "omni_zeroshot": "OMNI-P2x, pretrained"}  # compact names for panel titles
STYLE = {"leh_L2": dict(linestyle=(0, (4, 2)), marker="D"), "omnift": dict(linestyle=(0, (4, 2)), marker="D"),
         "omniftA": dict(linestyle=(0, (6, 1.5, 1, 1.5)), marker="P"),
         "omni_zeroshot": dict(linestyle=(0, (1, 2)), marker=""), "xmace": dict(marker="^"), "msani": dict(marker="v"),
         "vecmsani": dict(marker="s"), "softmin": dict(marker="o"), "leh": dict(marker="o")}
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def setup():
    plt.rcParams.update({
        # Nature figure specifications: sans-serif text of 5-7 pt at the final size (183 mm double column, <= 170 mm
        # high), panel letters 8 pt bold, no background gridlines, no coloured text, units in parentheses.
        "font.family": "sans-serif", "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6, "legend.frameon": False,
        "axes.edgecolor": INK2, "axes.linewidth": 0.6, "axes.labelcolor": INK, "xtick.color": INK2,
        "ytick.color": INK2, "text.color": INK, "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.5,
        "grid.linestyle": "-", "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.5, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "lines.markersize": 4.5, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 300,
        "savefig.bbox": "tight", "pdf.fonttype": 42,
    })


PANEL_LABEL_SIZE = 8  # Nature: bold lowercase panel letters, 8 pt; no panel titles (content goes into the caption)


def panel_labels(fig, labels, dy_pt=3.0):
    """Bold lowercase panel letters at the top left of each panel, aligned with the outer left edge of the panel
    (tick and axis labels included) and just above the axes. labels: list of (ax, "a"). Call once, after layout."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for ax, s in labels:
        x0 = inv.transform(ax.get_tightbbox(r).p0)[0]
        y1 = inv.transform(ax.get_window_extent(r).p1)[1] + dy_pt / 72 / fig.get_figheight()
        fig.text(x0, y1, s, fontsize=PANEL_LABEL_SIZE, fontweight="bold", color=INK, ha="left", va="bottom")


def line(ax, x, y, model, **kw):
    """2px-equivalent line with surface-ringed markers in the model's color."""
    kw.setdefault("marker", "o")
    kw.setdefault("label", LABEL.get(model, model))
    return ax.plot(x, y, color=MODEL_COLOR.get(model, SLOTS[6]), markeredgecolor="white", markeredgewidth=0.8, **kw)
