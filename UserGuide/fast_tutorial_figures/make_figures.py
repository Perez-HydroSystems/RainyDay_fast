"""Static figures of the RainyDay_fast User's Guide (UserGuide/RainyDay_fast_tutorial.ipynb).

    python make_figures.py              # all four PNGs, written next to this script
    python make_figures.py bands strict # only some of them: steps, catsearch, strict, bands

One function per figure:
  fig_steps      fast-steps_flow.png          the five SST steps and the RainyDay_fast changes on each (no data)
  fig_catsearch  feat-catsearch_schematic.png the original irregular-domain catalog search on a small grid (no data)
  fig_strict     feat-strict_schematic.png    original vs strict transposition on a synthetic domain (no data)
  fig_bands      feat-bands_schematic.png     the bands of Case Study 1 (Iowa pixel, 24 h, T = 100 yr), read from
                                              ../fast_tutorial_data

Only numpy, pandas and matplotlib. With the RainyDay_Env of RainyDay.yml's era (Python 3.8, matplotlib 3.5) the PNGs are
byte-identical to the ones in this folder; other matplotlib versions give visually identical figures.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Patch, Polygon, Rectangle
from matplotlib.path import Path

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "fast_tutorial_data"))

BLACK, INK, GREY, BLUE, ORANGE, RED = "#222222", "#222222", "#9e9e9e", "#1f5fa8", "#d9822b", "#c62828"


# =====================================================================================================================
# fast-steps_flow.png
# =====================================================================================================================
def fig_steps(out=os.path.join(HERE, "fast-steps_flow.png")):
    """The five SST steps (left) and the RainyDay_fast changes that act on each step (right), coloured by kind."""
    kinds = {  # edge colour, fill colour, legend label
        "result": ("#c62828", "#fbe9e7", "changes results vs the original RainyDay"),
        "speed": ("#616161", "#eeeeee", "bit-identical speed-up"),
        "new": ("#1f5fa8", "#e3edf8", "new option or output"),
    }
    steps = [
        ("1", "Transposition domain", [("whole-box rule", "result"), ("cropped input", "speed")]),
        ("2", "Storm catalog", [("domain-aware search", "result"), ("scan engines, COMPUTE", "speed"),
                                ("light catalog", "speed"), ("CATALOGPATH", "new")]),
        ("3", "Storms per year (Poisson)", []),
        ("4", "Resample, transpose,\nbasin depth per draw", [("strict transposition", "result"),
                                                              ("compiled code", "speed"), ("table engine", "speed"),
                                                              ("MULTIPLEDURATIONS", "new")]),
        ("5", "Rank, return periods", [("all_realizations", "new"), ("RECORDCI", "new"), ("FREQENGINE", "new"),
                                       ("compare_runs", "new")]),
    ]
    note3 = "unchanged; set INCLUDEYEARS so the years are counted right"

    fig = plt.figure(figsize=(10, 5.6), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()

    def width_of(text, size, weight="normal"):
        t = ax.text(0, 0, text, fontsize=size, fontweight=weight)
        bb = t.get_window_extent(renderer=renderer)
        t.remove()
        (x0, _), (x1, _) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        return x1 - x0

    ax.text(0.03, 0.955, "RainyDay steps", fontsize=11, fontweight="bold", va="center", color="#222222")
    ax.text(0.37, 0.955, "What RainyDay_fast changes", fontsize=11, fontweight="bold", va="center", color="#222222")

    top, bottom = 0.89, 0.17
    h = 0.105
    ys = [top - i * (top - bottom - h) / (len(steps) - 1) - h / 2 for i in range(len(steps))]
    bx0, bx1 = 0.03, 0.31
    for i, (num, name, tags) in enumerate(steps):
        y = ys[i]
        ax.add_patch(FancyBboxPatch((bx0, y - h / 2), bx1 - bx0, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                    linewidth=1.3, edgecolor="#222222", facecolor="white"))
        ax.text(bx0 + 0.018, y, num, fontsize=13, fontweight="bold", va="center", ha="left", color="#222222")
        ax.text(bx0 + 0.05, y, name, fontsize=10, va="center", ha="left", color="#222222", linespacing=1.15)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch(((bx0 + bx1) / 2, y - h / 2 - 0.004),
                                         ((bx0 + bx1) / 2, ys[i + 1] + h / 2 + 0.004),
                                         arrowstyle="-|>", mutation_scale=12, linewidth=1.2, color="#222222"))
        # connector from the step box to its tags
        ax.plot([bx1 + 0.008, 0.355], [y, y], color="#9e9e9e", linewidth=1.0, linestyle=":")
        x = 0.37
        if not tags:
            ax.text(x, y, note3, fontsize=9.5, va="center", ha="left", color="#616161", style="italic")
            continue
        th = 0.058
        for label, kind in tags:
            edge, fill, _ = kinds[kind]
            w = width_of(label, 9.5) + 0.022
            ax.add_patch(FancyBboxPatch((x, y - th / 2), w, th, boxstyle="round,pad=0.002,rounding_size=0.015",
                                        linewidth=1.3, edgecolor=edge, facecolor=fill))
            ax.text(x + w / 2, y, label, fontsize=9.5, va="center", ha="center", color="#222222")
            x += w + 0.014
        assert x < 0.995, "tags overflow the figure: " + name

    # legend
    lx, ly = 0.03, 0.055
    for kind in ("result", "speed", "new"):
        edge, fill, label = kinds[kind]
        ax.add_patch(FancyBboxPatch((lx, ly - 0.018), 0.03, 0.036, boxstyle="round,pad=0.002,rounding_size=0.01",
                                    linewidth=1.3, edgecolor=edge, facecolor=fill))
        ax.text(lx + 0.04, ly, label, fontsize=9.5, va="center", ha="left", color="#222222")
        lx += 0.04 + width_of(label, 9.5) + 0.05

    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print("wrote", out)


# =====================================================================================================================
# feat-catsearch_schematic.png
# =====================================================================================================================
def _original_search(H, W, mh, mw):
    """(computed anchor -> stored cell) pairs of the original irregular-domain search, catalogNumba_irregular
    (b14a011 RainyDay_functions.py:219-235, with xlen, ylen, xloop, yloop from b14a011 RainyDay_Py3.py:1042-1051),
    in loop order."""
    xlen = W - mw
    xloop = (W - mw - 1) // 2 if (W - mw) % 2 != 0 else (W - mw) // 2
    ylen = H - mh - 2
    yloop = (H - mh - 1) // 2 if (H - mh) % 2 != 0 else (H - mh) // 2
    pairs = []
    for y in range(yloop):
        for x in range(xloop):
            pairs += [((y, x), (y, x)),
                      ((y, xlen - x), (y, xlen - x - 1)),
                      ((ylen - y - 1, x), (ylen - y - 1, x)),
                      ((ylen - y, xlen - x), (ylen - y - 1, xlen - x - 1))]
    return pairs


def _classify(H, W, mh, mw):
    """0 never searched, 1 stored in place, 2 stored one cell away, 3 searched but overwritten."""
    na_y, na_x = H - mh + 1, W - mw + 1
    cls = np.zeros((na_y, na_x), dtype=int)
    final = {}
    for comp, store in _original_search(H, W, mh, mw):
        cls[comp] = 3
        final[store] = comp                           # last write wins, as in the kernel
    for store, comp in final.items():
        cls[comp] = 1 if store == comp else 2
    return cls, final


def fig_catsearch(out=os.path.join(HERE, "feat-catsearch_schematic.png")):
    """The original irregular-domain search replayed on a 12 x 17 array with a one-cell basin. Also prints the counts
    of the same replay on the Iowa domain (152 x 233): 732 of 35,416 anchors never searched, 116 overwritten, half of
    the stored sums one cell away."""
    grey = "#d6d6d6"
    big, big_final = _classify(152, 233, 1, 1)
    print("Iowa 152 x 233, 1x1 basin: anchors", big.size, "never searched", int((big == 0).sum()),
          "overwritten", int((big == 3).sum()),
          "| stored sums one cell away:", sum(1 for s, c in big_final.items() if s != c), "of", len(big_final))

    H, W = 12, 17
    cls, final = _classify(H, W, 1, 1)
    colors = {0: RED, 1: grey, 2: matplotlib.colors.to_rgba(BLUE, 0.45), 3: "white"}
    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    for y in range(cls.shape[0]):
        for x in range(cls.shape[1]):
            c = cls[y, x]
            ax.add_patch(Rectangle((x, y), 1, 1, facecolor=colors[c] if c else "white",
                                   edgecolor="white", lw=1.5, hatch=None))
            if c == 3:
                ax.add_patch(Rectangle((x + 0.06, y + 0.06), 0.88, 0.88, facecolor="white", edgecolor=RED, lw=1.4))
            if c == 0:
                ax.add_patch(Rectangle((x + 0.06, y + 0.06), 0.88, 0.88,
                                       facecolor=matplotlib.colors.to_rgba(RED, 0.20), edgecolor=RED, lw=1.2,
                                       hatch="xx"))
    for store, comp in final.items():
        if store != comp:
            (cy, cx), (sy, sx) = comp, store
            ax.annotate("", xy=(sx + 0.5, sy + 0.5), xytext=(cx + 0.5, cy + 0.5),
                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9, mutation_scale=8, shrinkA=2, shrinkB=2))
    ax.set_xlim(-0.2, W + 0.2)
    ax.set_ylim(H + 0.2, -0.2)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Original irregular-domain catalog search (one-cell basin, 12 x 17 anchors)", fontsize=11.5,
                 color=INK, loc="left")
    handles = [Patch(facecolor=grey, label="searched; sum stored at its own cell"),
               Patch(facecolor=matplotlib.colors.to_rgba(BLUE, 0.45),
                     label="searched; sum stored one cell away (arrow: computed -> stored)"),
               Patch(facecolor="white", edgecolor=RED, lw=1.4, label="searched; sum overwritten by a neighbour's"),
               Patch(facecolor=matplotlib.colors.to_rgba(RED, 0.20), edgecolor=RED, hatch="xx",
                     label="never searched")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=1, frameon=False, fontsize=10)
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("wrote", out, "| small grid: never searched", int((cls == 0).sum()), "overwritten", int((cls == 3).sum()),
          "of", cls.size)


# =====================================================================================================================
# feat-strict_schematic.png
# =====================================================================================================================
def fig_strict(out=os.path.join(HERE, "feat-strict_schematic.png")):
    """A synthetic irregular domain (a bay cut into its east side), a basin box of 9 x 12 cells and a storm core on
    the bay edge. (a) The original rule (b14a011 RainyDay_Py3.py:1954-1959): an anchor (the box's upper-left cell) is
    valid if it is a domain cell outside the last maskheight rows and maskwidth columns of the array. (b) The strict
    rule: valid only if the whole box lies inside the domain. Each box is drawn at the valid anchor that catches the
    most storm rain, counting rain outside the domain, as transposition does. Synthetic geometry; no real data."""
    H, W = 36, 48            # array = bounding rectangle of the domain (rows down, columns right)
    MH, MW = 9, 12           # basin bounding box (maskheight x maskwidth)
    poly = np.array([(0, 6), (8, 0), (38, 0), (48, 8), (48, 13), (38, 13), (33, 18), (37, 24), (48, 25),
                     (44, 36), (6, 36), (0, 28)], dtype=float)
    yy, xx = np.mgrid[0:H, 0:W]
    centres = np.column_stack([xx.ravel() + 0.5, yy.ravel() + 0.5])
    domain = Path(poly).contains_points(centres).reshape(H, W)

    # storm core: a smooth blob centred on the bay edge
    core_x, core_y, sx, sy = 36.0, 18.5, 5.0, 3.5
    rain = np.exp(-0.5 * (((xx + 0.5 - core_x) / sx) ** 2 + ((yy + 0.5 - core_y) / sy) ** 2))

    # valid anchors under each rule
    orig = domain.copy()
    orig[-MH:, :] = False
    orig[:, -MW:] = False
    strict = np.zeros((H, W), dtype=bool)
    for y in range(H - MH + 1):
        for x in range(W - MW + 1):
            strict[y, x] = domain[y:y + MH, x:x + MW].all()

    def best_anchor(valid):
        best, where = -1.0, None
        for y, x in zip(*np.nonzero(valid)):
            if y + MH > H or x + MW > W:
                continue
            s = rain[y:y + MH, x:x + MW].sum()
            if s > best:
                best, where = s, (y, x)
        return where

    def draw(ax, valid, title, anchor, outside_hatch):
        img = np.ones((H, W, 4))
        img[domain] = matplotlib.colors.to_rgba("#e3e3e3")
        img[valid] = matplotlib.colors.to_rgba(BLUE, 0.38)
        ax.imshow(img, extent=(0, W, H, 0), interpolation="nearest")
        ax.add_patch(Ellipse((core_x, core_y), 2 * 1.35 * sx, 2 * 1.35 * sy, facecolor="#9e9e9e", alpha=0.35,
                             edgecolor="#616161", lw=1.2, ls="--", zorder=2))
        ax.add_patch(Polygon(poly, closed=True, fill=False, edgecolor=INK, lw=1.8, zorder=3))
        y, x = anchor
        if outside_hatch:
            for cy in range(y, y + MH):
                for cx in range(x, x + MW):
                    if not domain[cy, cx]:
                        ax.add_patch(Rectangle((cx, cy), 1, 1, facecolor="none", edgecolor=RED, hatch="////",
                                               lw=0, zorder=4))
        ax.add_patch(Rectangle((x, y), MW, MH, fill=False, edgecolor=INK, lw=2.2, zorder=5))
        ax.plot(x + 0.5, y + 0.5, "o", ms=8, mfc="white", mec=INK, mew=2, zorder=6)
        ax.annotate("anchor", (x + 0.5, y + 0.5), xytext=(x - 9, y - 5.5), fontsize=10, color=INK,
                    arrowprops=dict(arrowstyle="-", color=INK, lw=0.9), zorder=7)
        ax.text(46.3, 19.0, "outside the domain", fontsize=9, color="#616161", ha="center", va="center", rotation=90)
        ax.set_xlim(-0.5, W + 0.5)
        ax.set_ylim(H + 0.5, -0.5)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, fontsize=11.5, color=INK, loc="left")

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0))
    draw(axes[0], orig, "(a) Original: only the anchor cell must be in the domain", best_anchor(orig), True)
    draw(axes[1], strict, "(b) Strict: the whole box must be in the domain", best_anchor(strict), False)

    handles = [Patch(facecolor="#e3e3e3", edgecolor=INK, label="transposition domain"),
               Patch(facecolor=matplotlib.colors.to_rgba(BLUE, 0.38),
                     label="valid anchors (upper-left cell of the box)"),
               Patch(facecolor="#9e9e9e", alpha=0.35, edgecolor="#616161", ls="--", label="storm core"),
               Line2D([], [], color=INK, lw=2.2, label="basin box (maskheight x maskwidth)"),
               Patch(facecolor="white", edgecolor=RED, hatch="////", label="part of the box outside the domain")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9.5)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    print("wrote", out, "| valid anchors: original", int(orig.sum()), "strict", int(strict.sum()),
          "| domain cells", int(domain.sum()))


# =====================================================================================================================
# feat-bands_schematic.png
# =====================================================================================================================
def _ndtri(p):
    """Standard normal quantile (Acklam's rational approximation, |error| < 1.2e-9); numpy only."""
    p = np.asarray(p, dtype=float)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    q = np.where(p < 0.5, p, 1 - p)
    out = np.empty_like(q)
    lo = q < 0.02425
    t = np.sqrt(-2 * np.log(q[lo]))
    out[lo] = (((((c[0] * t + c[1]) * t + c[2]) * t + c[3]) * t + c[4]) * t + c[5]) / \
        ((((d[0] * t + d[1]) * t + d[2]) * t + d[3]) * t + 1)
    u = q[~lo] - 0.5
    s = u * u
    out[~lo] = (((((a[0] * s + a[1]) * s + a[2]) * s + a[3]) * s + a[4]) * s + a[5]) * u / \
        (((((b[0] * s + b[1]) * s + b[2]) * s + b[3]) * s + b[4]) * s + 1)
    return np.where(p < 0.5, out, -out)


def _expected_range(n):
    """E[max - min] of n independent standard normal draws (numerical integral)."""
    x = np.linspace(-8, 8, 16001)
    phi = np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)
    Phi = np.cumsum(phi) * (x[1] - x[0])
    trap = getattr(np, "trapezoid", None) or np.trapz        # numpy 2 renamed trapz
    return 2 * trap(x * n * phi * Phi ** (n - 1), x)


def fig_bands(out=os.path.join(HERE, "feat-bands_schematic.png"), data=DATA):
    """(a) The five bands of Case Study 1, Iowa pixel, 24 h, T = 100 yr (FREQENGINE "both" + RECORDCI, B = 1000).
    (b) The width of each band against NREALIZATIONS N: the record parts are the run's own bootstrap values; the Monte
    Carlo parts use the standard deviation of the run's 100 realizations with a normal approximation (expected range
    of N normal draws; standard error sd/sqrt(N)); est is combined as RainyDay_fast does it, over every pair of the
    1,000 bootstrap records and 1,000 equally probable normal deviates."""
    run = os.path.join(data, "case1_iowa_pixel", "Pixel_r1_A1km2_rot0_24", "Pixel_r1_A1km2_rot0_24")
    T = 100.
    fa = pd.read_csv(run + "_FreqAnalysis.csv", skipinitialspace=True)
    rc = pd.read_csv(run + "_RecordCI.csv", comment="#")
    bt = pd.read_csv(run + "_RecordBootstrap.csv")
    i = int(np.argmin(np.abs(rc["returnperiod"].values - T)))
    rlz = fa.filter(regex=r"^rlz\d+$").values[i]
    r = rc.iloc[i]
    N0 = rlz.size
    sd = rlz.std(ddof=1)
    z90 = 1.6448536269514722                                   # 95th percentile of the standard normal
    dev = bt.filter(regex=r"^rec\d+$").values[i] - bt["sst_expected"].values[i]   # record deviations
    zk = _ndtri((np.arange(1000) + 0.5) / 1000)               # equally probable normal deviates (as RainyDay_fast)

    est_mean = r["rd_mean"]
    se0 = sd / np.sqrt(N0)
    bands = [  # label, lo, hi, colour, style
        ("Ensemble spread\n(min–max of 100 realizations)", r["ens_lo"], r["ens_hi"], GREY, "-"),
        ("Monte Carlo error of the mean\n(90 %, ±1.645 SE)", est_mean - z90 * se0, est_mean + z90 * se0, BLACK, "-"),
        ("Record CI (90 %)", r["rec_lo"], r["rec_hi"], ORANGE, "-"),
        ("est = record ⊕ Monte Carlo\n(90 % CI of the estimate)", r["est_lo"], r["est_hi"], ORANGE, "--"),
        ("pred = record ⊕ ensemble\n(90 %, NOT a CI)", r["pred_lo"], r["pred_hi"], GREY, "--"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw=dict(width_ratios=[1.15, 1]))
    for k, (lab, lo, hi, col, ls) in enumerate(bands):
        y = len(bands) - k
        ax1.plot([lo, hi], [y, y], color=col, lw=6 if ls == "-" else 2.2, ls=ls, solid_capstyle="butt")
        ax1.text(hi + 1.2, y, "%.0f–%.0f mm" % (lo, hi), va="center", fontsize=8, color=BLACK)
    ax1.axvline(est_mean, color=BLACK, lw=1, ls=":", label="Monte Carlo mean")
    ax1.axvline(r["sst_expected"], color=BLUE, lw=1, ls=":", label="Exact mean")
    ax1.legend(fontsize=7.5, frameon=False, loc="center right")
    ax1.set_yticks(range(len(bands), 0, -1))
    ax1.set_yticklabels([b[0] for b in bands], fontsize=8)
    ax1.set_xlim(140, 214)
    ax1.set_ylim(0.4, len(bands) + 0.6)
    ax1.set_xlabel("24 h depth P at T = 100 yr (mm)")
    ax1.set_title("(a) Iowa pixel, 24 h, T = 100 yr", fontsize=9, loc="left")

    Ns = np.array([2, 3, 5, 10, 20, 50, 100, 200, 500, 1000])
    pct = 100. / r["sst_expected"]
    ens_w = np.array([_expected_range(n) for n in Ns]) * sd * pct
    mc_w = 2 * z90 * sd / np.sqrt(Ns) * pct
    rec_w = (r["rec_hi"] - r["rec_lo"]) * pct
    est_w = np.array([np.diff(np.percentile((dev[:, None] + sd / np.sqrt(n) * zk[None, :]).ravel(), [5, 95]))[0]
                      for n in Ns]) * pct
    pred_w = (r["pred_hi"] - r["pred_lo"]) * pct
    ax2.plot(Ns, ens_w, color=GREY, lw=2, marker="o", ms=4, label="Ensemble spread, min–max (grows with N)")
    ax2.plot(Ns, mc_w, color=BLACK, lw=1.8, marker="o", ms=4, label="Monte Carlo error of the mean, 90 % (shrinks)")
    ax2.plot(Ns, est_w, color=ORANGE, lw=1.8, ls="--", label="est: record ⊕ Monte Carlo, 90 %")
    ax2.axhline(rec_w, color=ORANGE, lw=1.8, label="Record CI, 90 % (needs more years, not more N)")
    ax2.axhline(pred_w, color=GREY, lw=1.5, ls="--", label="pred: record ⊕ one realization, 90 % (not a CI)")
    ax2.axvline(N0, color=BLACK, lw=0.8, ls=":")
    ax2.text(N0 * 1.1, 15, "this run\n(N = 100)", fontsize=8, color=BLACK)
    ax2.set_xscale("log")
    ax2.set_xlabel("NREALIZATIONS N")
    ax2.set_ylabel("Width of the band (% of the estimate)")
    ax2.set_ylim(0, 60)
    ax2.set_xticks([2, 10, 100, 1000])
    ax2.set_xticklabels(["2", "10", "100", "1000"])
    ax2.set_title("(b) Band width vs NREALIZATIONS", fontsize=9, loc="left")
    ax2.legend(fontsize=7.5, frameon=False, loc="upper left")
    for ax in (ax1, ax2):
        ax.grid(color="#dddddd", lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)
    print("N = %d: sd %.2f mm, SE %.2f mm; widths at N = 100: ens %.1f %%, mc %.1f %%, rec %.1f %%, est %.1f %%, "
          "pred %.1f %%" % (N0, sd, se0, ens_w[Ns == 100][0], mc_w[Ns == 100][0], rec_w, est_w[Ns == 100][0], pred_w))


FIGURES = {"steps": fig_steps, "catsearch": fig_catsearch, "strict": fig_strict, "bands": fig_bands}

if __name__ == "__main__":
    names = sys.argv[1:] or list(FIGURES)
    for name in names:
        if name not in FIGURES:
            sys.exit("unknown figure %r; choose from %s" % (name, ", ".join(FIGURES)))
        FIGURES[name]()
