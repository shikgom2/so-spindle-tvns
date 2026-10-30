# -*- coding: utf-8 -*-
"""Figures 1 and 2 — design / primary result, and prevalence in detail."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
import figstyle as S

S.apply_style()
PAPER = Path(__file__).resolve().parent.parent / "data" / "paper"
ROIS = ["Frontal", "Central", "Parietal", "Temporal"]
SHORT = {"Frontal": "Frontal", "Central": "Central",
         "Parietal": "Parietal", "Temporal": "Temporal"}

DIVERGING = LinearSegmentedColormap.from_list(
    "ctrl_tvns", [S.CTRL, "#E9EDF1", "#F5F0EC", S.TVNS])


def ci_from_dz(mean_diff, dz, n):
    """95% CI of a paired mean difference, recovered from dz (= diff / SD_diff)."""
    if dz == 0 or not np.isfinite(dz):
        return np.nan, np.nan
    sd = mean_diff / dz
    se = abs(sd) / np.sqrt(n)
    return mean_diff - 1.96 * se, mean_diff + 1.96 * se


# ==========================================================  FIGURE 1
def figure1():
    t1 = pd.read_csv(PAPER / "T1_paired_prevalence_by_window.csv")
    f3 = pd.read_csv(PAPER / "full" / "F03_within_stimulation_sessions.csv")

    fig = plt.figure(figsize=(S.W2, 4.15))
    gs = fig.add_gridspec(2, 12, hspace=0.46, wspace=3.0,
                          left=.065, right=.985, top=.93, bottom=.105)

    # ---------------------------------------------- a  design
    ax = fig.add_subplot(gs[0, 0:6])
    S.panel(ax, "a", dx=-.055, dy=1.12)
    ax.set_xlim(-2.0, 44.0)
    ax.set_ylim(-0.95, 2.35)
    ax.axis("off")
    ax.text(21, 2.30, "55 participants  \u00b7  162 sessions  \u00b7  "
                      "119 stimulation / 43 no current  \u00b7  42 with both",
            fontsize=6.3, color=S.MUTED, ha="center", va="top")
    rows = [("Visits 1\u20133", "stimulation", 1.30, S.TVNS, True),
            ("Visit 4", "no current", 0.25, S.CTRL, False)]
    for vlab, midlab, y, col, on in rows:
        ax.text(-1.8, y + 0.55, vlab, fontsize=6.8, color=col,
                va="center", fontweight="bold")
        for x0, w, nm in [(0, 10, "pre"), (10, 20, midlab), (30, 10, "post")]:
            active = on and w == 20
            ax.add_patch(FancyBboxPatch(
                (x0 + 0.3, y), w - 0.6, 0.40,
                boxstyle="round,pad=0,rounding_size=0.10",
                linewidth=0.8, edgecolor=col,
                facecolor=col if active else "white"))
            _t = ax.text(x0 + w / 2, y + 0.20, nm, ha="center", va="center",
                         fontsize=6.4, color="white" if active else S.MUTED)
            if active:
                _t.set_gid("ink-ok")       # white label inside its filled block
        ax.text(41.0, y + 0.20, "40 min", fontsize=6, color=S.MUTED, va="center")
    for x in (0, 10, 30, 40):
        ax.plot([x, x], [-0.30, -0.20], lw=.6, color=S.MUTED)
        ax.text(x, -0.38, f"{x}", fontsize=5.8, color=S.MUTED,
                ha="center", va="top")
    ax.plot([0, 40], [-0.25, -0.25], lw=.6, color=S.MUTED)
    ax.text(20, -0.72, "time from recording onset (min)", fontsize=6.3,
            color=S.MUTED, ha="center", va="top")

    # ---------------------------------------------- b  coupling rule
    ax = fig.add_subplot(gs[0, 6:12])
    S.panel(ax, "b", dx=-.055, dy=1.12)
    t = np.linspace(-1.15, 1.15, 1400)
    so = -np.cos(2 * np.pi * 0.9 * t)
    env = np.exp(-((t - 0.40) ** 2) / (2 * 0.085 ** 2))
    spin = 0.40 * env * np.sin(2 * np.pi * 14 * t)
    ax.axvspan(0.2778, 0.8333, color=S.HILITE, zorder=0, lw=0, ymax=.955)
    SP_Y = 2.60                      # spindle trace raised to open a clear band
    ax.plot(t, so, color=S.INK, lw=1.4, zorder=3)
    ax.plot(t, spin + SP_Y, color=S.TVNS, lw=0.7, zorder=3)
    ax.plot(t, 0.40 * env + SP_Y, color=S.TVNS, lw=1.1,
            ls=(0, (2.4, 1.6)), zorder=4)
    ax.plot([0.5556], [1.0], marker="o", ms=3.6, mfc="white",
            mec=S.INK, mew=0.9, zorder=5)
    ax.plot([0.40, 0.40], [SP_Y - 0.42, 1.04], lw=.6, color=S.TVNS,
            ls=(0, (1.4, 1.4)), zorder=2)
    # both labels sit on the pale \u00b190\u00b0 wash by design, not on data
    ax.text(0.5556, 3.72, "\u00b190\u00b0 coupling window", fontsize=6.2,
            color=S.TVNS, ha="center", va="bottom").set_gid("ink-ok")
    # leader drawn explicitly: an Annotation's extent would include the arrow
    ax.plot([0.5556, 0.5556], [1.06, 1.26], lw=.5, color=S.MUTED, zorder=2)
    ax.text(0.5556, 1.76, "up-state peak (0\u00b0)", fontsize=6, color=S.INK,
            ha="center", va="top").set_gid("ink-ok")
    ax.text(-1.30, 3.26, "spindle, 12\u201316 Hz", fontsize=6.1, color=S.TVNS,
            va="bottom", ha="left")
    ax.text(-1.30, -1.62, "slow oscillation, 0.16\u20131.25 Hz", fontsize=6.1,
            color=S.INK, va="top", ha="left")
    ax.set_ylim(-3.10, 4.25)
    ax.set_xlim(-1.32, 1.34)
    ax.axis("off")
    ax.plot([0.80, 1.30], [-2.52, -2.52], lw=1.0, color=S.INK)
    ax.text(1.05, -2.62, "0.5 s", fontsize=5.8, color=S.MUTED,
            ha="center", va="top")

    # ---------------------------------------------- c  trajectory, parietal
    ax = fig.add_subplot(gs[1, 0:5])
    S.panel(ax, "c", dx=-.155)
    wins = ["pre", "stim", "post"]
    sub = t1[t1.ROI == "Parietal"].set_index("win").loc[wins]
    x = np.arange(3)
    for col, key, lab, dy in [(S.TVNS, "mean_t", "tVNS", -.55),
                              (S.CTRL, "mean_s", "No current", -.50)]:
        y = sub[key].values * 100
        ax.plot(x, y, color=col, lw=1.6, marker="o", ms=4.6, zorder=3,
                mfc="white", mec=col, mew=1.4)
        ax.text(2.16, y[-1] + dy, lab, color=col, fontsize=6.6, va="center")
    for i, w in enumerate(wins):
        r = sub.loc[w]
        lo, hi = ci_from_dz(r.mean_diff * 100, r.dz, r.n)
        top = max(r.mean_t, r.mean_s) * 100
        txt = (f"{r.mean_diff*100:+.2f} pp\n[{lo:+.1f}, {hi:+.1f}]\n{S.pfmt(r.p_wilcoxon)}"
               if w == "stim" else f"{r.mean_diff*100:+.2f} pp")
        ax.text(i, top + 0.55, txt, ha="center", va="bottom", fontsize=5.9,
                color=S.INK if r.p_wilcoxon < .05 else S.MUTED, linespacing=1.35)
    ax.set_xticks(x)
    ax.set_xticklabels(["pre", "stimulation", "post"])
    ax.set_ylabel("Parietal coupling prevalence (%)")
    ax.set_xlim(-0.35, 3.25)
    ax.set_ylim(9.0, 17.6)
    ax.text(-0.35, 17.58, "difference, tVNS \u2212 no current", fontsize=5.9,
            color=S.MUTED, va="top", style="italic")

    # ---------------------------------------------- d  primary forest
    ax = fig.add_subplot(gs[1, 6:12])
    S.panel(ax, "d", dx=-.20)
    rows = []
    for roi in ["Parietal", "Frontal", "Central", "Temporal"]:
        r = f3[(f3.ROI == roi) & (f3.contrast == "stim - pre")].iloc[0]
        rows.append((roi, r.mean_pp, np.nan, np.nan, r.p, int(r.n)))
    for roi in ["Parietal", "Central"]:
        r = t1[(t1.ROI == roi) & (t1.win == "stim")].iloc[0]
        lo, hi = ci_from_dz(r.mean_diff * 100, r.dz, r.n)
        rows.append((roi, r.mean_diff * 100, lo, hi, r.p_wilcoxon, int(r.n)))
    ys = np.array([6, 5, 4, 3, 1, 0])      # blank row at y = 2 for the caption
    for y, (lab, est, lo, hi, pv, n) in zip(ys, rows):
        sig = pv < .05
        col = S.TVNS if sig else S.MUTED
        if np.isfinite(lo):
            ax.plot([lo, hi], [y, y], lw=1.1, color=col, alpha=.8, zorder=2,
                    solid_capstyle="round")
        else:
            ax.plot([0, est], [y, y], lw=1.0, color=col, alpha=.35, zorder=2,
                    solid_capstyle="butt")
        ax.plot([est], [y], "o", ms=5.2 if sig else 4.0, color=col,
                mec="white", mew=.8, zorder=3)
        ax.text(10.2, y, f"{est:+.2f}", fontsize=6.1, va="center", ha="right",
                color=S.INK if sig else S.MUTED)
        ax.text(14.0, y, S.pfmt(pv), fontsize=6.1, va="center", ha="right",
                color=S.INK if sig else S.MUTED)
        ax.text(16.6, y, f"{n}", fontsize=6.1, va="center", ha="right",
                color=S.MUTED)
    ax.axvline(0, color=S.INK, lw=.6, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("Change in coupling prevalence (percentage points)")
    ax.set_xlim(-1.2, 16.8)
    ax.set_ylim(-0.85, 7.85)
    ax.spines["bottom"].set_bounds(-1.0, 8.0)
    ax.set_xticks([0, 2, 4, 6, 8])
    for lbl, yy in [("within stimulated sessions, stim \u2212 pre", 6.72),
                    ("tVNS vs own no-current session", 1.90)]:
        ax.text(0.35, yy, lbl, fontsize=6, color=S.MUTED, va="bottom",
                style="italic")
    for xx, lbl in [(10.2, "pp"), (14.0, "P"), (16.6, "n")]:
        ax.text(xx, 7.55, lbl, fontsize=6, color=S.MUTED,
                ha="right", va="bottom")
    S.save(fig, "Figure_1")


# ==========================================================  FIGURE 2
_RAW = {"Fp1": (-.30, .88), "Fp2": (.30, .88),
        "F7": (-.76, .44), "F3": (-.36, .47), "F4": (.36, .47), "F8": (.76, .44),
        "T3": (-.94, .00), "C3": (-.45, .00), "Cz": (0, 0), "C4": (.45, .00),
        "T4": (.94, .00),
        "P3": (-.36, -.47), "P4": (.36, -.47)}
_SC = 1.62                       # spread the montage so labels never touch markers
CHPOS = {k: (x * _SC, y * _SC) for k, (x, y) in _RAW.items()}
LBL_UP = {"Fp1", "Fp2", "F7", "F8"}   # F3/F4 label below, clear of Fp1/Fp2


def figure2():
    f3 = pd.read_csv(PAPER / "full" / "F03_within_stimulation_sessions.csv")
    t1 = pd.read_csv(PAPER / "T1_paired_prevalence_by_window.csv")
    t5 = pd.read_csv(PAPER / "T5_channel_paired_stimulation.csv")
    f1 = pd.read_csv(PAPER / "full" / "F01_detector_grid.csv")

    fig = plt.figure(figsize=(S.W2, 4.45))
    gs = fig.add_gridspec(2, 13, hspace=0.52, wspace=3.2,
                          left=.065, right=.985, top=.90, bottom=.11)

    # ---- a  stim-pre by ROI
    ax = fig.add_subplot(gs[0, 0:4])
    S.panel(ax, "a", dx=-.21)
    sub = f3[f3.contrast == "stim - pre"].set_index("ROI").loc[ROIS]
    x = np.arange(4)
    cols = [S.TVNS if pv < .05 else S.FAINT for pv in sub.p]
    ax.bar(x, sub.mean_pp, width=.60, color=cols, edgecolor="none", zorder=2)
    for xi, (v, pv) in enumerate(zip(sub.mean_pp, sub.p)):
        ax.text(xi, v + .16, S.stars(pv) if pv < .05 else "n.s.", ha="center",
                fontsize=6.4, color=S.INK if pv < .05 else S.MUTED)
    ax.axhline(0, color=S.INK, lw=.6)
    ax.set_xticks(x)
    ax.set_xticklabels([r[:4] for r in ROIS])
    ax.set_ylabel("stim \u2212 pre (pp)")
    ax.set_ylim(0, 5.5)
    ax.set_title("Within stimulated sessions", fontsize=6.5,
                 color=S.MUTED, pad=4)

    # ---- b  paired, stim window
    ax = fig.add_subplot(gs[0, 4:8])
    S.panel(ax, "b", dx=-.20)
    sub = t1[t1.win == "stim"].set_index("ROI").loc[ROIS]
    for i, roi in enumerate(ROIS):
        r = sub.loc[roi]
        sig = r.p_wilcoxon < .05
        ax.plot([i - .17, i + .17], [r.mean_s * 100, r.mean_t * 100],
                color=S.INK if sig else S.FAINT, lw=1.0, zorder=2)
        ax.plot(i - .17, r.mean_s * 100, "o", ms=4.6, color=S.CTRL,
                mec="white", mew=.7, zorder=3)
        ax.plot(i + .17, r.mean_t * 100, "o", ms=4.6, color=S.TVNS,
                mec="white", mew=.7, zorder=3)
        if sig:
            ax.text(i, max(r.mean_s, r.mean_t) * 100 + .50, S.stars(r.p_wilcoxon),
                    ha="center", fontsize=6.4)
    ax.set_xticks(range(4))
    ax.set_xticklabels([r[:4] for r in ROIS])
    ax.set_ylabel("Coupling prevalence (%)")
    ax.set_ylim(8.8, 16.2)
    ax.set_xlim(-.5, 3.5)
    ax.plot([], [], "o", color=S.CTRL, label="No current")
    ax.plot([], [], "o", color=S.TVNS, label="tVNS")
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.00), ncol=2,
              handletextpad=.3, columnspacing=1.2)

    # ---- c  channel map
    ax = fig.add_subplot(gs[0, 8:13])
    S.panel(ax, "c", dx=-.08)
    ax.add_patch(plt.Circle((0, 0), 2.42, fill=False, lw=.8, color=S.MUTED))
    ax.plot([-.24, 0, .24], [2.38, 2.70, 2.38], lw=.8, color=S.MUTED,
            solid_joinstyle="miter")
    vmax = float(np.abs(t5.dz).max())
    for _, r in t5.iterrows():
        xx, yy = CHPOS[r.ch]
        f = abs(r.dz) / vmax
        col = S.TVNS if r.dz > 0 else S.CTRL
        ax.plot(xx, yy, "o", ms=2.6 + 7.4 * f, color=col,
                alpha=.25 + .60 * f, mec="white", mew=.6, zorder=3)
        if r.p_wilcoxon < .05:
            st_ = ax.text(xx, yy, "*", ha="center", va="center", fontsize=8,
                          color="white", zorder=4, fontweight="bold")
            st_.set_gid("nocheck")        # deliberate: glyph sits inside its marker
        # label pushed radially outward, away from the crowded centre
        rad = np.hypot(xx, yy)
        if rad < 1e-6:                       # Cz sits at the origin
            lx, ly, va = xx, yy - 0.34, "top"
        else:
            lx, ly = xx + 0.42 * xx / rad, yy + 0.42 * yy / rad
            va = "bottom" if ly >= yy else "top"
        ax.text(lx, ly, r.ch, ha="center", va=va, fontsize=5.5,
                color=S.MUTED, zorder=4)
    ax.set_xlim(-2.95, 2.95)
    ax.set_ylim(-4.25, 2.95)
    ax.set_aspect("equal")
    ax.axis("off")
    for i, (sz, lab) in enumerate([(0.15, "0.15"), (0.40, "0.40")]):
        ax.plot(-1.85 + i * 1.15, -3.25, "o", ms=2.6 + 7.4 * (sz / vmax),
                color=S.TVNS, alpha=.25 + .60 * (sz / vmax),
                mec="white", mew=.6)
        ax.text(-1.85 + i * 1.15, -3.62, lab, fontsize=5.4,
                color=S.MUTED, ha="center", va="top")
    ax.text(0.55, -3.30, "paired d$_z$\n* P < 0.05", fontsize=5.6,
            color=S.MUTED, ha="left", va="center")

    # ---- d  detector x staging matrix
    ax = fig.add_subplot(gs[1, 0:6])
    S.panel(ax, "d", dx=-.14)
    dets = ["jy_N2", "jy_N2N3", "staresina_N2", "staresina_N2N3"]
    dlab = ["Primary\nN2", "Primary\nN2+N3", "Alternative\nN2", "Alternative\nN2+N3"]
    M = np.zeros((4, 4))
    Q = np.zeros((4, 4))
    for i, det in enumerate(dets):
        for j, roi in enumerate(ROIS):
            r = f1[(f1.ROI == roi) & (f1.detector == det)].iloc[0]
            M[i, j] = r.dz
            Q[i, j] = r.p_wilcoxon
    norm = TwoSlopeNorm(vmin=-0.45, vcenter=0, vmax=0.45)
    ax.imshow(M, cmap=DIVERGING, norm=norm, aspect="auto")
    for i in range(4):
        for j in range(4):
            sig = Q[i, j] < .05
            ax.text(j, i, f"{M[i, j]:+.2f}" + ("*" if sig else ""),
                    ha="center", va="center", fontsize=6.1,
                    color="white" if abs(M[i, j]) > 0.30 else S.INK,
                    fontweight="bold" if sig else "normal").set_gid("ink-ok")
    ax.set_xticks(range(4))
    ax.set_xticklabels([r[:4] for r in ROIS])
    ax.set_yticks(range(4))
    ax.set_yticklabels(dlab, fontsize=5.9, linespacing=1.2)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Paired d$_z$ across detector \u00d7 staging  (* q < 0.05)",
                 fontsize=6.5, color=S.MUTED, pad=4)

    # ---- e  three windows
    gse = gs[1, 6:13].subgridspec(1, 3, wspace=.28)
    order = ["stim - pre", "post - pre", "post - stim"]
    lab = ["stim \u2212 pre", "post \u2212 pre", "post \u2212 stim"]
    for j, (c, lb) in enumerate(zip(order, lab)):
        ax = fig.add_subplot(gse[0, j])
        if j == 0:
            S.panel(ax, "e", dx=-.42)
            ax.set_ylabel("Change in prevalence (pp)")
        sub = f3[f3.contrast == c].set_index("ROI").loc[ROIS]
        for k, roi in enumerate(ROIS):
            r = sub.loc[roi]
            sig = r.p < .05
            col = S.TVNS if r.mean_pp > 0 else S.CTRL
            ax.plot([k, k], [0, r.mean_pp], lw=1.2, color=col, alpha=.40)
            ax.plot(k, r.mean_pp, "o", ms=5.0 if sig else 3.4, color=col,
                    mec="white", mew=.7, zorder=3)
            if sig:
                ax.text(k, r.mean_pp + .30, S.stars(r.p), ha="center",
                        fontsize=6.3)
        ax.axhline(0, color=S.INK, lw=.6)
        ax.set_xticks(range(4))
        ax.set_xticklabels([r[0] for r in ROIS], fontsize=6)
        ax.set_ylim(-2.6, 6.0)
        ax.set_xlim(-.6, 3.6)
        ax.set_title(lb, fontsize=6.4, color=S.INK, pad=3)
        if j > 0:
            ax.set_yticklabels([])
            ax.spines["left"].set_visible(False)
            ax.tick_params(left=False)
    fig.text(.985, .125, "F, Frontal \u00b7 C, Central \u00b7 P, Parietal \u00b7 "
                         "T, Temporal", fontsize=5.6, color=S.MUTED, ha="right")
    S.save(fig, "Figure_2")


if __name__ == "__main__":
    print("building figures 1-2")
    figure1()
    figure2()
