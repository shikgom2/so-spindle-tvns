# -*- coding: utf-8 -*-
"""Figures 3-6 — persistence & envelope, strength & phase, event properties,
stratification & autonomic state."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import figstyle as S

S.apply_style()
ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "data" / "paper"
P1 = ROOT / "data" / "vns" / "01_PROJECT1_SO_SPINDLE_EFFECT" / "outputs" / "shared"
STATS = ROOT / "analysis" / "outputs"    # written by analysis/s05_reviewer_models.py
ROIS = ["Frontal", "Central", "Parietal", "Temporal"]


def arm_of(sid):
    return "tVNS" if str(sid).startswith("T") else "No current"


def ci_from_dz(mean_diff, dz, n):
    if dz == 0 or not np.isfinite(dz):
        return np.nan, np.nan
    sd = mean_diff / dz
    se = abs(sd) / np.sqrt(n)
    return mean_diff - 1.96 * se, mean_diff + 1.96 * se


# ==========================================================  FIGURE 3
def figure3():
    z = np.load(PAPER / "TB_so_locked_env.npz", allow_pickle=True)
    t = z["t"]
    recs = {}
    for k in z.files:
        if k == "t":
            continue
        sid, win, roi = k.split("|")
        recs.setdefault((arm_of(sid), win, roi), []).append(z[k])

    f3 = pd.read_csv(PAPER / "full" / "F03_within_stimulation_sessions.csv")

    fig = plt.figure(figsize=(S.W2, 3.3))
    axA, axB, axC = [fig.add_subplot(1, 3, i) for i in (1, 2, 3)]
    fig.subplots_adjust(left=.075, right=.985, top=.855, bottom=.165,
                        wspace=.60)

    WINCOL = {"pre": "#9AA3AD", "stim": S.TVNS, "post": "#E3A08A"}

    # ---- a  SO-locked sigma envelope inside stimulated sessions
    ax = axA
    S.panel(ax, "a", dx=-.20)
    for win in ["pre", "stim", "post"]:
        arrs = recs.get(("tVNS", win, "Parietal"), [])
        if not arrs:
            continue
        A = np.vstack(arrs)
        m = np.nanmean(A, 0)
        sem = np.nanstd(A, 0) / np.sqrt(np.sum(~np.isnan(A[:, 0])))
        ax.fill_between(t, m - sem, m + sem, color=WINCOL[win],
                        alpha=.18, lw=0, zorder=2)
        ax.plot(t, m, color=WINCOL[win], lw=1.3, zorder=3,
                label=f"{win} (n={len(arrs)})")
    ax.axvline(0, color=S.MUTED, lw=.6, ls=(0, (2, 2)), zorder=1)
    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel("Time from SO trough (s)")
    ax.set_ylabel("Sigma envelope (\u03bcV)")
    ax.set_title("Stimulated sessions, parietal", fontsize=6.6,
                 color=S.MUTED, pad=4)
    ax.legend(loc="upper left", fontsize=5.8, handlelength=1.2,
              handletextpad=.4, labelspacing=.25)

    # ---- b  envelope peak across windows, stimulated sessions
    ax = axB
    S.panel(ax, "b", dx=-.22)
    sel = (t > -0.5) & (t < 0.8)
    for k, win in enumerate(["pre", "stim", "post"]):
        arrs = recs.get(("tVNS", win, "Parietal"), [])
        if not arrs:
            continue
        v = np.nanmax(np.vstack(arrs)[:, sel], axis=1)
        m = np.nanmean(v)
        se = np.nanstd(v) / np.sqrt(len(v))
        ax.plot([k, k], [m - se, m + se], color=WINCOL[win], lw=1.2, zorder=2)
        ax.plot(k, m, "o", ms=5.0, color=WINCOL[win], mec="white",
                mew=.8, zorder=3)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["pre", "stim", "post"])
    ax.set_ylabel("Envelope peak (\u03bcV)")
    ax.set_xlim(-.5, 2.5)
    ax.set_title("Mean \u00b1 s.e.m.", fontsize=6.6, color=S.MUTED, pad=4)

    # ---- c  persistence
    ax = axC
    S.panel(ax, "c", dx=-.26)
    sub = f3[f3.contrast == "post - stim"].set_index("ROI").loc[ROIS]
    ys = np.arange(4)[::-1]
    for y, roi in zip(ys, ROIS):
        r = sub.loc[roi]
        lo, hi = ci_from_dz(r.mean_pp, r.dz, r.n)
        ax.plot([lo, hi], [y, y], lw=1.1, color=S.MUTED, zorder=2)
        ax.plot(r.mean_pp, y, "o", ms=4.4, color=S.MUTED, mec="white",
                mew=.7, zorder=3)
    ax.axvline(0, color=S.INK, lw=.6)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[:4] for r in ROIS])
    ax.set_xlabel("post \u2212 stim (pp)")
    ax.set_ylim(-.7, 4.8)
    ax.set_title("all n.s. \u2014 no rebound", fontsize=6.3, color=S.MUTED, pad=4)
    S.save(fig, "Figure_3")


# ==========================================================  FIGURE 4
def figure4():
    """Coupling strength by region, the direct regional contrast, and coupling phase.

    Colour convention inside this figure: panels a and b carry no condition colour -
    they show model estimates, and significance is shown by a filled marker.  The
    condition palette (blue = no current, orange = tVNS) is used only in panel c,
    where two conditions are actually being compared.  The polar histogram used in an
    earlier version was dropped: it cannot resolve a shift of a few degrees, which is
    the size of the effect.
    """
    st = pd.read_csv(PAPER / "full" / "TB_full_summary_strength.csv")
    rc = pd.read_csv(STATS / "B_region_contrast.csv")
    ph = pd.read_csv(P1 / "P1_13_phase_all_cohorts.csv")

    fig = plt.figure(figsize=(S.W2, 3.2))
    gs = fig.add_gridspec(1, 30, wspace=3.0, left=.070, right=.795,
                          top=.84, bottom=.19)

    # ---- a  interaction beta by region, both strength metrics
    ax = fig.add_subplot(gs[0, 0:7])
    S.panel(ax, "a", dx=-.26)
    for j, (out, mk) in enumerate([("ppc", "o"), ("mrl", "s")]):
        sub = (st[st.analysis == f"coupling strength {out.upper()} interaction"]
               .set_index("ROI").loc[ROIS])
        for k, roi in enumerate(ROIS):
            r = sub.loc[roi]
            x = k + (j - .5) * .24
            sig = r.p < .05
            ax.plot([x, x], [0, r.estimate], lw=.9, color=S.FAINT, zorder=2)
            ax.plot(x, r.estimate, mk, ms=5.0 if sig else 3.6,
                    mfc=S.INK if sig else "white", mec=S.INK, mew=.9, zorder=3)
            if sig:
                ax.text(x, r.estimate + .030, "*", ha="center", va="bottom",
                        fontsize=7)
    ax.axhline(0, color=S.INK, lw=.6)
    ax.set_xticks(range(4))
    ax.set_xticklabels([r[0] for r in ROIS])
    ax.set_ylabel("Interaction β")
    ax.set_xlim(-.5, 3.5)
    ax.set_ylim(-.10, .44)
    ax.plot([], [], "o", mfc=S.INK, mec=S.INK, label="PPC")
    ax.plot([], [], "s", mfc="white", mec=S.INK, label="MRL")
    ax.legend(loc="upper left", ncol=2, fontsize=6, handletextpad=.3,
              columnspacing=.8, borderpad=.2)
    ax.set_title("Strength by region", fontsize=6.6, color=S.MUTED, pad=4)

    # ---- b  the direct frontal-vs-parietal contrast, with 95% CIs
    ax = fig.add_subplot(gs[0, 12:20])
    S.panel(ax, "b", dx=-.20)
    SHORT = {"Frontal": "Frontal", "Parietal": "Parietal",
             "Parietal − Frontal": "Par − Front"}
    order = [("PPC", "Frontal"), ("PPC", "Parietal"), ("PPC", "Parietal − Frontal"),
             ("MRL", "Frontal"), ("MRL", "Parietal"), ("MRL", "Parietal − Frontal")]
    ys = np.array([6, 5, 4, 2, 1, 0])
    for y, (met, term) in zip(ys, order):
        r = rc[(rc.metric == met) & (rc.term == term)].iloc[0]
        sig = (r.p < .05) if np.isfinite(r.p) else False
        diff = term.startswith("Parietal −")
        ax.plot([r.lo, r.hi], [y, y], lw=1.3 if diff else 1.0,
                color=S.INK if sig else S.MUTED, alpha=1.0 if sig else .55,
                zorder=2, solid_capstyle="round")
        ax.plot(r.est, y, "D" if diff else "o", ms=5.0 if sig else 3.8,
                mfc=S.INK if sig else "white", mec=S.INK, mew=.9, zorder=3)
    ax.axvline(0, color=S.INK, lw=.6, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([SHORT[t] for _, t in order], fontsize=6.1)
    ax.set_xlabel("Condition effect on coupling strength")
    ax.set_xlim(-.175, .175)
    ax.set_ylim(-.9, 7.1)
    ax.text(-.168, 6.75, "PPC", fontsize=6.2, color=S.MUTED, style="italic")
    ax.text(-.168, 2.75, "MRL", fontsize=6.2, color=S.MUTED, style="italic")
    ax.set_title("Direct regional contrast", fontsize=6.6, color=S.MUTED, pad=4)

    # ---- c  distance of the mean coupling phase from the up-state peak
    ax = fig.add_subplot(gs[0, 23:30])
    S.panel(ax, "c", dx=-.12)
    sig = ph[ph.p_FDR < .05].copy()
    sig["d_ctrl"] = sig.mean_ref_deg.abs()
    sig["d_tvns"] = sig.mean_cmp_deg.abs()
    sig = sig.sort_values("d_ctrl")
    lbl = {"TB": "This cohort", "IIT": "Cohort 2", "EC": "Cohort 3", "VIP": "Cohort 4"}
    yy = np.arange(len(sig))[::-1]
    for y, (_, r) in zip(yy, sig.iterrows()):
        closer = r.d_tvns < r.d_ctrl
        ax.plot([r.d_ctrl, r.d_tvns], [y, y], lw=1.0,
                color=S.INK if closer else S.MUTED, alpha=.45, zorder=2)
        ax.plot(r.d_ctrl, y, "o", ms=4.0, color=S.CTRL, mec="white", mew=.7, zorder=3)
        ax.plot(r.d_tvns, y, "o", ms=4.8 if closer else 3.4, color=S.TVNS,
                mec="white", mew=.7, zorder=4)
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{lbl.get(r.cohort, r.cohort)} · {r.roi[:4]}"
                        for _, r in sig.iterrows()], fontsize=5.8)
    ax.yaxis.tick_right()
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_xlabel("|mean coupling phase| (°)" + chr(10) + "0° = up-state peak")
    ax.set_xlim(-0.6, 15.5)
    ax.set_ylim(-2.3, len(sig) - .2)
    ax.plot([], [], "o", color=S.CTRL, label="No current")
    ax.plot([], [], "o", color=S.TVNS, label="tVNS")
    ax.legend(loc="lower left", fontsize=6, handletextpad=.3, labelspacing=.2,
              borderpad=.2, ncol=2, columnspacing=.9)
    ax.set_title("Phase alignment", fontsize=6.6, color=S.MUTED, pad=4)
    S.save(fig, "Figure_4")



# ==========================================================  FIGURE 5
def figure5():
    full = pd.read_csv(PAPER / "full" / "TB_full_summary.csv")
    f5 = pd.read_csv(PAPER / "full" / "F05_amplitude_quintiles.csv")
    t2 = pd.read_csv(PAPER / "T2_paired_densities_by_window.csv")
    ev = pd.read_parquet(PAPER / "TB_phase_events.parquet")
    ev["arm"] = ev.session_id.map(arm_of)

    fig = plt.figure(figsize=(S.W2, 3.5))
    axA, axB, axC, axD = [fig.add_subplot(1, 4, i) for i in range(1, 5)]
    fig.subplots_adjust(left=.075, right=.985, top=.855, bottom=.155,
                        wspace=.92)

    # ---- a  sigma power, within-session change
    ax = axA
    S.panel(ax, "a", dx=-.30)
    for j, (who, col) in enumerate([("Sham", S.CTRL), ("Treatment", S.TVNS)]):
        sub = (full[full.analysis == f"SWA within-session change (sigma, {who})"]
               .set_index("ROI").loc[ROIS])
        for k, roi in enumerate(ROIS):
            r = sub.loc[roi]
            x = k + (j - .5) * .28
            ax.bar(x, r.estimate, width=.25, color=col,
                   alpha=1.0 if r.p < .05 else .35, edgecolor="none", zorder=2)
            if r.p < .05:
                ax.text(x, r.estimate + 1.1, S.stars(r.p), ha="center",
                        fontsize=6.3)
    ax.axhline(0, color=S.INK, lw=.6)
    ax.set_xticks(range(4))
    ax.set_xticklabels([r[:4] for r in ROIS])
    ax.set_ylabel("Sigma power change (%)")
    ax.set_xlim(-.5, 3.5)
    ax.set_ylim(0, 36)
    ax.set_xticklabels([r[0] for r in ROIS])
    ax.set_title("Manipulation check", fontsize=6.6, color=S.MUTED, pad=4)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=S.CTRL, label="No current"),
                       Patch(facecolor=S.TVNS, label="tVNS")],
              loc="upper right", fontsize=6, handlelength=.9,
              handletextpad=.35, borderpad=.2)

    # ---- b  amplitude quintiles
    ax = axB
    S.panel(ax, "b", dx=-.30)
    for roi, mk, ls in [("Parietal", "o", "-"), ("Frontal", "s", (0, (3, 1.6)))]:
        sub = f5[f5.ROI == roi].sort_values("quintile")
        ax.plot(sub.quintile, sub.beta, marker=mk, ls=ls, color=S.TVNS,
                ms=4.2, lw=1.1, mfc="white" if roi == "Frontal" else S.TVNS,
                mec=S.TVNS, mew=1.1, label=roi, zorder=3)
        for _, r in sub.iterrows():
            if r.p < .05:
                ax.text(r.quintile, r.beta + .016, S.stars(r.p), ha="center",
                        fontsize=6.3)
    ax.axhline(0, color=S.INK, lw=.6)
    ax.set_xticks(range(1, 6))
    ax.set_xlabel("SO amplitude quintile")
    ax.set_ylabel("Interaction \u03b2")
    ax.set_xlim(.6, 5.4)
    ax.set_ylim(-.015, .275)
    ax.legend(loc="lower left", ncol=2, fontsize=6, handletextpad=.3,
              columnspacing=.9, handlelength=1.5, bbox_to_anchor=(-.02, -.03))

    # ---- c  densities
    ax = axC
    S.panel(ax, "c", dx=-.30)
    outs = ["coupled_density", "uncoupled_density", "SO_density", "spindle_density"]
    olab = ["Coupled", "Uncoupled", "All SO", "Spindles"]
    sub = t2[(t2.win == "stim") & (t2.ROI == "Parietal")].set_index("outcome")
    ys = np.arange(4)[::-1]
    for y, o in zip(ys, outs):
        r = sub.loc[o]
        col = S.TVNS if r.dz > 0 else S.CTRL
        ax.plot([r.ci_lo, r.ci_hi], [y, y], lw=1.1, color=col, zorder=2)
        ax.plot(r.dz, y, "o", ms=4.4, color=col, mec="white", mew=.7, zorder=3)
    ax.axvline(0, color=S.INK, lw=.6)
    ax.set_yticks(ys)
    ax.set_yticklabels(olab, fontsize=6.2)
    ax.tick_params(axis="y", pad=1.5)
    ax.set_xlabel("Paired d$_z$, parietal")
    ax.set_ylim(-.7, 3.7)
    ax.set_title("Event densities", fontsize=6.6, color=S.MUTED, pad=4)

    # ---- d  spindle timing relative to SO trough
    ax = axD
    S.panel(ax, "d", dx=-.30)
    sel = ev[(ev.win == "stim") & (ev.ch.isin(["P3", "P4"]))]
    bins = np.linspace(-1.0, 1.4, 49)
    for a, col in [("No current", S.CTRL), ("tVNS", S.TVNS)]:
        v = sel[sel.arm == a].dt_s.values
        h, e = np.histogram(v, bins=bins, density=True)
        c = (e[:-1] + e[1:]) / 2
        ax.plot(c, h, color=col, lw=1.3, zorder=3)
        ax.fill_between(c, 0, h, color=col, alpha=.12, lw=0, zorder=2)
        ax.axvline(np.median(v), color=col, lw=.9, ls=(0, (2.2, 1.6)), zorder=4)
    ax.set_xlabel("Time from SO trough (s)")
    ax.set_ylabel("Density")
    ax.set_xlim(-1.0, 1.4)
    ax.set_ylim(0, 1.80)
    ax.plot([], [], color=S.CTRL, label="No current")
    ax.plot([], [], color=S.TVNS, label="tVNS")
    ax.legend(loc="upper left", fontsize=6, handlelength=1.0,
              handletextpad=.35, borderpad=.15, labelspacing=.25)
    S.save(fig, "Figure_5")


# ==========================================================  FIGURE 6
# Source: SI_vagal/Stratification_report_2026-10-29.md, section 1
STRATA = [
    # label, level, TB stim (est, lo, hi, p), TB DiD, Cohort-2 DiD
    ("RMSSD", "low", (6.7, -0.1, 11.7, .039), (13.3, 2.8, 25.4, .014),
     (7.0, -0.1, 14.9, .060)),
    ("RMSSD", "high", (1.9, -2.7, 5.4, .36), (-2.4, -8.7, 4.7, .46),
     (4.9, 0.2, 9.1, .023)),
    ("ISI", "\u2265 15", (6.2, 2.9, 9.4, .001), (9.7, 1.0, 22.4, .026),
     (7.8, 1.2, 17.2, .036)),
    ("ISI", "< 15", (2.0, -2.3, 6.4, .39), (2.6, -4.5, 9.9, .50),
     (4.5, 0.4, 9.0, .036)),
    ("PSQI", "> 10", (4.7, 1.1, 7.6, .012), (9.1, 1.5, 18.0, .012),
     (6.2, 1.7, 11.9, .010)),
    ("PSQI", "\u2264 10", (2.8, -3.7, 7.9, .40), (0.5, -8.1, 9.5, .96),
     (0.9, -1.7, 2.6, .40)),
]
# Source: within-arm HRV, stage-matched N2, pre -> stimulation
HRV = [("RMSSD (ms)", 2.31, .011, 2.81, .087),
       ("SDNN (ms)", 7.7, .0005, 3.3, .060),
       ("Mean NN (ms)", 22.2, .0005, 28.7, .001),
       ("pNN50 (%)", 1.43, .050, 2.26, .018)]


def figure6():
    fig = plt.figure(figsize=(S.W2, 3.45))
    gs = fig.add_gridspec(1, 26, wspace=4.2, left=.075, right=.985,
                          top=.84, bottom=.18)

    # ---- a  baseline vagal tone as a CONTINUOUS moderator (the primary test)
    ax = fig.add_subplot(gs[0, 0:7])
    S.panel(ax, "a", dx=-.26)
    mod = pd.read_csv(STATS / "A_rmssd_continuous_data.csv")
    xs = np.linspace(mod.z_log_rmssd.min(), mod.z_log_rmssd.max(), 50)
    for c, col, lab in [(0, S.CTRL, "No current"), (1, S.TVNS, "tVNS")]:
        sub = mod[mod.cond == c]
        ax.plot(sub.z_log_rmssd, sub.dprev, "o", ms=2.8, color=col,
                alpha=.45, mec="none", zorder=2, label=lab)
        if len(sub) > 2:
            b1, b0 = np.polyfit(sub.z_log_rmssd, sub.dprev, 1)
            ax.plot(xs, b0 + b1 * xs, color=col, lw=1.4, zorder=3)
    ax.axhline(0, color=S.INK, lw=.6)
    ax.set_xlabel("Baseline log RMSSD (z)")
    ax.set_ylabel("Δ coupling prevalence (pp)")
    ax.legend(loc="lower right", fontsize=6, handletextpad=.3,
              labelspacing=.25)
    ax.set_title("Continuous moderation\ncondition × tone +1.6 pp/SD, P = 0.41",
                 fontsize=6.3, color=S.MUTED, pad=4, linespacing=1.4)

    # ---- b  DiD by stratum
    ax = fig.add_subplot(gs[0, 10:19])
    S.panel(ax, "b", dx=-.155)
    ys = np.arange(len(STRATA))[::-1]
    for y, (lab, lev, _stim, did, _c2) in zip(ys, STRATA):
        est, lo, hi, p = did
        sig = p < .05
        col = S.TVNS if est > 0 else S.CTRL
        ax.plot([lo, hi], [y + .13, y + .13], lw=1.2 if sig else .9, color=col,
                alpha=.95 if sig else .4, zorder=2, solid_capstyle="round")
        ax.plot(est, y + .13, "o", ms=5.0 if sig else 3.4, color=col,
                mec="white", mew=.8, zorder=3)
        e2, l2, h2, p2 = _c2
        c2col = S.TVNS if e2 > 0 else S.CTRL
        ax.plot([l2, h2], [y - .19, y - .19], lw=.9, color=c2col,
                alpha=.85 if p2 < .05 else .3, zorder=2)
        ax.plot(e2, y - .19, "^", ms=4.0 if p2 < .05 else 2.9, color=c2col,
                mec="white", mew=.6, zorder=3)
        ax.text(32.0, y + .13, f"{est:+.1f}", fontsize=6.1, ha="right", va="center",
                color=S.INK if sig else S.MUTED)
        ax.text(45.0, y + .13, S.pfmt(p), fontsize=6.1, ha="right", va="center",
                color=S.INK if sig else S.MUTED)
    ax.axvline(0, color=S.INK, lw=.6)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{lab.replace(chr(10), ' ')}  {lev}"
                        for lab, lev, *_ in STRATA], fontsize=6.1)
    ax.set_xlabel("Difference-in-differences in coupling prevalence (pp)")
    ax.set_xlim(-10, 46)
    ax.set_xticks([-10, 0, 10, 20])
    ax.spines["bottom"].set_bounds(-10, 25)
    ax.set_ylim(-.7, len(STRATA) - .1)
    for xx, lb in [(32.0, "pp"), (45.0, "P")]:
        ax.text(xx, len(STRATA) - .35, lb, fontsize=6, color=S.MUTED,
                ha="right", va="bottom")
    ax.set_title("By baseline stratum (exploratory)",
                 fontsize=6.5, color=S.MUTED, pad=4, loc="left")
    ax.plot([], [], "o", color=S.MUTED, label="This cohort")
    ax.plot([], [], "^", color=S.MUTED, label="Cohort 2")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.14), ncol=2,
              fontsize=5.8, handletextpad=.3, columnspacing=.8,
              labelspacing=.2, borderpad=.2)

    # ---- c  HRV within the stimulation arm
    ax = fig.add_subplot(gs[0, 21:26])
    S.panel(ax, "c", dx=-.30)
    yv = np.arange(len(HRV))[::-1]
    for y, (lab, dt, pt, ds, ps) in zip(yv, HRV):
        ax.plot([0, dt], [y, y], lw=1.3, color=S.TVNS,
                alpha=.95 if pt < .05 else .35)
        ax.plot(dt, y, "o", ms=4.8 if pt < .05 else 3.4, color=S.TVNS,
                mec="white", mew=.7, zorder=3)
        if pt < .05:
            ax.text(dt + .9, y, S.stars(pt), fontsize=6,
                    va="center", color=S.TVNS)
    ax.axvline(0, color=S.INK, lw=.6)
    ax.set_yticks(yv)
    ax.set_yticklabels([h[0] for h in HRV], fontsize=6.1)
    ax.set_xlabel("Change, pre \u2192 stimulation")
    ax.set_ylim(-.7, len(HRV) - .3)
    ax.set_title("Within tVNS sessions", fontsize=6.5,
                 color=S.MUTED, pad=4)
    S.save(fig, "Figure_6")


if __name__ == "__main__":
    print("building figures 3-6")
    figure3()
    figure4()
    figure5()
    figure6()
