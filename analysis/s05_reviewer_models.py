# -*- coding: utf-8 -*-
"""s05_reviewer_models.py — the two analyses Reviewer #2 asked for.

A. Baseline vagal tone as a CONTINUOUS moderator of the stimulation-associated
   change in parietal coupling prevalence (replaces the median split).
B. A DIRECT test of the frontal-versus-parietal contrast in coupling strength
   (replaces "significant here, not there").
"""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

DATA = Path(__file__).resolve().parent.parent / "data"
VNS = DATA / "vns"
PAPER = DATA / "paper"
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

pd.set_option("display.width", 200)
lines = []


def say(s=""):
    print(s)
    lines.append(str(s))


# ===================================================================== A
say("=" * 78)
say("A. Continuous baseline log RMSSD as a moderator of the coupling change")
say("=" * 78)

ar = pd.read_parquet(VNS / "01_PROJECT1_SO_SPINDLE_EFFECT/outputs/TB/"
                           "TB_analysis_ready.parquet")
ar = ar[ar.qc_pass]

ROI_CH = {"Parietal": ["P3", "P4"],
          "Frontal": ["Fp1", "Fp2", "F3", "F4"]}
ROI_PICK = __import__("os").environ.get("ROI", "Parietal")
say(f"[region = {ROI_PICK}  channels {ROI_CH[ROI_PICK]}]")
par = ar[ar.channel.isin(ROI_CH[ROI_PICK])]

# session-level prevalence per block
sess = (par.groupby(["id", "order", "Condition", "block"], observed=True)
           .agg(prev=("coupled_ratio", "mean"), n_SO=("n_SO", "sum"))
           .reset_index())
wide = sess.pivot_table(index=["id", "order", "Condition"], columns="block",
                        values="prev").reset_index()
wide = wide.dropna(subset=["baseline", "stimulation 1"])
wide["dprev"] = (wide["stimulation 1"] - wide["baseline"]) * 100   # pp

# baseline vagal tone: N2 RMSSD in the pre-stimulation window
hrv = pd.read_parquet(VNS / "rawfiles_fromJY/DATA/Data/Data/TB/"
                            "TB_HRV_stage_matched_DB.parquet")
base = hrv[(hrv.sleep_stage == "N2") & (hrv.phase == "baseline")]
base = base.dropna(subset=["RMSSD"])
# person-level trait: mean baseline RMSSD across that participant's sessions
subj = (base.groupby("ID").RMSSD.mean().rename("rmssd_base").reset_index()
            .rename(columns={"ID": "id"}))
subj["log_rmssd"] = np.log(subj.rmssd_base)
subj["z_log_rmssd"] = (subj.log_rmssd - subj.log_rmssd.mean()) / subj.log_rmssd.std(ddof=1)

d = wide.merge(subj, on="id", how="inner")
d["cond"] = (d.Condition.astype(str).str.lower().str.startswith("t")).astype(int)
say(f"sessions {len(d)} | participants {d.id.nunique()} | "
    f"tVNS {int(d.cond.sum())} / no current {int((1 - d.cond).sum())}")
say(f"baseline RMSSD (person mean): median {subj.rmssd_base.median():.1f} ms, "
    f"range {subj.rmssd_base.min():.1f}-{subj.rmssd_base.max():.1f}")
say()

m1 = smf.mixedlm("dprev ~ cond * z_log_rmssd", d, groups=d["id"]).fit(reml=False)
say("Model A1  dprev ~ cond * z_log_rmssd + (1|participant)")
say(m1.summary().tables[1].to_string())
b = m1.params["cond:z_log_rmssd"]
se = m1.bse["cond:z_log_rmssd"]
say(f"\n  INTERACTION cond x z_log_rmssd: {b:+.2f} pp per SD "
    f"[{b - 1.96*se:+.2f}, {b + 1.96*se:+.2f}], "
    f"P = {m1.pvalues['cond:z_log_rmssd']:.3f}")
say(f"  main effect of cond at mean baseline tone: "
    f"{m1.params['cond']:+.2f} pp, P = {m1.pvalues['cond']:.3f}")

# simple slopes across the observed range
say("\n  estimated tVNS - no current difference by baseline tone:")
for z in (-1.5, -1.0, 0.0, 1.0, 1.5):
    eff = m1.params["cond"] + b * z
    rm = np.exp(subj.log_rmssd.mean() + z * subj.log_rmssd.std(ddof=1))
    say(f"    z = {z:+.1f}  (RMSSD {rm:5.1f} ms)   {eff:+.2f} pp")

# sensitivity: session-specific baseline tone rather than the person mean
sb = (base.groupby(["ID", "Order"]).RMSSD.mean().rename("rmssd_sess")
          .reset_index().rename(columns={"ID": "id", "Order": "order"}))
d2 = wide.merge(sb, on=["id", "order"], how="inner")
d2["cond"] = (d2.Condition.astype(str).str.lower().str.startswith("t")).astype(int)
d2["z_log_rmssd"] = ((np.log(d2.rmssd_sess) - np.log(d2.rmssd_sess).mean())
                     / np.log(d2.rmssd_sess).std(ddof=1))
m2 = smf.mixedlm("dprev ~ cond * z_log_rmssd", d2, groups=d2["id"]).fit(reml=False)
say(f"\n  sensitivity, session-specific baseline tone (n = {len(d2)}): "
    f"interaction {m2.params['cond:z_log_rmssd']:+.2f} pp/SD, "
    f"P = {m2.pvalues['cond:z_log_rmssd']:.3f}")
d.to_csv(OUT / "A_rmssd_continuous_data.csv", index=False, encoding="utf-8-sig")

# ===================================================================== B
say()
say("=" * 78)
say("B. Direct frontal-versus-parietal contrast in coupling strength")
say("=" * 78)

pp = pd.read_csv(PAPER / "full" / "F16_strength_participant.csv")
st = pp[(pp.win == "stim") & (pp.ROI.isin(["Frontal", "Parietal"]))].copy()
st["cond"] = (st.arm.astype(str).str.lower().str.startswith("t")).astype(int)
st["is_par"] = (st.ROI == "Parietal").astype(int)
say(f"rows {len(st)} | participants {st.participant.nunique()} | "
    f"arms {sorted(st.arm.unique())}")

panel_rows = []
for metric in ["ppc", "mrl"]:
    sub = st.dropna(subset=[metric])
    m = smf.mixedlm(f"{metric} ~ cond * is_par", sub,
                    groups=sub["participant"]).fit(reml=False)
    b = m.params["cond:is_par"]
    se = m.bse["cond:is_par"]
    # frontal effect, parietal effect and their difference, each with a 95% CI
    bf, sef = m.params["cond"], m.bse["cond"]
    cov = m.cov_params()
    var_par = (cov.loc["cond", "cond"] + cov.loc["cond:is_par", "cond:is_par"]
               + 2 * cov.loc["cond", "cond:is_par"])
    bp, sep = bf + b, float(np.sqrt(var_par))
    for lab, est, s_, pv in [("Frontal", bf, sef, m.pvalues["cond"]),
                             ("Parietal", bp, sep, np.nan),
                             ("Parietal − Frontal", b, se, m.pvalues["cond:is_par"])]:
        panel_rows.append({"metric": metric.upper(), "term": lab, "est": est,
                           "lo": est - 1.96 * s_, "hi": est + 1.96 * s_, "p": pv})
    say(f"\n  {metric.upper()}  ~ cond * region + (1|participant), n = {len(sub)}")
    say(f"    condition effect at FRONTAL : {m.params['cond']:+.4f} "
        f"(P = {m.pvalues['cond']:.3f})")
    say(f"    region x condition INTERACTION: {b:+.4f} "
        f"[{b - 1.96*se:+.4f}, {b + 1.96*se:+.4f}], P = {m.pvalues['cond:is_par']:.3f}")
    say(f"    condition effect at PARIETAL: {m.params['cond'] + b:+.4f}")

# paired version: participants contributing both regions, difference of differences
say("\n  paired check (participants with both regions in both arms):")
for metric in ["ppc", "mrl"]:
    w = st.pivot_table(index="participant", columns=["ROI", "cond"],
                       values=metric)
    need = [("Frontal", 0), ("Frontal", 1), ("Parietal", 0), ("Parietal", 1)]
    if not all(c in w.columns for c in need):
        say(f"    {metric.upper()}: cells missing, skipped")
        continue
    w = w.dropna(subset=need)
    dfr = w[("Frontal", 1)] - w[("Frontal", 0)]
    dpa = w[("Parietal", 1)] - w[("Parietal", 0)]
    from scipy import stats
    t, p = stats.ttest_rel(dpa, dfr)
    say(f"    {metric.upper()}: n = {len(w)}, parietal-minus-frontal effect "
        f"{np.mean(dpa - dfr):+.4f}, t = {t:.2f}, P = {p:.3f}")

pd.DataFrame(panel_rows).to_csv(OUT / "B_region_contrast.csv", index=False,
                                encoding="utf-8-sig")

(OUT / "reviewer_models_report.md").write_text("\n".join(lines), encoding="utf-8")
print("\n[saved]", OUT / "reviewer_models_report.md")
