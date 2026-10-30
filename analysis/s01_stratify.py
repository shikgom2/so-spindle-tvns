# -*- coding: utf-8 -*-
"""
s01_stratify.py — Build all 5 cutoff options (A-E) as subject-level strata.

Subject unit = `ID` (original P###/I### from session_mapping_with_scales.csv).
`participant_id` in sessions_key.xlsx is NOT a person and is never used here.

Group label is fixed at the subject's FIRST scored visit (scores drift ~3-4 pts
between visits, so per-visit labelling would flip borderline subjects).

Options
  A  Beck manual moderate+      BAI>=16        BDI-2>=20
  B  Beck manual mild+          BAI>=8         BDI-2>=14
  C  Korean validation          BAI>=22        BDI-2>=14
  D  Sample tertiles            top vs bottom third of first-visit score
  E  Continuous                 no cut; z-scored score as moderator
Plus: SYMPTOMATIC = (BAI>=16) | (BDI-2>=20)   -- combined flag for VNS contrast
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "analysis" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

CUTS = {
    "A_beck_moderate": {"BAI": 16, "BDI2": 20},
    "B_beck_mild":     {"BAI": 8,  "BDI2": 14},
    "C_korean":        {"BAI": 22, "BDI2": 14},
}


def main():
    m = pd.read_csv(DATA / "session_mapping_with_scales.csv")

    # ---- subject level: first scored visit per subject -------------------
    scored = m.dropna(subset=["BAI", "BDI2"]).sort_values(["ID", "rec_time"])
    subj = (scored.groupby("ID")
                  .agg(Exp=("Exp", "first"),
                       Sex=("Sex", "first"),
                       Age=("Age", "first"),
                       Comorbidity=("Comorbidity", "first"),
                       n_scored_visits=("session", "size"),
                       BAI=("BAI", "first"),
                       BDI2=("BDI2", "first"),
                       BAI_mean=("BAI", "mean"),
                       BDI2_mean=("BDI2", "mean"),
                       BAI_sd=("BAI", "std"),
                       BDI2_sd=("BDI2", "std"))
                  .reset_index())

    # ---- A / B / C : fixed clinical thresholds ---------------------------
    for name, th in CUTS.items():
        subj[f"{name}_anx"] = np.where(subj.BAI >= th["BAI"], "high", "low")
        subj[f"{name}_dep"] = np.where(subj.BDI2 >= th["BDI2"], "high", "low")

    # ---- D : sample tertiles (middle third dropped) ----------------------
    for scale, col in [("anx", "BAI"), ("dep", "BDI2")]:
        lo, hi = np.percentile(subj[col], [33.333, 66.667])
        subj[f"D_tertile_{scale}"] = np.select(
            [subj[col] <= lo, subj[col] >= hi], ["low", "high"], default="mid")
        subj.attrs[f"D_{scale}_bounds"] = (lo, hi)

    # ---- E : continuous moderators (z-scored within cohort) --------------
    for col in ["BAI", "BDI2"]:
        subj[f"E_{col}_z"] = (subj.groupby("Exp")[col]
                                  .transform(lambda s: (s - s.mean()) / s.std(ddof=1)))

    # ---- combined symptomatic flag ---------------------------------------
    subj["SYMPTOMATIC"] = np.where((subj.BAI >= 16) | (subj.BDI2 >= 20),
                                   "symptomatic", "nonsymptomatic")
    subj["BOTH_moderate"] = np.where((subj.BAI >= 16) & (subj.BDI2 >= 20),
                                     "both", "not_both")

    subj.to_csv(OUT / "strata_subject_level.csv", index=False,
                encoding="utf-8-sig")

    # ---- session-level join (every session inherits its subject's label) --
    label_cols = [c for c in subj.columns
                  if c.startswith(("A_", "B_", "C_", "D_", "E_"))
                  or c in ("SYMPTOMATIC", "BOTH_moderate")]
    sess = m.merge(subj[["ID"] + label_cols], on="ID", how="left")
    sess.to_csv(OUT / "strata_session_level.csv", index=False,
                encoding="utf-8-sig")

    # ---- report -----------------------------------------------------------
    lines = ["# Stratification by cutoff option", "",
             f"Subjects with scores: **{len(subj)}** "
             f"(TB {int((subj.Exp=='TB').sum())}, IIT {int((subj.Exp=='IIT').sum())}) "
             f"of 74 scored; labels fixed at first scored visit.", ""]

    lo_a, hi_a = subj.attrs["D_anx_bounds"]
    lo_d, hi_d = subj.attrs["D_dep_bounds"]
    lines += [f"Option D tertile bounds: BAI low<={lo_a:.0f} / high>={hi_a:.0f}; "
              f"BDI-2 low<={lo_d:.0f} / high>={hi_d:.0f}", ""]

    rows = []
    for name in list(CUTS) + ["D_tertile"]:
        for scale, pretty in [("anx", "Anxiety (BAI)"), ("dep", "Depression (BDI-2)")]:
            col = f"{name}_{scale}" if name != "D_tertile" else f"D_tertile_{scale}"
            vc = subj[col].value_counts()
            for cohort in ["TB", "IIT"]:
                vcc = subj.loc[subj.Exp == cohort, col].value_counts()
                rows.append({"option": name, "scale": pretty, "cohort": cohort,
                             "n_high": int(vcc.get("high", 0)),
                             "n_low": int(vcc.get("low", 0)),
                             "n_mid_dropped": int(vcc.get("mid", 0))})
            rows.append({"option": name, "scale": pretty, "cohort": "BOTH",
                         "n_high": int(vc.get("high", 0)),
                         "n_low": int(vc.get("low", 0)),
                         "n_mid_dropped": int(vc.get("mid", 0))})
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "strata_counts.csv", index=False)

    lines += ["## Subject counts per option", "",
              tab.pivot_table(index=["option", "scale"], columns="cohort",
                              values=["n_high", "n_low"], aggfunc="sum")
                 .to_markdown(), ""]

    # sessions available per stratum, split by stimulation arm
    sess_tab = []
    for name in list(CUTS) + ["D_tertile"]:
        for scale in ["anx", "dep"]:
            col = f"{name}_{scale}"
            g = (sess.dropna(subset=[col])
                     .groupby([col, "arm"]).size().unstack(fill_value=0))
            for lvl in g.index:
                sess_tab.append({"option": name, "scale": scale, "level": lvl,
                                 "sessions_stim": int(g.loc[lvl].get("Treatment", 0)),
                                 "sessions_nostim": int(g.loc[lvl].get("Sham", 0))})
    st = pd.DataFrame(sess_tab)
    st.to_csv(OUT / "strata_session_counts.csv", index=False)
    lines += ["## Sessions per stratum (stim vs no-stim)", "",
              st.to_markdown(index=False), ""]

    sym = subj.SYMPTOMATIC.value_counts()
    lines += ["## Combined flags", "",
              f"- SYMPTOMATIC (BAI>=16 or BDI-2>=20): "
              f"**{int(sym.get('symptomatic', 0))}** symptomatic / "
              f"{int(sym.get('nonsymptomatic', 0))} not",
              f"- BOTH moderate (BAI>=16 and BDI-2>=20): "
              f"**{int((subj.BOTH_moderate=='both').sum())}**",
              f"- BAI-BDI correlation at first visit: "
              f"r = {subj[['BAI','BDI2']].corr().iloc[0,1]:.2f}", ""]

    (OUT / "strata_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
