# -*- coding: utf-8 -*-
"""s04_positive_scan.py - inventory every POSITIVE (tVNS-favouring) cell across
the TB crossover full summary and the four-cohort P1_13 tables, always with the
denominator (how many cells were tested in that family).
"""
from pathlib import Path
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
TBX = DATA / "paper" / "full" / "TB_full_summary.csv"
TBS = DATA / "paper" / "full" / "TB_full_summary_strength.csv"
P1 = DATA / "vns" / "01_PROJECT1_SO_SPINDLE_EFFECT" / "outputs" / "shared"
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 400)

lines = []


def say(s=""):
    print(s)
    lines.append(str(s))


# ---------------------------------------------------------------- TB crossover
frames = []
for f in (TBX, TBS):
    if f.exists():
        frames.append(pd.read_csv(f))
tb = pd.concat(frames, ignore_index=True).drop_duplicates()

say("=" * 100)
say("TB CROSSOVER full inventory  (source: TB crossover/paper/full/)")
say(f"total cells: {len(tb)}")
say(tb.direction.value_counts(dropna=False).to_string())
say()

pos = tb[tb.direction == "+"].copy()
neg = tb[tb.direction.isin(["-", "−"])].copy()   # data uses U+2212 MINUS SIGN
say(f"POSITIVE (tVNS-favouring) cells: {len(pos)} / {len(tb)} "
    f"= {100*len(pos)/len(tb):.1f}%")
say(f"NEGATIVE (sham-favouring) cells: {len(neg)} / {len(tb)} "
    f"= {100*len(neg)/len(tb):.1f}%")
say()
say("--- ALL POSITIVE CELLS, sorted by q then p ---")
pos_s = pos.sort_values(["q", "p"], na_position="last")
say(pos_s[["analysis", "outcome", "ROI", "estimate", "p", "q", "n", "unit"]]
    .to_string(index=False))
say()
say("--- POSITIVE cells that survive q < .05 ---")
say(pos_s[pos_s.q < .05][["analysis", "outcome", "ROI", "estimate", "p", "q", "n", "unit"]]
    .to_string(index=False))
say()
say("--- ALL NEGATIVE (sham-favouring) CELLS, for balance ---")
say(neg.sort_values(["q", "p"], na_position="last")
    [["analysis", "outcome", "ROI", "estimate", "p", "q", "n", "unit"]]
    .to_string(index=False))
say()

# denominator per analysis family
say("--- per-analysis family: positives / total ---")
fam = (tb.assign(is_pos=(tb.direction == "+").astype(int),
                 is_neg=(tb.direction.isin(["-", "−"])).astype(int))
         .groupby("analysis")
         .agg(cells=("direction", "size"), pos=("is_pos", "sum"), neg=("is_neg", "sum"))
         .sort_values("pos", ascending=False))
say(fam.to_string())
say()

pos_s.to_csv(OUT / "positive_cells_TBcrossover.csv", index=False, encoding="utf-8-sig")
fam.to_csv(OUT / "family_denominators_TBcrossover.csv", encoding="utf-8-sig")

# ------------------------------------------------- four-cohort P1_13 (3 metrics)
say("=" * 100)
say("FOUR-COHORT P1_13 tables  (prevalence / strength / phase; tracks jy, iy, origstg)")
for metric in ["prevalence", "strength", "phase"]:
    f = P1 / f"P1_13_{metric}_all_cohorts.csv"
    if not f.exists():
        continue
    d = pd.read_csv(f)
    say()
    say(f"### {metric}  ({len(d)} cells)")
    if metric == "phase":
        d["eff"] = d.get("shift_deg")
        pcol = "p"
    else:
        d["eff"] = d["beta"].fillna(d.get("g"))
        pcol = "p"
    hit = d[(d[pcol] < .05)].copy()
    say(f"  nominal p<.05: {len(hit)} / {len(d)}")
    cols = [c for c in ["cohort", "track", "roi", "eff", "se", "p", "p_FDR",
                        "n", "n_subject", "design"] if c in d.columns]
    if len(hit):
        say(hit.sort_values("p")[cols].to_string(index=False))
    fdr = d[d.p_FDR < .05] if "p_FDR" in d else d.iloc[:0]
    say(f"  survives FDR: {len(fdr)} / {len(d)}")
    if len(fdr):
        say(fdr.sort_values("p_FDR")[cols].to_string(index=False))

(OUT / "positive_scan_report.md").write_text("\n".join(lines), encoding="utf-8")
print("\n[saved]", OUT / "positive_scan_report.md")
