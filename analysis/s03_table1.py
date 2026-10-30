# -*- coding: utf-8 -*-
"""s03_table1.py - TB cohort characteristics for the manuscript Table 1."""
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT/'data'
OUT = ROOT/'analysis'/'outputs'
m = pd.read_csv(DATA/'session_mapping_with_scales.csv')
tb = m[m.Exp == 'TB'].copy()

# -800 is a missing-data sentinel in ESS / COMPASS31
for c in ['ESS', 'COMPASS31', 'PSQI', 'ISI', 'BAI', 'BDI2']:
    tb.loc[tb[c] <= -100, c] = np.nan

inv = pd.read_csv(OUT/'tb_session_inventory.csv')
tb = tb.merge(inv[['session', 'dur_min', 'bad_frac']], on='session', how='left')

subj = (tb.sort_values(['ID', 'rec_time'])
          .groupby('ID')
          .agg(Sex=('Sex', 'first'), Age=('Age', 'first'),
               n_sess=('session', 'size'),
               n_tvns=('arm', lambda s: (s == 'Treatment').sum()),
               n_ctrl=('arm', lambda s: (s == 'Sham').sum()),
               BAI=('BAI', 'first'), BDI2=('BDI2', 'first'),
               PSQI=('PSQI', 'first'), ISI=('ISI', 'first'),
               ESS=('ESS', 'first'))
          .reset_index())

def ms(s):
    s = s.dropna()
    return f"{s.mean():.1f} ({s.std():.1f})"

lines = ['# TB cohort characteristics (manuscript Table 1)', '']
lines += [f"Sessions: {len(tb)} (tVNS {int((tb.arm=='Treatment').sum())}, "
          f"unstimulated control {int((tb.arm=='Sham').sum())})",
          f"Participants: {len(subj)} ({int((subj.Sex=='F').sum())} women, "
          f"{int((subj.Sex=='M').sum())} men)",
          f"Age, years: {ms(subj.Age)}  [range {subj.Age.min():.0f}-{subj.Age.max():.0f}]",
          f"Sessions per participant: median {subj.n_sess.median():.0f} "
          f"(range {subj.n_sess.min()}-{subj.n_sess.max()})",
          f"Participants with >=1 control session: "
          f"{int((subj.n_ctrl>0).sum())}/{len(subj)}",
          f"Recording duration, min: {ms(tb.dur_min)}",
          f"Artifact-annotated fraction: {tb.bad_frac.mean():.3f} ({tb.bad_frac.std():.3f})",
          '']
lines += ['Baseline questionnaires (first visit, mean (SD)):']
for c, lab in [('PSQI', 'PSQI'), ('ISI', 'ISI'), ('BAI', 'BAI'),
               ('BDI2', 'BDI-II'), ('ESS', 'ESS')]:
    s = subj[c].dropna()
    lines.append(f"  {lab}: {ms(subj[c])}  n={len(s)}")
lines += ['', f"BAI-BDI-II correlation (first visit): r = "
          f"{subj[['BAI','BDI2']].corr().iloc[0,1]:.2f}"]

# sleep-stage composition per block, TB only
st = inv.copy()   # inventory already carries `arm`
lines += ['', '## Sleep-stage composition per block (fraction, mean)']
names = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
rows = []
for b in ['baseline', 'stimulation 1', 'recovery 1']:
    for s_ in range(5):
        c = f'st{s_}_{b}'
        if c in st:
            rows.append({'block': b, 'stage': names[s_],
                         'tVNS': st.loc[st.arm == 'Treatment', c].mean(),
                         'control': st.loc[st.arm == 'Sham', c].mean()})
sc = pd.DataFrame(rows).round(3)
lines += [sc.to_markdown(index=False)]
sc.to_csv(OUT/'tb_stage_composition.csv', index=False)
subj.to_csv(OUT/'tb_subject_table.csv', index=False, encoding='utf-8-sig')
(OUT/'tb_table1.md').write_text('\n'.join(lines), encoding='utf-8')
print('\n'.join(lines))
