# -*- coding: utf-8 -*-
"""s02_scan_tb.py - inventory every TB session: blocks, trigger, ECG, sleep stages."""
from pathlib import Path
import numpy as np, pandas as pd, mne
mne.set_log_level('ERROR')

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT/'data'
OUT = ROOT/'analysis'/'outputs'; OUT.mkdir(parents=True, exist_ok=True)
BLOCKS = ['baseline', 'stimulation 1', 'recovery 1']

m = pd.read_csv(DATA/'session_mapping_with_scales.csv')
tb = m[m.Exp == 'TB'].copy()

rows = []
for _, r in tb.iterrows():
    f = DATA/'fif'/f'{r.session}_raw_fin.fif'
    d = {'session': r.session, 'ID': r.ID, 'arm': r.arm, 'use': r.Use_flag}
    if not f.exists():
        d['err'] = 'missing'; rows.append(d); continue
    raw = mne.io.read_raw_fif(f)
    sf = raw.info['sfreq']
    d['dur_min'] = raw.times[-1]/60
    d['n_eeg'] = len(mne.pick_types(raw.info, eeg=True))
    d['has_ecg'] = 'ecg' in raw.ch_names
    d['has_trig'] = 'trigger' in raw.ch_names
    a = raw.annotations
    for b in BLOCKS:
        hit = [(o, du) for dd, o, du in zip(a.description, a.onset, a.duration) if dd == b]
        d[f'{b}_n'] = len(hit)
        d[f'{b}_min'] = round(hit[0][1]/60, 1) if hit else np.nan
        d[f'{b}_on'] = round(hit[0][0], 1) if hit else np.nan
    # artifact burden
    bad = sum(du for dd, du in zip(a.description, a.duration) if dd.startswith('BAD'))
    d['bad_frac'] = round(bad/raw.times[-1], 3)
    # trigger activity per block
    if d['has_trig']:
        t = raw.get_data(picks=[raw.ch_names.index('trigger')])[0]
        d['trig_uniq'] = int(len(np.unique(t)))
        d['trig_nz_frac'] = round(float((t != 0).mean()), 3)
        for b in BLOCKS:
            hit = [(o, du) for dd, o, du in zip(a.description, a.onset, a.duration) if dd == b]
            if hit:
                o, du = hit[0]
                seg = t[int(o*sf):int((o+du)*sf)]
                d[f'trig_nz_{b}'] = round(float((seg != 0).mean()), 3)
    # sleep stage coverage per block
    if 'sleep' in raw.ch_names:
        s = raw.get_data(picks=[raw.ch_names.index('sleep')])[0]
        for b in BLOCKS:
            hit = [(o, du) for dd, o, du in zip(a.description, a.onset, a.duration) if dd == b]
            if hit:
                o, du = hit[0]
                seg = s[int(o*sf):int((o+du)*sf)]
                for st in range(5):
                    d[f'st{st}_{b}'] = round(float((seg == st).mean()), 3)
    # ECG sanity
    if d['has_ecg']:
        e = raw.get_data(picks=[raw.ch_names.index('ecg')])[0]
        d['ecg_sd'] = float(np.std(e)); d['ecg_flat'] = bool(np.std(e) == 0)
    rows.append(d)

df = pd.DataFrame(rows)
df.to_csv(OUT/'tb_session_inventory.csv', index=False, encoding='utf-8-sig')

print('TB sessions scanned:', len(df))
print('\n-- block presence by arm --')
for b in BLOCKS:
    print(b, df.groupby('arm')[f'{b}_n'].apply(lambda s: (s > 0).sum()).to_dict(),
          '| median min', df.groupby('arm')[f'{b}_min'].median().round(1).to_dict())
print('\n-- channels / signals --')
print(df.groupby('arm')[['n_eeg', 'dur_min', 'bad_frac']].median().round(2))
print('has_ecg', df.has_ecg.sum(), 'ecg_flat', df.ecg_flat.sum() if 'ecg_flat' in df else 'NA')
print('\n-- trigger non-zero fraction per block (mean by arm) --')
tc = [c for c in df.columns if c.startswith('trig_nz')]
print(df.groupby('arm')[tc].mean().round(3).T)
print('\n-- sleep stage fraction per block (mean by arm) --')
for b in BLOCKS:
    cs = [f'st{st}_{b}' for st in range(5) if f'st{st}_{b}' in df]
    print(b); print(df.groupby('arm')[cs].mean().round(3))
