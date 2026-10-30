# tVNS × SO–spindle coupling — analysis and figure code

Code accompanying *"Cervical vagal afferent stimulation during daytime sleep increases
parietal slow oscillation–spindle coupling and regionally redistributes coupling
precision."*

All scripts are read-only with respect to the source recordings: they read from `data/`
and write only into `analysis/outputs/` (tables) or `figures/` (rendered figures).

## Layout

```
analysis/     statistical analyses that produce the numbers in the manuscript
figures/      figure generation, the figure style, and the QC tools
data/         local copy of the derived-data tree (not tracked; see below)
```

### `analysis/`

| script | what it does |
|---|---|
| `s01_stratify.py` | builds subject-level symptom strata (BAI / BDI-II cut-off options A–E) |
| `s02_scan_tb.py` | inventories every session: block structure, trigger, ECG, sleep-stage coverage |
| `s03_table1.py` | participant characteristics and sleep-stage composition (manuscript Table 1–2) |
| `s04_positive_scan.py` | enumerates every tVNS-favouring cell **with its denominator**, so effects are never read without the number of tests behind them |
| `s05_reviewer_models.py` | the two models requested at review: (A) baseline log RMSSD as a *continuous* moderator of the coupling change, replacing a median split; (B) a direct frontal-versus-parietal contrast in coupling strength. Set `ROI=Frontal` in the environment to run (A) at frontal sites. |

### `figures/`

| script | what it does |
|---|---|
| `figstyle.py` | Nature-style rcParams, the two-colour condition palette, panel letters, `save()` |
| `fig1_2.py` | Figures 1–2 (design and coupling rule; prevalence in detail) |
| `fig3_6.py` | Figures 3–6 (SO-locked sigma; strength and phase; event properties; moderation) |
| `overlapcheck.py` | **collision auditor** — see below |
| `validate_palette.py` | colour-blindness validator (OKLab ΔE under deuteranopia / protanopia / tritanopia, lightness band, chroma floor, contrast) |

#### The collision auditor

`figstyle.save()` runs `overlapcheck.audit()` on every figure before writing it. The
auditor renders the figure twice — once with every `Text` artist hidden, to build an
"ink" mask of all non-text drawing, and once normally, to get each text's bounding box —
then reports any text that sits on drawn content. Because it works on rendered pixels it
catches collisions with lines, bars, shaded bands, image cells and polar grids alike,
which a check against a hand-picked list of artists does not.

Five checks: text/text, off-canvas, text crossing into a neighbouring axes, text on ink,
and frameless-legend-on-ink. Deliberate overlays (a white label inside a filled block, a
significance glyph centred in its marker, matrix values on their coloured cells) are
exempted with `artist.set_gid("nocheck")` or `"ink-ok"`.

All six manuscript figures pass with zero collisions.

#### Colour

Categorical hue is reserved for the two experimental conditions, which is the only true
categorical dimension in these data; region of interest is encoded by position, never by
hue. The condition pair was validated with `validate_palette.py`:

```
#1B6CA8 / #D1603D   normal ΔE 28.5 · deuteranopia 25.5 · protanopia 17.6 · tritanopia 40.7
                    lightness, chroma and contrast all pass
```

## Reproducing

```bash
python analysis/s03_table1.py          # Tables 1–2
python analysis/s05_reviewer_models.py # the two review models
python figures/fig1_2.py               # Figures 1–2 (+ collision audit)
python figures/fig3_6.py               # Figures 3–6 (+ collision audit)
```

Run `s05_reviewer_models.py` before `fig3_6.py`: Figures 4 and 6 read its tables from
`analysis/outputs/`.

All paths are relative to the repository root. Place a local copy of the derived-data
tree under `data/` (git-ignored) with this layout:

```
data/
  session_mapping_with_scales.csv    session ↔ participant map with symptom scales (s01–s03)
  fif/<session>_raw_fin.fif          preprocessed recordings (s02)
  paper/                             frozen TB-crossover summaries (T*, full/F*, TB_*)
  vns/
    01_PROJECT1_SO_SPINDLE_EFFECT/outputs/TB/TB_analysis_ready.parquet
    01_PROJECT1_SO_SPINDLE_EFFECT/outputs/shared/P1_13_*_all_cohorts.csv
    rawfiles_fromJY/DATA/Data/Data/TB/TB_HRV_stage_matched_DB.parquet
```

## Data availability

The recordings and derived event tables are not included here. They contain information
that could compromise participant privacy and are available from the corresponding author
on reasonable request.

## Environment

Python 3.12 with `numpy`, `pandas`, `scipy`, `statsmodels`, `matplotlib`, `mne`, `yasa`,
`python-docx`, `pyarrow`.

## Licence

MIT — see [`LICENSE`](LICENSE).
