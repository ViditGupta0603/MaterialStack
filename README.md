# MaterialStack

Band alignment screening for device stacks (solar cells, LEDs). For each layer it gives the band gap
(Eg) and the band edges vs vacuum (VBM, CBM); for each pair of neighbouring layers it gives the band
offsets and the junction type (I / II / III) with a confidence, so that only uncertain cases need DFT.

## How it works (one paragraph)

Measured values are used whenever they exist; a single ML model fills in the band gap when they don't.
Band edges come from photoemission measurements, else hybrid-DFT surface calculations (oxides), else
the Butler–Ginley electronegativity estimate (VBM = −χ − Eg/2). The junction type is not learned: it
follows from the four band edges. Its probability comes from the validated error of every input.

## Three steps

```bash
pip install -r requirements.txt
python build_data.py   # 1. downloads sources to data/raw/, cleans them, writes the data/*.csv tables (~15 s + download)
python train.py        # 2. trains the band-gap model → models/gap_model.joblib (~5 s)
python validate.py     # 3. cross-validation + benchmarks → results/metrics.csv (~15 s)
```

Then use it:

```bash
python -m materialstack predict TiO2 MAPbI3 Spiro-OMeTAD   # layers from top to bottom
python -m materialstack serve                              # web UI + API on http://127.0.0.1:8000
```

## Data (all CSV, open in Excel)

| File | What it is | Made by |
|---|---|---|
| `data/curated/measured_band_edges.csv` | Measured IE / EA / VBM of device layers, one cited paper per row | by hand |
| `data/curated/measured_band_offsets.csv` | Measured interface band offsets (photoemission), cited | by hand |
| `data/curated/aliases.csv` | Layer names → formulas (MAPbI3, CIGS, Spiro-OMeTAD …) | by hand |
| `data/band_gaps.csv` | One measured band gap per material: training labels and lookup | `build_data.py` |
| `data/band_gaps_rejected.csv` | Every report that was not used, with the rule (R1–R6) and the reason | `build_data.py` |
| `data/dft_gaps.csv` | One GGA and one hybrid DFT gap per formula (stable polymorphs): the model's hint | `build_data.py` |
| `data/band_edges.csv` | VBM (and CBM) per material: measured and/or hybrid-DFT surfaces | `build_data.py` |
| `data/raw/` | Downloaded source files, never edited (gitignored) | `build_data.py` |

Sources: measured gaps from Zhuo et al. 2018 (JPCL) and Borlido et al. 2019 (JCTC); DFT gaps from
JARVIS-DFT (Choudhary 2020) and SNUMAT (Kim 2020); oxide surfaces from Kiyohara, Hinuma & Oba 2024
(JACS); band edges and offsets from the photoemission papers cited in each row.

**Cleaning rules** (`build_data.py`): R1 unreadable formula · R2 radioactive elements · R3 gap outside
0–15 eV · R4 transcription error (> 3× the other reports) · R5 no agreement between reports (different
phases) · R6 measured gap > 2 eV from hybrid DFT. Borlido and curated values are preferred over the
compilation; otherwise the label is the median of the agreeing reports.

## The model

One LightGBM regressor (`materialstack/model.py`) predicts log(1 + Eg) for semiconductors and insulators
from ~150 composition features (matminer Magpie set) plus the DFT gap of the formula when JARVIS/SNUMAT
has one (missing otherwise). Cross-validation keeps materials made of the same elements in the same
fold, so the score is for unseen chemistry.

## Validation (results/metrics.csv)

| Check | Result | Baseline |
|---|---|---|
| Band gap, grouped 5-fold CV (2,509 materials) | MAE 0.40 eV, R² 0.86 | 1.12 eV (mean); hybrid DFT alone 0.54 eV |
| VBM when the measurement is hidden | Butler–Ginley 1.08 eV, hybrid-DFT surfaces 0.52 eV | — |
| 21 measured band offsets (InterMat) | 0.45 eV (0.59 without measured edges) | 0.84 eV (ΔEv = 0) |
| Gold device stacks: ΔEc / ΔEv sign right | 80 % / 83 % | — |
| Gold device stacks: junction type | 61 %; 82 % for confident calls | 61 % (always Type II) |

The gold stacks' band edges are SCAPS simulation inputs, which differ from photoemission measurements
by 0.4–0.8 eV for TiO2, SnO2, C60 and others (`literature_advay/findings.md`, F22), so type accuracy
against them has a ceiling set by the reference itself.

## Code

| File | Lines | Role |
|---|---|---|
| `build_data.py` | ~260 | step 1: sources → cleaned CSV tables |
| `train.py` | ~20 | step 2: train and save the model |
| `validate.py` | ~210 | step 3: metrics |
| `materialstack/model.py` | ~100 | features and the LightGBM model |
| `materialstack/predict.py` | ~240 | layers, band edges, junctions, confidence |
| `materialstack/chem.py` | ~80 | formulas, electronegativity, Butler–Ginley |
| `materialstack/api.py`, `cli.py` | ~120 | web API and command line |
| `frontend/` | | React UI (`npm run build`; served by `serve`) |
| `tests/` | | `python -m pytest -q` |

`literature_advay/` holds the literature notes, data-credibility audit and the dated findings log;
`CHANGELOG.md` records every change.
