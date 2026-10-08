# MaterialStack

Band alignment screening for device stacks (solar cells, LEDs). For each layer it gives the band gap
(Eg) and the band edges vs vacuum (VBM, CBM); for each pair of neighbouring layers it gives the band
offsets and the junction type (I / II / III) with a confidence, so that only uncertain cases need DFT.

## How it works (one paragraph)

Measured values are used whenever they exist; a single ML model fills in the band gap when they don't.
Band edges come from photoemission measurements, else hybrid-DFT surface calculations (oxides), else
the Butler–Ginley electronegativity estimate (VBM = −χ − Eg/2; in hybrid perovskites the organic cation
counts as Cs, since it sets no band edge). The junction type is not learned: it
follows from the four band edges. Its probability comes from the validated error of every input.

## Set up on another computer

Everything the tool needs is in the repository: the source data (`data/raw/`), the built tables, the trained
model and the built web UI. Only Python packages are installed.

1. Install **Python 3.12 or newer** (python.org; on Windows tick "Add python.exe to PATH") and Git.
2. `git clone https://github.com/advaymakhija/MaterialStack.git` (or GitHub → Code → Download ZIP).
3. In the `MaterialStack` folder run **`setup.bat`** (Windows, double-click works) or **`bash setup.sh`**
   (macOS / Linux). It creates `.venv`, installs the packages (needs internet, a few minutes) and runs one
   test prediction.
4. Run it: `.venv\Scripts\python -m materialstack serve` (Windows) or
   `.venv/bin/python -m materialstack serve`, then open http://127.0.0.1:8000.

After `git pull`, nothing needs rebuilding. When you change data or code, rerun the three steps below and commit
the regenerated files (`data/*.csv`, `models/gap_model.joblib`, `results/`; `frontend/dist/` after a UI change).

## Three steps

```bash
pip install -r requirements.txt
python build_data.py   # 1. reads data/raw/ (downloads a source only if its file is missing), cleans them, writes the data/*.csv tables (~15 s + download)
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
| `data/features.csv` | The exact table the model trained on (formula, gap, 157 features); for inspection only | `train.py` |
| `data/multilayer_gold_standard.csv` | 50 published device stacks with band edges, the validation reference | given |
| `data/raw/` | Source files as downloaded, never edited (committed, so setup needs no downloads) | `build_data.py` |

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

`validate.py` is a fixed protocol: it is never edited when the data grows. Each test set is a rule over the
current data (all measured gaps, all measured VBMs, every row of `data/curated/validation_band_offsets.csv`,
every gold stack), so new data joins validation automatically. Cross-validation folds are fixed by the
element system (a hash), so adding a material never reshuffles the others. Every mean has a 95 % bootstrap
interval (`ci_low`, `ci_high`); rerunning on unchanged data gives an identical file, so
`git diff results/metrics.csv` shows exactly what a change did. Decide on the **primary** rows:

| Primary metric | Result [95 % interval] | Baseline |
|---|---|---|
| Band-gap MAE, unseen chemistry (2,535 materials) | 0.40 eV [0.38–0.43] | 1.13 eV (mean); hybrid DFT alone 0.59 eV |
| VBM MAE when the measurement is hidden (44) | 0.57 eV [0.41–0.74] | — |
| \|ΔEv\| MAE, 21 measured offsets, no measured edges | 0.59 eV [0.44–0.74] | 0.84 eV (ΔEv = 0) |
| Gold stacks: ΔEc / ΔEv sign right | 80 % / 85 % | — |
| Gold stacks: ΔEc / ΔEv MAE | 0.50 / 0.53 eV | 0.45 / 0.80 eV (zero offset) |
| Junction-type Brier score (gold stacks) | 0.64 [0.55–0.73] | 0.78 (always Type II) |
| Confident calls (≥ 80 %): correct / share of junctions | 75 % (16) / 16 % | — |

Diagnostics: A2 (error by label source, chemistry and gap size; label-noise floor 0.15 eV), D (are the σ
values in `predict.py` right; reliability table), E (data coverage). The gold stacks' band edges are SCAPS
simulation inputs, which differ from photoemission by 0.4–0.8 eV for TiO2, SnO2, C60 and others, so type
accuracy against them has a ceiling set by the reference itself; it is reported but not a primary metric.

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
| `frontend/` | | React UI with four pages: Predict, Validation, Method, Guide (how to use + notes); `npm run build`, served by `serve` |
| `tests/` | | `python -m pytest -q` |

Literature notes, the data-credibility audit, the findings log and the change history are kept outside
this folder, in `../MaterialStack_extras/`.
