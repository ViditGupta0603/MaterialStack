# MaterialStack

Band gap / band edge / junction-type pipeline for layered material stacks (solar-cell absorbers,
ETLs, HTLs and other inorganic layers).

Stage 1 builds the **materials database**. Stage 2 trains **LightGBM** models for band gap
(Eg) and vacuum band-edge correction (CBM/VBM). Stage 3 serves the **MaterialStack** web UI
(lookup-first predict + performance report).

## Install

```bash
pip install -r requirements.txt
```

Python 3.10+ (developed on 3.13). `automatminer` is intentionally not used (unmaintained).
Primary ML stack: **LightGBM** (thesis-aligned). XGBoost is optional as a baseline in `train`.

## Build the database

```bash
python -m materialstack build-db                 # default sources, no API key needed (~1-2 GB download)
python -m materialstack build-db --heavy         # also matbench_mp_gap + mp_all_20181018 (needs ~8 GB RAM)
python -m materialstack build-db --mp-api-key KEY   # add a live Materials Project pull (cached to parquet)
python -m materialstack build-db --only jarvis_surfacedb jarvis_interfacedb   # re-ingest selected sources
python -m materialstack stats
python -m materialstack lookup TiO2
python -m materialstack lookup MAPbI3            # aliases are resolved
python -m materialstack featurize --n-jobs 4     # cache composition + structure features
python -m materialstack predict TiO2            # lookup-first Eg/CBM/VBM
python -m materialstack predict TiO2 MAPbI3     # stack → junctions (Type I/II/III)
python -m materialstack predict TiO2 --max-lookup-rank 0 --polymorph mean   # force ML, average polymorphs
```

## MaterialStack web UI

```bash
pip install -r requirements.txt
cd frontend && npm install && npm run build && cd ..
python -m materialstack serve                  # http://127.0.0.1:8000
```

Dev (API + Vite hot reload):

```bash
# terminal 1
python -m materialstack serve --reload
# terminal 2
cd frontend && npm run dev                 # http://127.0.0.1:5173 (proxies /api)
```

The UI keeps the trained LightGBM models as-is and shows the full performance report from
`models/metrics.json`.

### Lookup-first prediction

`predict` prefers trusted DB labels, then falls back to LightGBM:

1. Resolve alias / formula → `materials_best`
2. If best gap method rank ≤ `--max-lookup-rank` (default **2** = experiment/literature/HSE06) → use DB Eg
3. If vacuum CBM/VBM exist → use DB edges; else ML δCBM (+ Butler–Ginley) or Butler–Ginley alone
4. ML band gap: the structure-aware model when the DB has a bulk crystal structure or stored DFT
   gaps for the material; unknown formulas → Magpie features on the fly + composition-only model
5. Junction type from CBO/VBO only (not a classifier); flagged **UNCERTAIN** near ~0.3 eV boundaries or when ML edges are involved

**Polymorphs** (`--polymorph` / UI toggle): one formula can have several crystal structures.
- `ground_state` (default): the structure with the lowest DFT energy among the formula's polymorphs
  (JARVIS), else the most trusted source — the conventional choice in materials science.
- `mean`: one structure per space group within 0.05 eV/atom of the lowest polymorph; predicted
  Eg is averaged. Stored JARVIS hull energies carry a per-chemical-system offset, so only energies
  relative to the lowest polymorph of the same formula are used.

## Train models

```bash
python -m materialstack featurize-structures --n-jobs 7   # ~2 min: structure features for every CIF
python -m materialstack train-proxies                     # ~2.5 min: DFT-gap proxy models
python -m materialstack train                    # LightGBM primary + XGBoost baseline CV + structure-aware model
python -m materialstack train --no-xgb-baseline  # LightGBM only (faster)
python -m materialstack train --no-structure     # composition-only, as before
# old mixed-fidelity labels (higher MAE ~0.6 eV):
python -m materialstack train --eg-methods experiment,HSE06,TBmBJ
```

Writes:
- `models/eg_model.joblib` — two-stage metal classifier + Eg regressor (composition only)
- `models/eg_structure_model.joblib` — same two-stage model on composition + structure + DFT features
- `models/proxy_{optb88vdw,tbmbj,hse06}.joblib`, `models/proxy_metrics.json` — DFT-gap proxies
- `models/edge_model.joblib` — Butler-Ginley CBM correction (VBM = CBM − Eg)
- `models/metrics.json` — GroupKFold metrics for LightGBM, XGBoost baseline and the structure ablation
- `models/train_clean_audit.json` — which cleaning rules dropped how many rows

**Structure-aware band gap model.** Features on top of the 180 composition features:
- 26 crystal-structure descriptors (`features_structure`, prefix `sf_`): volume per atom, density,
  packing fraction, space group / crystal system / centrosymmetry, nearest-neighbour distances,
  bond length ÷ covalent-radius sum, coordination numbers, fraction of unlike neighbours. Only
  cell-choice-invariant quantities are used, because CIFs mix primitive and conventional cells.
- 3 DFT proxies (`structure_proxy`, transfer learning): two-stage LightGBM models trained on ~94k
  JARVIS OptB88vdW, ~21.6k TBmBJ and ~10.8k HSE06 gaps. Structures in a proxy's training set get
  out-of-fold predictions (GroupKFold by element set), so no proxy value was fitted on its own chemistry.
- 5 stored DFT gaps (OptB88vdW, TBmBJ, HSE06, PBE, GLLB-SC) from the DB: the calculation on the
  chosen structure if present, else the material's median.

Missing values (no structure / no DFT) are left as NaN, which LightGBM routes natively, so the model
trains on all materials. It is scored on exactly the composition-only folds, with an ablation per
feature group. Materials with no structure and no stored DFT keep the composition-only model when
that scores better on that subset.

**Label cleaning (training frames only; SQLite is not modified):**
- Default Eg methods: **experiment + literature** (HSE/TBmBJ excluded from Model A targets)
- Drop non-metals with Eg > `--max-gap` (default 12 eV)
- Drop materials with large experiment-label scatter (`--max-expt-spread`)
- Drop experiment materials that disagree with nearest HSE/TBmBJ by > `--max-dft-conflict` (default 2 eV)
- Edge rows: require |(CBM−VBM)−Eg| ≤ 0.35 eV and |δCBM| ≤ 5 eV

Borlido-only materials stay held out of Eg training and are scored separately.

Output: `data/materials_db.sqlite` and `data/materials_db_summary.csv`.

### Sources

| source key | rows | provides | method tag |
|---|---|---|---|
| `expt_gap`, `expt_gap_kingsbury` | 6354 / 4604 | experimental Eg | experiment |
| `foundry_ml_exp_bandgaps` | 2069 | experimental Eg | experiment |
| `borlido_expt` | 472 | experimental Eg (held-out test set) | experiment |
| `snumat` | 10481 | HSE06 + PBE Eg, structure | HSE06 / PBE |
| `jarvis_dft_3d` | ~94k | OptB88vdW, TBmBJ, HSE06 Eg, structure | TBmBJ / OptB88vdW / HSE06 |
| `jarvis_dft_2d` | ~1.1k | same, 2D materials | |
| `jarvis_halide_perovskites` | 229 | HSE06 + PBE Eg, structure | |
| `jarvis_surfacedb` | 607 | vacuum-referenced VBM, CBM, work function per surface | DFT-surface-OptB88vdW, DFT-surface-TBmBJ |
| `jarvis_interfacedb` | 593 | DFT valence-band offsets (validation) | |
| `castelli_perovskites` | 18928 | GLLB-SC Eg, VBM, CBM (cubic ABX3) | GLLB-SC |
| `double_perovskites_gap` | 1306 | GLLB-SC Eg | GLLB-SC |
| `mp_nostruct_20181018` | 83989 | PBE Eg, mpid | PBE |
| `wolverton_oxides`, `dielectric_constant` | 4914 / 1056 | PBE Eg | PBE |
| `matbench_mp_gap`, `mp_all_20181018` (`--heavy`) | 106k / 84k | PBE Eg + structures | PBE |
| `materials_project` (API key) | ~150k | PBE/r2SCAN Eg, internal VBM/CBM, hull energy | PBE |
| `literature_csv` | user | experimental Eg / VBM / CBM for device materials incl. organics | literature |

Method ranking for `materials_best`: experiment = literature (1) > HSE06 (2) > TBmBJ (3) > GLLB-SC (4)
> r2SCAN / OptB88vdW (5) > PBE (6) > ml_prediction (9).

### Literature CSV

Drop CSV files into `data/literature/`. A header template `TEMPLATE_literature.csv` is written on first build:

```
material,formula,phase_tag,material_class,role,gap,vbm,cbm,ip,ea,method,edge_reference,reference
```

`vbm`/`cbm` are eV vs vacuum (negative). `ip`/`ea` (positive) are accepted instead. Organic materials
without a stoichiometric formula (Spiro-OMeTAD, PCBM, ...) are stored as aliases plus their values.

## Schema

- `materials`: one row per (reduced formula, phase tag): family, anonymized formula, perovskite descriptors,
  Mulliken electronegativity.
- `records`: one row per measurement: `gap_ev`, `vbm_ev`, `cbm_ev`, `edge_reference`, `ip_ev`, `ea_ev`,
  `work_function_ev`, `method`, `method_rank`, `source`, `external_id`, `reference`, `structure_id`.
- `structures`: CIF + space group + density + volume per atom + energy above hull.
- `interfaces`: DFT band offsets between two JARVIS materials.
- `aliases`: common layer names -> formula / class / role.
- `features_composition`: cached matminer features (after `featurize`).
- `features_structure`: fast structure descriptors per CIF (after `featurize-structures`).
- `structure_proxy`: DFT-gap proxy predictions per structure (after `train-proxies`).
- `materials_best` (view): best-ranked gap and vacuum edges per material with record counts and spread.
- `build_log`: per-source status, counts and timing.
#   M a t e r i a l S t a c k  
 