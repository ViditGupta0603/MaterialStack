# MaterialStack — Complete Evaluation Guide (Start → End)

**Product name:** MaterialStack  
**Engine / package:** `materialstack` (Python)  
**Branch:** `cursor/materialstack-pipeline`  
**Purpose of this doc:** Kal evaluation ke liye ek hi jagah — *why, what, how, when, numbers, limits, demo commands*.

---

## 0. 60-second elevator pitch (yaad rakhna)

> MaterialStack predicts **band gap (Eg)** and **vacuum band edges (CBM/VBM)** for each layer in a solar / optoelectronic stack, then **computes** heterojunction type (I / II / III) from band overlap.  
> It **looks up** trusted experimental/HSE values from a large materials DB first; only if missing does it use **Magpie composition features + LightGBM**.  
> Junction type is **not** an ML classifier — it is physics from offsets.

**One number they will ask:** Eg MAE ≈ **0.46 eV** (cleaned experiment, GroupKFold).  
**One honest caveat:** composition-only cannot distinguish anatase vs rutile TiO₂; junctions within ~0.3 eV of a Type boundary are uncertain.

---

## 1. WHY — Problem & motivation

### Real-world problem
In perovskite / thin-film solar cells (and other optoelectronics) you stack layers:

- absorber (e.g. MAPbI₃)
- ETL (electron transport, e.g. TiO₂, SnO₂)
- HTL (hole transport, e.g. NiO, Spiro)

Device behaviour depends on **band alignment**:

| Quantity | Meaning |
|---|---|
| **Eg** | Band gap (eV) |
| **VBM** | Valence band maximum vs vacuum |
| **CBM** | Conduction band minimum vs vacuum |
| **CBO / VBO** | Conduction / valence band offsets between layers |
| **Type I / II / III** | How the two gaps nest (straddling / staggered / broken) |

Manually collecting Eg/VBM/CBM for every chemistry is slow. Cheap DFT (PBE) is systematically wrong vs experiment. Graph neural nets need crystal structures for every query — not always available.

### Thesis / design constraints (fixed early)
- Prefer **tabular composition features** (Magpie-style) as the ML fallback.
- Thesis named **LightGBM / Random Forest**; ruled out CGCNN/MEGNet for the *composition-only* path.
- Junction type = **deterministic** from edges, not learned.
- Validation must avoid leakage across chemically similar substitutions → **GroupKFold by element set**.

### Product goal
Input: one or more material names/formulas.  
Output: per-layer Eg, CBM, VBM + provenance (lookup vs ML) + junction type + uncertainty flag.

---

## 2. WHAT — System overview

```text
User formula / alias / stack
        │
        ▼
┌───────────────────┐
│  MaterialStack UI │  (React)  or CLI `predict` / `serve`
└─────────┬─────────┘
          ▼
┌───────────────────┐
│  FastAPI / CLI    │  materialstack.api / materialstack.predict
└─────────┬─────────┘
          ▼
   ┌──────────────┐     hit trusted method?
   │ materials_db │ ──yes──► return DB Eg / edges
   │   SQLite     │
   └──────┬───────┘
          │ miss / weak method
          ▼
   Magpie features → LightGBM Eg (2-stage)
                   → LightGBM δCBM (+ Butler–Ginley)
          ▼
   Junction Type I/II/III from CBO/VBO
```

| Layer | Tech |
|---|---|
| Product UI | **MaterialStack** — Vite + React + Framer Motion |
| API | FastAPI (`python -m materialstack serve`) |
| Engine | Python package `materialstack` |
| DB | `data/materials_db.sqlite` (~105k materials) |
| Models | `models/eg_model.joblib`, `models/edge_model.joblib` |
| Metrics | `models/metrics.json`, `train_clean_audit.json` |

---

## 3. WHEN — Build order (timeline of the project)

| Stage | What happened | Why |
|---|---|---|
| **1. Database** | Ingest matminer, JARVIS, Castelli, Borlido, aliases, optional MP | Need one place for gaps + edges + formulas |
| **2. Featurize** | Magpie composition features for all ~105k materials | ML needs numeric X |
| **3. Model bakeoff** | RF / HistGB / CatBoost / LightGBM / XGBoost on experiment slice | Choose primary booster |
| **4. Decision** | **LightGBM primary**, XGBoost baseline only | Thesis + speed; XGB only ~0.02 eV better |
| **5. Train (mixed)** | experiment+HSE+TBmBJ → MAE ~**0.60 eV** | Too noisy (fidelity mix) |
| **6. Clean + retrain** | experiment+literature + filters → MAE ~**0.46 eV** | Honest target |
| **7. Lookup-first** | DB before ML | Product accuracy ≠ CV MAE alone |
| **8. MaterialStack UI** | Frontend + API + performance page | Demo for evaluation |

**Models are frozen for the UI** — we did not retrain again for the frontend.

---

## 4. HOW — Database (Stage 1) in depth

### Scale (current)
| Item | Count |
|---|---|
| Materials | **~104,915** |
| Records | **~263,024** |
| Structures (CIF stored) | **~125,699** |
| Magpie feature rows | **104,915** |
| Structure *features* table | **0** (not featurized yet) |
| Interfaces (JARVIS) | **439** |
| Aliases | **64** |

### Important distinction (evaluation favourite)
- **~105k** = materials with **composition features** (can *query* / featurize).
- **~5.2k** = cleaned **experimental** labels used to *train* Model A.
- You cannot train supervised Eg on all 105k — most labels are cheap PBE / OptB88, not experiment.

### Method trust ranking (`METHOD_RANK`)
Lower rank = more trusted when picking `materials_best`:

| Rank | Method |
|---|---|
| 1 | experiment, literature |
| 2 | HSE06 |
| 3 | TBmBJ, DFT-surface-TBmBJ |
| 4 | GLLB-SC |
| 5 | r2SCAN, OptB88vdW, DFT-surface-OptB88vdW |
| 6 | PBE |
| 9 | ml_prediction |

### Major sources
| Source | Provides |
|---|---|
| `expt_gap`, `expt_gap_kingsbury`, `foundry_ml_exp_bandgaps` | Experimental Eg |
| `borlido_expt` | Experimental Eg **hold-out** stress set |
| `snumat` | HSE06 + PBE + structures |
| `jarvis_dft_3d/2d` | OptB88vdW, TBmBJ, some HSE |
| `jarvis_surfacedb` | Vacuum VBM/CBM (surfaces) |
| `jarvis_interfacedb` | Interface VBO offsets (formulas often NULL; **no junction_type labels**) |
| `castelli_perovskites` | GLLB-SC Eg + vacuum edges |
| `mp_nostruct_20181018`, oxides, dielectric | PBE gaps |
| `literature_csv` | User device materials / organics (slot; fill as needed) |
| `aliases` | MAPbI₃ → formula, PC60BM, etc. |

### Families in full DB
intermetallic, oxide, halide, chalcogenide, pnictide, perovskite_ABX3, double_perovskite, organic, hybrid_perovskite.

### Families in **Model A training** (cleaned, n≈5225)
| Family | ≈n |
|---|---|
| chalcogenide | 1652 |
| intermetallic | 1432 |
| oxide | 904 |
| pnictide | 493 |
| halide | 389 |
| perovskite_ABX3 | 203 |
| double_perovskite | 149 |
| organic / hybrid | ~3 total |

→ Organics almost **not** learnable from ML; need literature CSV / lookup.

### Interface-type data?
**No usable Type I/II/III labels.**  
439 JARVIS interface rows: `junction_type` all NULL, formulas mostly NULL, some `vbo_ev` only.  
So we **compute** junction type; we do **not** train a junction classifier.

---

## 5. HOW — Features (inputs to ML)

### Composition features (`features_composition`)
Built with **matminer**:
- Magpie element-property statistics
- Stoichiometry norms
- Valence orbital fractions
- IonProperty
- AtomicOrbitals (HOMO/LUMO character → one-hots)
- BandCenter
- Extra: family one-hots, Mulliken χ, perovskite tolerance factors when applicable

≈ **180 numeric columns** after encoding.

### Why Magpie / composition-only?
- Works for **any formula** without a CIF.
- Matches thesis “composition fallback”.
- **Cost:** same feature vector for polymorphs (anatase = rutile as composition).

### Structure features
Code path exists (`features_structure`) but **not filled** in current DB build. Future lever for lower MAE.

---

## 6. HOW — Models (math you should be able to write on board)

### Model A — Band gap (two-stage LightGBM)

**Why two-stage?** Gaps are bimodal: many metals with Eg≈0, and semiconductors with Eg>0. One regressor smears both.

1. **Classifier:** metal vs non-metal  
   - Label: metal if `Eg ≤ 0.001 eV`
2. If metal → **Êg = 0**
3. If non-metal → **regressor** on transformed target:

\[
t = \log(1 + E_g),\qquad \hat{E}_g = e^{\hat{t}} - 1
\]

(`log1p` / `expm1`) — compresses wide-gap skew, avoids `log(0)`.

**Primary:** `LGBMClassifier` + `LGBMRegressor`  
**Baseline in metrics only:** XGBoost same architecture.

### Model B — Band edges

Butler–Ginley baseline from Mulliken electronegativity χ and gap:

\[
\mathrm{CBM}_{BG} = -\bigl(\chi - E_g/2\bigr),\qquad
\mathrm{VBM}_{BG} = \mathrm{CBM}_{BG} - E_g
\]

(as implemented in `butler_ginley_edges`)

Learn residual:

\[
\delta\mathrm{CBM} = \mathrm{CBM}_{\mathrm{true}} - \mathrm{CBM}_{BG}
\]

At inference:

\[
\mathrm{CBM} = \mathrm{CBM}_{BG} + \widehat{\delta\mathrm{CBM}},\qquad
\mathrm{VBM} = \mathrm{CBM} - E_g
\]

Trained on vacuum-referenced edges (Castelli + JARVIS surfaces after cleaning).

### Junction type (NOT ML)

From two layers’ CBM/VBM vs vacuum:

- **Type I** — straddling (one gap fully contains the other)
- **Type II** — staggered (partial overlap; typical for charge separation)
- **Type III** — broken gap (no overlap)

Implemented in `classify_junction`. Flag **UNCERTAIN** if offset margin &lt; ~0.3 eV or ML edges involved near boundary.

### Why LightGBM not XGBoost as primary?
Bakeoff on experiment-only slice: XGB ≈ 0.457 eV MAE, LGBM ≈ 0.478 eV (+0.021 eV).  
Gap &lt; label noise and &lt;&lt; junction-boundary uncertainty (~0.3 eV). LGBM matches thesis, trains faster, already in requirements. XGB kept in `metrics.json` as baseline.

---

## 7. HOW — Training & cleaning

### Validation
**GroupKFold (k=5) by element-set JSON** so substituted variants of the same chemistry don’t leak train→test.

### Metrics (report these)

| Metric | LightGBM (current) | Notes |
|---|---|---|
| Eg non-metal **MAE** | **0.459 eV** | Headline |
| Eg RMSE | 0.714 eV | |
| Eg R² | 0.745 | |
| Metal accuracy / F1 | 0.925 / 0.919 | |
| XGB Eg MAE | 0.458 eV | Baseline |
| Edge CBM MAE | **0.196 eV** | |
| Train n (Eg) | 5,225 | after clean |
| Train n (edges) | 809 | after clean |
| Wall time | ~117 s | last train |
| Borlido-only holdout | MAE ~1.21 eV (n=41) | OOD stress; never in fit |

### Why MAE was once ~0.60 eV
First `train` used **experiment + HSE06 + TBmBJ** (~26k rows). Mixed DFT≠experiment → higher MAE.  
Cleaning + **experiment/literature only** brought MAE back to ~0.46 eV (matches bakeoff).

### Cleaning rules (non-destructive — SQLite not deleted)
| Rule | Effect |
|---|---|
| Method allow-list | Default: experiment, literature |
| Expt scatter | Drop if IQR>0.5 or max−min>1.0 eV |
| Gap bounds | Non-metals Eg ≤ 12 eV |
| DFT conflict | Drop if \|Eg_exp − nearest HSE/TBmBJ\| > 2 eV |
| Feature NaNs | Drop if >50% Magpie NaN |
| Edge consistency | \|(CBM−VBM)−Eg\| ≤ 0.35 eV |
| δCBM outliers | \|\δCBM\| ≤ 5 eV |

Audit file: `models/train_clean_audit.json`.

### Train commands
```bash
python -m materialstack train
python -m materialstack train --eg-methods experiment,HSE06,TBmBJ   # old noisy mix
```

---

## 8. HOW — Inference (lookup-first) — **this is the product answer to “0.1 eV”**

CV MAE ~0.46 eV is the **ML fallback** quality.  
For materials **already in DB with experiment/HSE**, system uses those values → error ≈ label noise, not 0.46.

### Pipeline (`resolve_layer`)
1. Resolve alias (e.g. MAPbI₃).
2. Parse formula → `materials_best`.
3. If best gap `method_rank ≤ 2` (experiment/literature/HSE06) → **use DB Eg** (`trusted_gap=True`).
4. If vacuum CBM/VBM exist → use DB edges; else ML δCBM or Butler–Ginley.
5. Unknown formula → featurize on the fly → ML.
6. Stack ≥2 → compute Type I/II/III.

### Demo numbers (smoke-tested)
| Query | Eg | Source |
|---|---|---|
| TiO₂ | **3.3 eV** | lookup:experiment |
| MAPbI₃ | **2.64 eV** | lookup:HSE06 |
| TiO₂ \| MAPbI₃ | Type **II** | computed |

```bash
python -m materialstack predict TiO2
python -m materialstack predict TiO2 MAPbI3
```

---

## 9. MaterialStack frontend

### What it shows
1. **Hero** — brand MaterialStack, one value prop, CTA  
2. **Lab** — enter stack formulas → Eg/CBM/VBM + junctions  
3. **Performance** — full training report from `metrics.json`  
4. **Pipeline** — DB → clean → LightGBM → lookup-first  

### Run
```bash
cd frontend && npm install && npm run build && cd ..
python -m materialstack serve          # http://127.0.0.1:8000
```

Dev: API on :8000, Vite on :5173 with `/api` proxy.

### API
- `GET /api/health`
- `GET /api/stats`
- `GET /api/metrics`
- `POST /api/predict` `{"materials":["TiO2","MAPbI3"]}`

---

## 10. Limits (say these before they catch you)

1. **Composition-only ceiling** — global Magpie+LGBM ≈ 0.35–0.50 eV; 0.1 eV not realistic without structure / domain / lookup.
2. **Polymorphs** — TiO₂ anatase vs rutile identical Magpie vector.
3. **Organics** — almost no train labels; aliases alone don’t give Magpie HOMO/LUMO until literature CSV.
4. **Vacuum edges scarce** — Model B n≈809 after clean.
5. **Interfaces table** ≠ junction labels.
6. **Borlido holdout hard** (~1.2 eV) — OOD; don’t cite as headline.
7. **Junction uncertainty** — if CBO/VBO within ~0.3 eV of boundary (or ML edges), mark uncertain.

### Paths to lower MAE (if asked “next?”)
1. Keep lookup-first (already done)  
2. Structure features / ALIGNN when CIF exists  
3. 2–3 family specialists (oxide, chalcogenide, perovskite) — not 9 tiny models  
4. Literature CSV for device stack materials  
5. Multi-fidelity: learn Eg_exp − Eg_DFT residual  

---

## 11. Repo map (where is the code?)

| Path | Role |
|---|---|
| `materialstack/build_db.py` | Ingest orchestration |
| `materialstack/db.py` | Schema, `materials_best`, lookup |
| `materialstack/features.py` | Magpie featurize |
| `materialstack/clean.py` | Training-label filters |
| `materialstack/models.py` | LightGBM train / CV / persist |
| `materialstack/chem.py` | Formula clean, χ, Butler–Ginley |
| `materialstack/predict.py` | Lookup-first + junctions |
| `materialstack/api.py` | FastAPI for MaterialStack |
| `materialstack/cli.py` | `build-db`, `train`, `predict`, `serve`, … |
| `frontend/` | MaterialStack React UI |
| `models/*.joblib` | Frozen trained models |
| `models/metrics.json` | CV numbers |
| `data/materials_db.sqlite` | Database |

---

## 12. Likely viva Q&A (short answers)

**Q: Why not neural nets / CGCNN?**  
A: Thesis composition fallback; no structure required for every query; LGBM fits tabular Magpie.

**Q: Why is MAE 0.46 not 0.1?**  
A: Composition-only global limit + experimental noise; product uses lookup-first so known materials aren’t stuck at 0.46.

**Q: Why GroupKFold?**  
A: Prevent leakage when many materials share the same element set / substitutions.

**Q: Is junction type predicted by AI?**  
A: No — computed from CBM/VBM overlap.

**Q: What is δCBM?**  
A: Correction to Butler–Ginley CBM using vacuum-labeled training edges.

**Q: 105k materials but train on 5k?**  
A: Only ~5k trusted experimental labels after cleaning; rest are features or weaker DFT.

**Q: LightGBM vs XGBoost?**  
A: ~0.02 eV MAE; chose LGBM for thesis + speed + simplicity; XGB in metrics.

**Q: Show me a demo.**  
A: `predict TiO2` → 3.3 eV experiment lookup; UI at `:8000`.

---

## 13. Demo checklist for tomorrow

1. Open http://127.0.0.1:8000 (or `python -m materialstack serve`).  
2. Lab: `TiO2` then `TiO2` + `MAPbI3` — show trusted vs ML tags + Type II.  
3. Performance tab — point to 0.459 eV MAE, cleaning audit, edge 0.196 eV.  
4. Say the pitch (section 0) + one limitation (polymorphs / 0.3 eV uncertainty).  
5. Optional CLI: `python -m materialstack stats` / `lookup TiO2`.

---

## 14. One-page formula sheet

```text
Metal:     Eg ≤ 0.001 eV → Êg = 0
Nonmetal:  t = log(1+Eg),  Êg = exp(t̂)−1

Butler–Ginley:
  CBM_BG = -(χ − Eg/2)
  δCBM   = CBM_true − CBM_BG
  CBM    = CBM_BG + δ̂CBM
  VBM    = CBM − Eg

Lookup rank: experiment/literature(1) > HSE(2) > TBmBJ(3) > … > PBE(6)

CV: GroupKFold(k=5) by element set
Headline: Eg MAE = 0.459 eV | Edge CBM MAE = 0.196 eV
```

---

*Document generated for MaterialStack evaluation prep. Models frozen at cleaned experiment LightGBM checkpoint.*
