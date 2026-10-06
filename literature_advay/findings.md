# Findings log

Newest first. Each entry says how the number was obtained.

---

## 2026-10-06: Simplified pipeline (CHANGELOG #26)

Numbers from `python validate.py` → `results/metrics.csv`.

### F23. One model and CSV data: what changed in accuracy
- **Band gap:** grouped 5-fold CV MAE 0.397 eV, R² 0.86 on 2,509 semiconductors (one LightGBM + DFT hint).
  Old: 0.465 eV composition-only / 0.428 eV structure-aware on 2,912 non-metals. Not a like-for-like
  comparison: the new labels drop Foundry-ML and add rule R6, and the metal classifier (which put 3.5 %
  of non-metals at 0 eV) is gone. Where a DFT gap exists the model beats it: 0.42 vs 0.54 eV (hybrid),
  0.43 vs 0.96 eV (GGA). Gaps of 1–3.5 eV: 0.36 eV.
- **DFT hint polymorphs:** the median over all JARVIS polymorphs gave Si 0.13 eV (39 entries, mostly
  hypothetical metallic allotropes); restricting to polymorphs within 50 meV/atom of the lowest gives
  1.19 eV. SNUMAT is all ICSD structures, so all its entries are used.
- **Band edges, measurement hidden:** Butler–Ginley VBM MAE 1.08 eV, biased 0.66 eV too deep (35
  measured materials, mostly halide perovskites and III-V); hybrid-DFT surfaces 0.52 eV (5 oxides).
  These set σ_VBM = 1.3 / 0.65 eV in predict.py.
- **21 measured offsets:** 0.45 eV (old 0.41 with JARVIS interfaces, 0.36 vacuum path). The loss comes
  from dropping JARVIS GGA slabs: CdS, CdSe, ZnSe, ZnS, AlAs, AlP, SiC and InN now use Butler–Ginley.
  Measured IEs for these II-VI/III-V layers would recover it.
- **Gold stacks:** type 61 % (old 59 %; always-Type-II 61 %), ΔEc MAE 0.51 eV (old 0.53; zero-offset
  baseline still 0.45), ΔEv 0.58 eV (old 0.56), sign agreement ΔEc 80 % / ΔEv 83 % (old 77 / 88 %),
  confident calls right 82 % (n = 11; old 75 %, n = 20, with smaller σ values). Without the curated
  measured edges: ΔEc 0.80 eV, signs 56 / 65 %: the measured data carry most of the offset accuracy.

## 2026-10-05: Database audit and measured-data curation

Scripts: `literature_advay/scripts/database_audit.py`, `find_measured_levels.py`; abstracts checked via
OpenAlex and Europe PMC full text. Only peer-reviewed measurement papers or reviews by the measuring
groups were used; every value was read in the paper's own text (abstract or body). Gold-file papers
were not used.

### F22. The gold file disagrees with direct measurements by 0.4–0.8 eV for common layers
- The gold edges are SCAPS-1D input parameters. For layers where we now have photoemission data:
  TiO2 gold CBM −4.0/−4.1 vs −4.8 eV (anatase IP 7.96 eV, Kashiwaya 2018); SnO2 −4.0 vs −4.8 eV (Klein
  2010); C60 VBM −5.6 vs −6.4 eV (Olthof/Kahn PRL 2012; Schulz 2015); CuSbS2 −4.5/−6.0 vs −3.43/−4.98 eV
  (Whittles 2017); SnS VBM −5.3 vs −4.71 eV (Whittles 2016); MAPbI3/CsPbI3 ~0.45 eV shallower than
  Tao 2019 UPS/IPES.
- Part of the spread is genuine: oxide IPs depend on the surface by up to 1 eV (ZnO 6.9–7.8 eV, SnO2
  7.9–8.9 eV; Klein 2010), and halide-perovskite IEs depend on stoichiometry and on how the valence-band
  onset is read (linear vs log scale, ~0.3–0.5 eV; Schulz, Cahen & Kahn, Chem. Rev. 2019).
- Consequence: type accuracy against the gold file has a ceiling set by the reference. Report it as
  "agreement with SCAPS literature parameters" and use measured interface offsets as the primary
  accuracy benchmark.

### F21. Audit: what the database got wrong and what was added
- Coverage before: measured edges for 9/24 absorbers, 0/22 ETLs, 0/18 HTLs, 0/7 organics; i-ZnO, IGZO,
  CBTS and organic layers could not be resolved (C60 parsed as carbon).
- Impossible records: 15 removed by documented rules (`build_db.sanitize`, audit file
  `data/sanitize_removed_records.csv`): negative gap (1), gaps > 15 eV outside noble-gas solids (10,
  e.g. 23 eV "gaps" for metallic Na, GaAs 15.1 eV), measured gaps > 3× the other measured values (4:
  InSe 13.0/13.2, Si 4.19, Ba(CuS)2 9.8). BN 14.5 eV is not caught (other measured BN values vary
  with polymorph); the consensus label handles it in training. A first version that compared with DFT would have deleted correct
  values (MnO 3.6 eV, diamond 5.5 eV); the reference is now other measured values only.
- Added 19 measured levels (PCBM, C60 ×2, Spiro ×2, MoO3, TiO2, ZnO ×2, SnO2 ×2, In2O3, NiO ×2, CuI,
  CuSCN, SnS, CuSbS2, Cs2SnI6) and 2 measured interface offsets (CdS/CuInSe2, ZnO/CdS), the latter
  preferred over JARVIS GGA interfaces (JARVIS ZnO/CdS had the wrong VBO sign: −0.48 vs +1.4 eV).
- Effect on the gold set (tool): VBM MAE 0.56 → 0.47 eV, CBM 0.48 → 0.45, ΔEc MAE 0.65 → 0.53,
  ΔEv 0.61 → 0.56, ΔEc sign agreement 74 → 77%, ΔEv 86 → 88%, coverage 91 → 100% (organics), type
  accuracy 52 → 59% (always-Type-II 61%); clear junctions (both offsets ≥ 0.2 eV, n = 22) 45%.
  Independent benchmark (21 measured offsets) unchanged: 0.41 eV.
- Still unverified / missing: PTAA, P3HT, PEDOT:PSS, Cu2O, CdS, CIGS, CZTS(Se), Sb2Se3, Cs2AgBiBr6,
  Cs2CuBiBr6, CBTS, IGZO, CeO2, MoTe2 — no value found that could be read in a good paper's text.

## 2026-10-05: Verification against the multilayer gold standard (verif/)

Script: `verif/verify_gold_standard.py`; results in `verif/results/`.

### F20. Layer values are good; junction types are not better than guessing Type II
- Gold set: 50 device stacks from SCAPS-1D papers (tier A: 42, C: 5) and 3 textbook heterojunctions (tier E). Gold values are simulation inputs chosen from literature, not new measurements.
- Entire tool vs model only (MAE): band gap 0.23 vs 0.33 eV; VBM 0.56 vs 0.65; CBM 0.54 vs 0.75; ΔEv 0.61 vs 0.65 (zero-offset baseline 0.85); ΔEc 0.65 vs 0.77 (zero baseline **0.47**).
- Junction type accuracy: tool 52%, model 55%, **always-Type-II baseline 62%**. ΔEc sign right 73% (tool) vs 55% (model).
- The tool flags 96% of these junctions for DFT; the 4 it is confident about are all correct.
- Largest VBM errors are the hole-transport layers again (NiO −7.76 vs −5.40, CuSCN, CuI, V2O5, BiFeO3): critique A10. Some band-gap disagreements reflect the gold file's simulation inputs (e.g. Cs2AgInBr6 1.47 eV in the gold file vs 2.72 from the database).
- By tier (tool): band gap 0.20 / 0.37 / 0.07 eV and type accuracy 53 / 40 / 67% for A / C / E.
- Model on materials outside its training set (14): band gap 0.81 eV, VBM 0.88 eV.

---

## 2026-10-03 (later): Edge model study (B1–B4, plan step 2.1)

Script: `scripts/edge_model_study.py`.

### F19. For offsets, simple beats flexible: blend slab VBMs with Butler–Ginley; no ML edge model
- Out-of-fold scores (VBM vs slab / DFT interface offsets / measured offsets, eV): Butler–Ginley 1.38 / 1.00 / 0.58; BG + ridge 1.18 / 0.98 / 0.58; BG + LightGBM 0.86 / 1.09 / 0.73; + structure 0.88 / 1.11 / 0.70; LightGBM direct 0.80 / 1.11 / 0.84; slab lookup — / 1.33 / 0.91.
- Flexible models learn the slab values' surface-specific noise, which does not carry over to interfaces.
- Blend w·slab + (1−w)·BG: w = 0.25 → 0.80 / 0.47; 0.5 → 0.82 / 0.58; 0.75 → 1.02 / 0.78. Shipped w = 0.3 from inverse-variance weighting.
- Pipeline after the change: DFT interfaces 0.79 eV (type agreement 64%), measured 0.48 eV (vacuum path).
- Failure mode not covered by the benchmarks: d-/s²-derived VBMs (Cu(I), Ni(II), Sn(II)) predicted 1–2 eV too deep (critique A10).

---

## 2026-10-03 (later): Surface selection study (A9, plan step 1.1)

Script: `scripts/surface_policy_study.py`.

### F18. No surface or polymorph selection rule makes slab vacuum alignment competitive
- Policies scored on 21 measured offsets / 327 DFT interface offsets (MAE, eV): median of all surfaces 0.91 / **1.33**; lowest surface energy 1.29 / 1.61; non-metallic slabs 1.01 / 1.39; more stable half 1.01 / 1.43; shallowest 1.09 / 1.79; ground-state polymorph + median 1.01 / 1.58; ground-state + lowest surface energy **0.85** / 1.50; ground-state + non-metallic 1.00 / 1.45. Butler–Ginley: 0.58 / 1.00.
- Plane-matching to each measured interface (as the paper does) gives 0.87 eV, not the paper's 0.47: the paper's 8 hand-picked pairs used specific slab variants.
- Formulas mix slabs of different bulk polymorphs (GaAs: zincblende JVASP-1174 and wurtzite JVASP-8185, the latter giving VBMs of −7.1 and −8.3 eV), but restricting to the ground-state polymorph does not help on balance.
- Decision: keep the median (simplest; best on the larger benchmark). Slab VBMs are noisy for alignment; step 2.1 tests combining them with Butler–Ginley/ML rather than trusting them outright.

---

## 2026-10-03 (later): Junction benchmark (C3)

Command: `python -m materialstack evaluate junctions`.

### F17. Vacuum alignment from median slab VBMs is the weak link for junctions
- 21 measured valence-band offsets (InterMat Table 2, magnitudes): direct interface DFT 0.22 eV; paper's vacuum alignment with specific (110) surfaces 0.47 eV (8 systems); Butler–Ginley 0.58 eV; **our pipeline 0.91 eV** (0.82 on the same 8).
- Worst cases come from mixing surfaces: GaN/ZnO 2.12 vs 0.70 measured, ZnSe/InP 2.56 vs 0.41, Si/AlN 0.38 vs 3.50.
- vs 321 JARVIS DFT interfaces: ΔEv MAE 1.31 eV (zero-offset baseline 0.86), type agreement 44%, sign agreement 71%.
- Even the paper's own vacuum-alignment and interface values differ by up to 1.7 eV (CdS/Si 3.22 vs 1.48): interface dipoles are large for some pairs (B6).
- Implications: (1) choose surfaces better than a median over all slabs (A9); (2) use directly computed interface offsets where they exist (B8).

---

## 2026-10-03 (later): Consensus labels (B7)

### F16. Cleaner labels improve the band-gap model by 0.09 eV on the fixed benchmark
- Compilations duplicate each other heavily: 5,850 copied values across 4,771 materials (Kingsbury and Foundry largely copy Zhuo), so naive agreement counts are inflated.
- Range-based cleaning dropped Si, GaAs, ZnO, CdTe over single bad reports (Si: 0.60 and 4.19 eV alongside 1.12/1.17; GaAs: one 15.11 eV entry).
- DFT tie-breaking for ambiguous materials was tested and rejected: it fixes ZrO2/La2O3/Dy2O3 but mislabels SnO2 (stored HSE06 2.27 eV vs real 3.6).
- DFT-conflict rule (|expt − HSE/TBmBJ| > 2 eV) catches a mix of experimental errors (CdPS3 12 eV, Ba(CuS)2 9.8 eV) and lanthanide oxides where DFT fails; kept as a conservative filter.
- Borlido benchmark, formula only: MAE 0.71 → 0.62 eV, within 0.5 eV 63% → 69%, false-metal rate 5.5% → 4.0%. On materials with HSE06 data the ML now equals HSE06 vs experiment (0.61 eV).

---

## 2026-10-03 (later): Honest band-gap benchmark (C2)

Command: `python -m materialstack evaluate gap` → `models/evaluation.json`.

### F15. Formula-only ML is better than PBE and close to HSE06 on Borlido
- All 453 Borlido materials, grouped 5-fold by element set (each fold's model trained without that fold's chemistry).
- Formula only: non-metal MAE 0.71 eV, R² 0.53, Spearman 0.82, 63% within 0.5 eV. With structure (no DFT): 0.63 eV, Spearman 0.86, 70%.
- Same materials, stored DFT vs Borlido: PBE 1.13 eV (bias −1.09), OptB88vdW 1.26, TBmBJ 0.62, HSE06 0.61. ML on the same materials 0.70–0.79.
- So as a pre-screen the ML is better than cheap (PBE) DFT and about 0.1–0.17 eV behind hybrid DFT.
- 5.5% of real non-metals are predicted to be metals (screening false rejections), e.g. AlPO4, CaCO3, BaSnO3.
- Heavy error tail: solid noble gases (no analogues), very wide-gap fluorides/oxides predicted several eV low.
- Earlier "Borlido holdout 1.21 eV" (41 leftover materials) is superseded.

---

## 2026-10-03 (later): Honest edge evaluation (C1)

Command: `python -m materialstack evaluate edges` → `models/evaluation.json`.

### F14. The deployed edge model is worse than a constant
- Reference: median vacuum-aligned slab VBM for 155 bulk semiconductors (JARVIS, DFT). Gap fed in = the pipeline's own gap (95 trusted lookups, 60 ML).
- MAE: constant 1.81 eV · Butler–Ginley 1.37 eV (bias −0.70, Spearman 0.68) · deployed model **3.29 eV** (bias **+2.99**, Spearman 0.69, 3% within 0.5 eV).
- The model reproduces the old wrong energy scale (F1). Rebuild target: beat Butler–Ginley first.
- Same-material surfaces differ by a median 1.29 eV (50 materials with several surfaces): a floor for composition-only edge prediction.

---

## 2026-10-03 (later): Surface data verified against InterMat Table 1

Script: `scripts/intermat_table_check.py`.

### F13. Corrected surface reading matches the published values
- InterMat Table 1 gives OptB88vdW work function φ and electron affinity χ for 19 non-polar semiconductor slabs. Their χ = IP − **bulk TBmBJ gap** (back-calculated; e.g. Si(111) 5.39 − 1.28 = 4.11 vs 4.10 published).
- With that definition our IP (= `avg_max − surf_vbm`) reproduces the published χ within about 0.05 eV for 15 of 21 matched rows (Si ×3, Ge, SiGe, GaAs, InAs, AlSb, GaSb, GaN, BP, InP, CdSe, ZnSe, ZnTe). Mismatches (AlN, BN, GaP, one BP variant) are different calculation variants of the same surface, not a pattern.
- The 2–9 eV work-function filter removes exactly the two broken rows in the table (SiC(001) φ 1.7 eV; GaN(100) φ 34 eV).
- `noDP` (no dipole correction) vs dipole-corrected runs of the same surface agree within 0.02 eV (AlSb, GaSb, GaP, InP); no evidence `noDP` slabs are worse for these non-polar surfaces.
- φ itself is not a good check for semiconductor slabs (the Fermi level can sit anywhere in the gap); φ mismatches of up to 2 eV coexist with matching IPs.
- Conclusion: the loader fix (F1) is correct; remaining scatter is data quality (some published slab IPs are far from experiment, e.g. CdSe IP 8.3 eV) plus Butler–Ginley's crudeness.

---

## 2026-10-03: Data source and edge-model audit

Setup: DB as built on 2026-10-02; models retrained with `train --no-xgb-baseline`
(Eg MAE 0.451 eV composition-only, 0.419 eV structure-aware; edge CBM MAE reported 0.184 eV).
Scripts: `scripts/edge_analysis.py`, `scripts/source_credibility.py`, `scripts/surface_vacuum_check.py`.

### F1. JARVIS surface edges are stored without vacuum alignment (bug)
- `materialstack/sources/jarvis_sources.py` (`surfacedb`) stores `surf_vbm`/`surf_cbm` as vacuum-referenced, but these are on the slab's internal energy scale. InterMat defines IP = E_vac − VBM with E_vac = `avg_max`.
- App output before fix: Si VBM **+0.72 eV** (above vacuum), GaAs −1.97, ZnO −2.64, CdTe −2.47 (expected about −5.2, −5.5, −7.6, −5.8). Si/GaAs is classified **Type III**.
- `surf_vbm − avg_max` gives Si −4.9 to −5.4, GaAs(110) −5.7, GaN −6.2 to −7.5, CdTe about −6.6: physically sensible.
- About 120 of 607 surfaces have broken vacuum levels (`avg_max` up to 167 eV); filter with `avg_max < 12` and `2 < avg_max − efermi < 9` (485 survive).
- Stored work function (`−efermi`) has the same problem (Si shows −0.72 eV).
- Status: **fixed 2026-10-03** (CHANGELOG #1); reading verified against the paper in F13.

### F2. Castelli band edges are the Butler–Ginley formula
- CBM − BG(χ, Eg): median |diff| 0.028 eV, 82% within 0.1 eV, corr 0.95 (18,928 rows; 96% metals).
- 683 of 809 edge-model training materials are Castelli, where Butler–Ginley alone scores 0.078 eV and the ML model 0.106 eV. The headline 0.18 eV edge MAE mostly measures reproduction of a formula.
- Castelli hypothetical "InInO3"/"GaGaO3" supply the edges shown for In2O3 and Ga2O3.
- Status: **open**: drop Castelli as edge labels and lookups.

### F3. Edge model on real DFT surfaces is no better than a constant
- On jarvis_surfacedb rows (n=126, wrong energy scale per F1): ML 0.608 eV, constant-δ baseline 0.544 eV, Butler–Ginley alone 3.93 eV.
- Re-check after fixing F1.

### F4. Train/inference gap mismatch
- Edge model is trained and scored using the label's own DFT gap (OptB88vdW / GLLB-SC). At prediction time it gets an experimental/HSE/ML gap; these differ by 1.18 eV on average (n=140).
- Scored the way the app uses it: CBM MAE 0.61, **VBM MAE 0.69 eV** (vs 0.29 eV with the label gap). Contaminated by F1.
- ~~`VBM = CBM − Eg` sends all of the gap opening to the VBM.~~ **Corrected 2026-10-03:** wrong. The baseline CBM = −χ + Eg/2 already splits a gap change symmetrically (+½ to CBM, −½ to VBM), and the measured shifts agree (VBM −0.43 eV, CBM +0.32 eV). The real issue is the train/inference gap mismatch above, plus surface-scale labels (F1). The symmetric split itself is an assumption to test against Hinuma 2014.

### F5. Surface dependence is a large irreducible error for bulk-only features
- Same material, different surfaces: VBM range median 1.39 eV (75th pct 2.27) across 142 materials. Measured on uncorrected values; recompute after F1.

### F6. Method mixing in edge training
- `load_edge_frame` takes medians over all rows sharing the top source weight, mixing OptB88vdW and TBmBJ CBMs of the same material. Likely why only 126 of 322 surface materials survive cleaning. Unverified root cause.

### F7. Interface data unused; vacuum alignment vs direct DFT offsets
- 425 interfaces have numeric offsets. The other 168 store an empty dict `{}`: JARVIS computed no offset for them (checked 2026-10-03), so nothing is lost.
- Plane-matched vacuum alignment from surfaces vs interface offsets (n=371): corr 0.42, MAE 1.51 eV; predicting 0 gives 0.87 eV. The paper reports 0.45 eV vs experiment for vacuum alignment on 8 interfaces, so either our matching/sign handling is off or many interfaces have large interface dipoles. Needs follow-up with the paper's convention (positive ΔEv at A/B means VBM higher in B).

### F8. Source credibility (experimental and computed gaps)
- Experimental sources: `expt_gap` vs `expt_gap_kingsbury` 95% identical (duplicate); Borlido vs Zhuo MAE 0.22 eV (label noise floor).
- Computed vs experiment MAE: SNUMAT HSE06 0.62, JARVIS TBmBJ 0.81, PBE/OptB88vdW 1.0–1.15 (about −1 eV bias), Castelli GLLB-SC 1.96 and Wolverton PBE 1.90 (hypothetical structures).
- MAPbI3 gets 2.64 eV from `jarvis_halide_perovskites` HSE06 (experiment about 1.6 eV); hybrid perovskites should use experimental values only.

### F11. Dead code / leftover files audit (vulture ≥60% confidence + manual checks)
- Python package is lean (about 3.7k lines). Real leftovers only: unused import `Iterable` (`db.py:8`), unused `transaction()` helper (`db.py:150`), unused parameter `source_tag` in `jarvis_sources._bulk`, unread `CleanName.dopant_tag`/`parseable` fields, `borlido_expt` registered twice in `LOADERS`, a redundant first `GroupKFold` in `_grouped_cv_edge`.
- Leftover files: `catboost_info/` (CatBoost isn't used anywhere), `data/exports/` (no code produces or reads it), Vite template assets `frontend/public/favicon.svg`, `icons.svg`, `src/assets/hero.png` (not referenced), template `frontend/README.md`.
- Vulture false positives: FastAPI route functions (registered by decorator), `LayerResult.polymorph_policy` (read via `to_dict`), `cv_metrics_` (saved in model bundles).
- Not waste: XGBoost baseline (comparison for the report), `--heavy` sources (optional).

### F12. Approach assessment
- Keep: lookup-first; two-stage LightGBM for Eg (gradient boosting stays competitive on about 5k experimental labels, e.g. Ye 2025); DFT-proxy features (a form of multi-fidelity learning).
- Change: edge formulation. Predict VBM (−IP) per structure (later per surface) from vacuum-referenced surface data (fixed JARVIS + Kiyohara oxides), then CBM = VBM + best Eg. Replaces the correction-to-Butler–Ginley model trained on formula-derived labels.
- Add: per-layer uncertainty (quantile or conformal) → junction-type probabilities instead of a fixed 0.3 eV flag; later a pairwise correction trained on interface offsets (JARVIS direct interface calculations + experiment) to go beyond Anderson's rule.
- Lowest priority: switching to graph neural networks; gains are mainly on large DFT datasets, not our small experimental ones.

### F10. Inputs are formula-only; band edges never see structure
- CLI/API/UI accept only formula or alias strings (`api.py` `PredictRequest.materials: list[str]`); no CIF, MP or JARVIS ID.
- Lookup is per formula (polymorphs pooled). The ML gap uses DB-stored structures when present (code-chosen polymorph: `ground_state` or `mean`), otherwise composition only.
- Edge model (`load_edge_frame`) uses composition features + gap + Butler–Ginley only, although structure/surface drive much of edge variation (F5) and the state of the art uses structure (InterMat graph neural network 0.26 eV) or surface plane (Kiyohara 2024, about 0.22 eV).
- Options: accept structure input (CIF / database ID, optional Miller index); add the existing `features_structure` columns to the edge model; surface-resolved labels to learn orientation.

### F9. Coverage of device layers
- No experimental or correctly-computed edges for TiO2, SnO2, NiO, Cu2O, MoO3, WO3, halide perovskites, CIGS, CZTS, Sb2Se3. `literature_csv` is empty.
