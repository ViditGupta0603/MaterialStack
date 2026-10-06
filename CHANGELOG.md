# Changelog

Changes to the MaterialStack pipeline, newest first. Each entry names the issue it addresses
(IDs from [`literature_advay/critique.md`](literature_advay/critique.md)), what changed, how it was
verified, and the effect on results. Rebuild/retrain steps needed after pulling a change are listed
under **Requires**.

---

## #1 · 2026-10-03 · Surface band edges put on the vacuum scale (A1)

**Problem.** The JARVIS surface loader stored slab VBM/CBM on each calculation's internal energy scale
instead of relative to vacuum. Si showed VBM = +0.72 eV (above vacuum) and Si/GaAs came out Type III.
The stored work function had the same error.

**Changes**
- `materialstack/sources/jarvis_sources.py` (`surfacedb`): VBM, CBM and Fermi level are shifted by the
  vacuum level `avg_max` (IP = E_vac − VBM, φ = E_vac − E_F, as defined in InterMat). The vacuum level
  is kept in `extra.vacuum_level`.
- Slabs with a work function outside 2–9 eV are skipped: their vacuum level never converged
  (e.g. E_vac = 27 or 167 eV). This drops 121 of 607 surfaces.
- `materialstack/build_db.py`: `build-db --only jarvis_surfacedb` (and `jarvis_interfacedb`) now loads
  the JARVIS ID → bulk-gap map from the existing database. Before, it was only filled when
  `jarvis_dft_3d` was in the same run, so the derived TBmBJ surface rows were silently lost and
  interface formulas were left blank.
- Decided with the user: when a material has several surfaces, lookups use the **median** VBM and
  report the min–max range (implemented in #2's lookup changes).
- Kept: `noDP` (no dipole correction) slabs; they agree with dipole-corrected runs of the same surface
  within 0.02 eV.

**Verification**
- InterMat Table 1: our IPs reproduce the published electron affinities (χ = IP − bulk TBmBJ gap)
  within about 0.05 eV for 15 of 21 matched slabs; the filter removes exactly the two broken ones.
  Script: `literature_advay/scripts/intermat_table_check.py`; details in findings F13.
- Surface records: 808 (486 OptB88vdW slabs + 322 derived TBmBJ rows) for 256 materials;
  mean VBM −4.8 eV. All 439 stored interfaces now have formulas.
- Snapshot `literature_advay/snapshots/01_surface_fix.csv` vs `00_baseline.csv`: Si VBM +0.72 → −4.92,
  GaAs −1.97 → −7.14 (first-loaded surface; median selection in #2 gives −6.3), CuI −0.28 → −4.99.

**Not yet addressed.** The edge model was trained on the old, wrong-scale data and still needs
retraining (critique B1–B3); until then ML edges remain unreliable.

**Requires:** `python -m materialstack build-db --only jarvis_surfacedb jarvis_interfacedb`

## #2 · 2026-10-03 · Hypothetical datasets quarantined; Castelli edges removed (A2)

**Problem.** Castelli (about 19k hypothetical cubic perovskites), Wolverton oxides (hypothetical ABO3)
and double perovskites (hypothetical A2BB′X6) use invented prototype structures. Matched by formula,
they contaminated real materials: In2O3 and Ga2O3 took band edges (CBM = VBM) and GLLB-SC gaps from
fake cubic "InInO3"/"GaGaO3". Castelli's band edges are also not computed: they are the Butler–Ginley
electronegativity estimate (median 0.03 eV from the formula), yet made up 85% of edge-model training
labels and were trusted lookups.

**Decision (with the user).** MaterialStack is a screening tool, so hypothetical compounds are kept
(they are what gets screened) but quarantined from real materials, instead of deleting the sources.

**Changes**
- `sources/matminer_sources.py`: Castelli VBM/CBM are no longer stored as band edges; kept in
  `extra.electronegativity_vbm/cbm`. Re-ingested; structure features re-computed for the renumbered
  Castelli structures (18,928 orphaned feature/proxy rows removed); proxy models retrained.
- `config.py`: `HYPOTHETICAL_SOURCES` = Castelli, Wolverton oxides, double perovskites.
- `db.py`: `hypothetical_filter()` hides hypothetical-source rows for any material that also has
  data from another source; applied in `gap_summary`, `edge_summary` and the `materials_best` view
  (now dropped and recreated by `init_db`). `is_hypothetical_only()` added.
- `structure.py`: the same filter in `candidate_structures` (structure selection) and
  `dft_gap_features` (DFT-gap model inputs).
- `predict.py`: hypothetical-only materials get the note "known only from hypothetical-structure
  datasets…".
- The database contents are unchanged by the quarantine; it is applied at read time and can be
  reverted by editing `HYPOTHETICAL_SOURCES`.

**Verification**
- In2O3 / Ga2O3: candidate structures now only JARVIS and SNUMAT; GLLB-SC feature NaN; best gap
  experimental (2.80 / 4.47 eV).
- 10,898 hypothetical-only materials keep their data, e.g. LaNbAlAgO6 is predicted using its stored
  GLLB-SC gap and carries the note.
- Snapshot `02_hypothetical_quarantine.csv` vs `01_surface_fix.csv` (this diff also includes the
  lookup changes still to be confirmed as #3): In2O3 and Ga2O3 no longer have CBM = VBM; Si|GaAs
  Type III → II.

**Not yet addressed.** Their edges now come from the old edge model, which is still unreliable
(B1–B3). The structure-aware gap model was trained with hypothetical DFT features for real
materials and should be retrained.

**Requires:** `build-db --only castelli_perovskites`, `featurize-structures`, `train-proxies`;
`train` pending.

## #3 · 2026-10-03 · Lookups use medians, flag conflicts, keep edges consistent with the gap (A4, A8)

**Problem.** The `materials_best` view picked, among equally ranked values, whichever record was
loaded first. Crystal phases are pooled under one formula, so CsPbI3 returned 1.67 eV (black phase)
or 2.76 eV (yellow phase) depending on load order, with no warning. Band edges likewise came from
the first-loaded surface (GaAs −7.14 eV of a −5.30 to −8.27 range). Looked-up DFT edges also kept the
DFT functional's gap, so CBM − VBM did not equal the displayed gap (ZnO: Eg 3.44, edges 2.37 apart).

**Changes**
- `db.py`: `gap_summary()` returns the median of all values at the best method rank, with min/max;
  `edge_summary()` returns the median VBM/CBM over a material's surfaces with the VBM range.
- `predict.py` (`resolve_layer`):
  - gap = median; if equal-rank values span more than 0.5 eV the gap is marked not trusted (so the
    junction is flagged uncertain) and a note lists the values and asks for the phase;
  - edges = median over surfaces; a note gives the range when it exceeds 0.3 eV; new result field
    `edge_spread_ev`;
  - computed (DFT slab) edges are anchored on the VBM: CBM = VBM + best gap, matching InterMat's
    definition (EA = IP − Eg). Measured (experiment/literature) edges are used as given.
- A spin–orbit rule (A3) was drafted and then removed: on hold by user decision.

**Verification**
- 13,361 materials have a trusted gap; 5,679 have several equal-rank values; **397** conflict by
  more than 0.5 eV and are now flagged (e.g. CsPbI3: 1.67 / 2.76 eV → median 2.22, not trusted).
- GaAs VBM −6.31 eV (median of 6 surfaces) and CBM = −6.31 + 1.51 = −4.80; Si CBM = −4.92 + 1.17.
- Snapshot `02_hypothetical_quarantine.csv` (taken with these rules active, plus the since-removed
  A3 rule affecting MAPbI3/FAPbI3/CsPbBr3 gaps).

**Requires:** nothing beyond #1–#2.

## #4 · 2026-10-03 · One record per surface; slab VBMs trusted explicitly (A6)

**Problem.** For every slab the loader also wrote a derived `DFT-surface-TBmBJ` record (same VBM,
CBM = VBM + bulk TBmBJ gap). It looked like independent data, double-counted each surface, and was
only "trusted" because its borrowed rank (3) passed the edge-trust threshold.

**Changes**
- `sources/jarvis_sources.py` (`surfacedb`): one `DFT-surface-OptB88vdW` record per slab; the bulk
  TBmBJ gap is kept as `extra.bulk_tbmbj_gap`.
- `config.py`: `DFT-surface-TBmBJ` removed from `METHOD_RANK`.
- `predict.py`: edge trust is explicit, `TRUSTED_EDGE_METHODS` = experiment, literature and
  DFT-surface-OptB88vdW (slab VBM; CBM rebuilt from the best gap as in #3), instead of
  `rank ≤ max(max_lookup_rank, 4)`.
- `literature_advay/scripts/edge_model_design.py`: selects bulk semiconductors via
  `extra.bulk_tbmbj_gap > 0`.

**Verification**
- Surface records 808 → 486 (256 materials); 322 carry a bulk TBmBJ gap.
- Snapshot `03_no_derived_surface_rows.csv` vs `02`: surface VBM/CBM unchanged and still trusted for
  Si, GaAs, CdTe, CdS, ZnO, GaN; ZnS and CuI medians now include all their surfaces
  (ZnS −7.09 → −6.62 eV, CuI −4.80 → −4.99 eV). Junction types unchanged.

**Requires:** `build-db --only jarvis_surfacedb`

## #5 · 2026-10-03 · Gap conflict flag judged by majority agreement (A4 follow-up)

**Problem.** #3 flagged a gap as "not trusted" when equal-rank values spanned more than 0.5 eV. A
single bad record in a compilation was enough: GaAs (34 values in 1.09–1.58 eV plus one of 15.11 eV)
and Si (1.12/1.17 plus stray 0.60 and 4.19 eV) were flagged, as were ZnO, GaN, CdSe and others —
397 materials in total, mostly well-measured ones.

**Change (agreed with the user).** A conflict is flagged only when fewer than half of the values
lie within 0.3 eV of their median.
- `db.py` `gap_summary()`: new `agree_frac` (share of values within `agree_window` = 0.3 eV of the
  median).
- `predict.py`: `GAP_MIN_AGREE_FRAC = 0.5` replaces `GAP_CONFLICT_EV`.
- The reported gap is unchanged (still the median, which already ignores stray values).

**Verification**
- Flagged materials 397 → 140 (analysis over all materials with several trusted values).
- Si 1.17 eV and GaAs 1.51 eV trusted again; CsPbI3 (1.67 / 2.76 eV) still flagged; ZrO2
  (0.56, 0.65, 3.80, 4.99 eV) flagged.

**Requires:** nothing.

## #6 · 2026-10-03 · Evaluation command; honest band-edge metrics (C1)

**Problem.** The only edge metric was the training CV (0.18 eV), dominated by Castelli rows that are
the Butler–Ginley formula, scored with the label's own DFT gap rather than the gap the tool uses, and
never compared with a baseline.

**Changes**
- New `materialstack/evaluate.py` and CLI `python -m materialstack evaluate [edges]`, writing
  `models/evaluation.json` (C2/C3 will be added as further parts).
- Edge evaluation: reference = median vacuum-aligned slab VBM for each bulk semiconductor in JARVIS
  surfacedb (155 materials; DFT, not experiment). The gap fed in is the one the pipeline would use
  (trusted lookup for 95, ML for 60). Compared with a leave-one-out constant and Butler–Ginley.
  Metrics: MAE, RMSE, bias, Spearman rank correlation, share within 0.3/0.5 eV.

**Result (baseline for the edge-model rebuild)**

| model | MAE (eV) | bias | Spearman | ≤0.5 eV |
|---|---|---|---|---|
| constant | 1.81 | 0.00 | n/a | 14% |
| Butler–Ginley | 1.37 | −0.70 | 0.68 | 25% |
| deployed edge model | **3.29** | **+2.99** | 0.69 | 3% |

The deployed model is worse than a constant: it learned the old wrong energy scale (VBMs ~3 eV too
shallow). The reference itself varies by a median 1.29 eV between surfaces of the same material
(50 materials with several surfaces), which bounds what a composition-only model can reach.

**Requires:** nothing (run `evaluate edges` after any model change).

## #7 · 2026-10-03 · Honest band-gap benchmark on all Borlido materials (C2)

**Problem.** The reported "Borlido holdout" covered only the 41 Borlido materials absent from every
other source; the other 407 were in the training data. 41 leftover materials are too few and
unrepresentative to say how well the model predicts unseen compounds.

**Changes**
- `evaluate.py`: `evaluate gap` scores every Borlido material (453 with features) against Borlido's
  curated values. Grouped 5-fold over the test materials by element set: each fold's evaluation
  model is trained on the normal training labels minus that fold's chemical groups (~5,100
  materials per fold). The deployed model is not touched.
- Two scenarios: (A) formula only — an unmade candidate; (B) crystal structure known, no stored DFT
  gap (structure descriptors + DFT-proxy predictions).
- Context: stored PBE / OptB88vdW / TBmBJ / HSE06 gaps of the same materials scored against Borlido.
- Screening metric `nonmetal_called_metal`: share of real semiconductors/insulators predicted to be
  metals (wrongly discarded by a screen). Largest errors listed.
- Inference-style label-fidelity flags (`label_method_experiment = 1`) are set on the test rows, as
  `predict_eg` does.
- An unrun draft of this benchmark inside `train()` was removed.

**Result (current labels; baseline for B7)**

| | MAE non-metals | RMSE | R² | Spearman | ≤0.5 eV | non-metal → metal |
|---|---|---|---|---|---|---|
| A: formula only | 0.71 eV | 1.65 | 0.53 | 0.82 | 63% | 5.5% |
| B: structure, no DFT | 0.63 eV | 1.51 | 0.61 | 0.86 | 70% | 5.7% |

Stored DFT on the same non-metals: PBE 1.13 eV (bias −1.09), OptB88vdW 1.26, TBmBJ 0.62, HSE06 0.61.
The formula-only ML beats PBE-level DFT and is ~0.1–0.17 eV behind TBmBJ/HSE06 on the same materials.
The large RMSE comes from solid noble gases (Ne 21.5 → 1.4 eV, no training analogues), very wide-gap
fluorides/oxides predicted several eV low, and insulators classified as metals (AlPO4, CaCO3, BaSnO3).

**Requires:** nothing (`python -m materialstack evaluate gap`).

## #8 · 2026-10-03 · Consensus band-gap labels, shared by lookup and training (B7)

**Problem.** Training dropped any material whose experimental gaps spanned > 1.0 eV (or IQR > 0.5 eV),
so a single bad report removed whole materials: Si, GaAs, ZnO and CdTe were missing from training
(104 materials dropped). Compilations copy each other (5,850 duplicate copies), so copied values
looked like independent agreement. Real disagreements below the thresholds were trained on a
meaningless midpoint (Na2Cd(GeS3)2: 2.57 / 3.21 → 2.89). Borlido's curated values were withheld
from the deployed model to serve as a test set.

**Changes (plan agreed with the user)**
- `clean.py`: new `consensus_gap()` — Borlido value used directly if present; otherwise copied
  values collapsed, values within max(0.3 eV, 10% of the gap) of the median count as agreeing; with a
  majority the label is the median of the agreeing values and the rest are set aside as outliers;
  without one the material is ambiguous (no DFT tie-break: tested, but it would mislabel SnO2 as
  2.60 eV because the stored HSE06 gap underestimates it). Replaces `materials_with_expt_scatter`;
  `CleanConfig.max_expt_spread/max_expt_iqr` → `consensus_abs_tol/rel_tol/min_agree`.
- `models.py` `load_eg_frame`: labels from `consensus_gap`; ambiguous materials dropped
  (`expt_no_consensus` audit rule); Borlido used as labels (no longer held out).
- `db.py` `gap_summary` and `predict.py`: lookups use the same function (`ambiguous` replaces the
  #5 agreement threshold), so lookup and training always agree on a material's gap.
- `cli.py`: `train --max-expt-spread` → `--consensus-tol`; the obsolete Borlido-holdout printout removed.
- README and CLAUDE.md updated.

**Verification — Borlido benchmark (`evaluate gap`, same 453 materials, before → after)**

| | formula only | structure, no DFT |
|---|---|---|
| MAE non-metals | 0.71 → **0.62 eV** | 0.63 → **0.57 eV** |
| R² | 0.53 → 0.59 | 0.61 → 0.66 |
| Spearman | 0.82 → 0.84 | 0.86 → 0.88 |
| within 0.5 eV | 63% → 69% | 70% → 71% |
| non-metal called metal | 5.5% → 4.0% | 5.7% → 4.4% |

On materials with a stored HSE06 gap, formula-only ML now matches HSE06 against experiment
(0.61 vs 0.61 eV). Training materials 5,266 → 5,332; 65 outlier reports set aside; 33 materials
without a majority. Part of the gain is cleaner labels, part is Borlido's curated values in training.

Lookups (snapshot `04_consensus_labels.csv` vs `03`): gaps move to curated values, e.g. Cu2O
2.58 → 2.17 eV, CuInSe2 0.96 → 1.04, CdTe 1.59 → 1.48.

**Requires:** `train` (pending; will be run together with the edge-model rebuild).

## #9 · 2026-10-03 · Junction benchmark: DFT interfaces and measured band offsets (C3)

**Problem.** Nothing checked the tool's final output — band offsets and junction types.

**Changes**
- `evaluate.py`: `evaluate junctions`.
  - Against the 321 JARVIS directly computed interfaces (ASJ, OptB88vdW; 239 semiconductor pairs):
    ΔEv = VBM(B) − VBM(A) for film A on substrate B, each layer through the normal pipeline;
    compared with Butler–Ginley and a zero-offset baseline; junction types compared using the
    pipeline gaps for both sides; results split by where the edges came from.
  - Against 21 **measured** valence-band offsets compiled in InterMat Table 2 (magnitudes), with the
    paper's own direct-interface (ASJ) and vacuum-alignment (IU) values as context. Values embedded
    as `INTERMAT_TABLE2` with the citation.

**Result**

| vs 21 measured offsets | MAE |ΔEv| |
|---|---|
| paper, direct interface DFT (ASJ) | 0.22 eV |
| paper, vacuum alignment on specific (110) surfaces (8 systems) | 0.47 eV |
| Butler–Ginley | 0.58 eV |
| **our pipeline** (vacuum alignment, median over all surfaces) | **0.91 eV** (0.82 on the paper's 8) |

Against the 321 DFT interfaces the pipeline's ΔEv MAE is 1.31 eV, worse than predicting zero
(0.86 eV); junction type agrees 44% of the time, sign 71% (offsets ≥ 0.2 eV). All these pairs use
looked-up slab edges, so this measures vacuum alignment itself. Even the paper's own vacuum-alignment
values differ from its interface calculations by up to 1.7 eV (CdS/Si 3.22 vs 1.48 eV).

**Findings that need decisions (added to the critique):** the median over all surfaces is the weak
point (A9), and the directly computed interface offsets — 0.22 eV from experiment — are not used by
the tool at all (B8).

**Requires:** nothing (`python -m materialstack evaluate junctions`).

## #10 · 2026-10-03 · Fast lookups (E2, plan step 0.1)

**Problem.** Every lookup queried the `materials_best` view, whose window functions ranked all ~260k
records before filtering to the requested material: about 1.5 s per material. Too slow for screening
and for the evaluation suite, which resolves hundreds of layers.

**Changes**
- `db.py`: the best-row query is one template (`_BEST_SQL`, built by `_best_sql(ids)`). The
  `materials_best` view uses it unfiltered (`VIEW_SQL`, recreated by `init_db`); `lookup()` uses it
  filtered to the formula's material ids *before* ranking. Same rules, one definition.

**Verification**
- Lookup: ~1.5 s → ~3 ms per material; a 5-layer stack prediction ~0.1 s.
- Same results (e.g. In2O3 best gap 2.8 eV, experiment); the view still lists all 104,915 materials.

**Requires:** nothing (the view is recreated on the next `init_db`/`build-db`).

## #11 · 2026-10-03 · Test suite; quarantine filter bug fixed (C4, plan step 0.2)

**Changes**
- New `tests/test_physics_and_data_rules.py` (12 tests, ~2 s; `.venv/bin/python -m pytest -q`):
  consensus labels (outlier ignored, two phases ambiguous, copies collapsed, Borlido preferred,
  metals), Butler–Ginley symmetry, Type I/II/III classification, surface loader vacuum alignment and
  work-function filter (fake JARVIS rows), hypothetical quarantine (in-memory DB), and the fast lookup
  query against the view (real DB; skipped if not built).
- **Bug found by the tests:** after #2's speed-up to a correlated `EXISTS`, `hypothetical_filter()`
  compared `material_id` with itself inside the subquery, so it hid hypothetical-source rows for
  *every* material — including the 10,898 known only from hypothetical datasets, which lost their
  structures and DFT gaps in lookups and model inputs. Real materials, the benchmarks and the
  snapshot panel were unaffected.
- `db.py`: `hypothetical_filter(table)` now takes the table name/alias and qualifies both columns;
  the view's filter is built from the same function (`_HYP_VIEW_FILTER = hypothetical_filter("r")`)
  instead of a separate copy. `structure.py` passes `"structures"`.

**Verification:** tests pass. In2O3 still sees only JARVIS/SNUMAT structures and no GLLB-SC gap;
hypothetical-only TeRhN3 again uses its Castelli structure and GLLB-SC gap.

**Requires:** nothing.

## #12 · 2026-10-03 · Directly computed interface offsets used for junctions (B8, plan step 1.2)

**Problem.** JARVIS has directly computed valence-band offsets for 239 semiconductor pairs (about
0.22 eV from experiment), but the tool always aligned each layer to vacuum separately (0.91 eV vs
measured offsets). Also: the junction rule existed twice (predict and evaluate), and the CBO comment
had the sign backwards ("> 0: bottom CBM lower").

**Changes**
- `db.py`: `interface_offset(con, top, bottom)` returns VBM(top) − VBM(bottom) from stored JARVIS
  interfaces of the pair in either order (sign flipped as needed), median over orientations, with n
  and range.
- `predict.py`:
  - `junction_type()` — the single Type I/II/III rule (also used by `evaluate.py`; its copy removed).
  - `classify_junction(top, bottom, interface=None)` — offset from the interface when given, else
    from the two VBMs; CBO, type and margin all follow from that offset and the two gaps; new field
    `offset_source`; sign conventions documented correctly.
  - `resolve_stack(..., use_interfaces=True)` looks up each adjacent pair.
- `cli.py`: prints where each junction's offset came from.
- `evaluate.py`: the measured-offset benchmark scores the full pipeline and the vacuum-alignment path
  separately (what pairs without interface data get).
- Tests: interface sign in both orders; interface offset overrides vacuum alignment (14 tests pass).
- A9 (surface selection, plan step 1.1) was studied first: no policy beat the current median on both
  benchmarks (findings F18), so it was left unchanged.

**Verification**
- 21 measured offsets: pipeline 0.91 → **0.41 eV** (all 21 pairs have interface data); vacuum
  alignment alone still 0.91 eV. Not the paper's 0.22 eV because the lookup takes the median over all
  computed orientations of a pair rather than the measured plane.
- Snapshot `05_interface_offsets.csv`: Si|GaAs VBO 1.39 → 0.25 eV (measured 0.23), Type II → I;
  GaN|ZnO 2.12 → 0.40 eV (measured 0.70). Perovskite/oxide stacks have no interface data — unchanged.

**Requires:** nothing.

## #13 · 2026-10-03 · Band edges rebuilt: slab + Butler–Ginley blend, no ML edge model (B1–B4, plan step 2.1)

**Problem.** The deployed edge model (LightGBM correction to Butler–Ginley) was trained on
formula-derived Castelli edges and wrong-scale slab data: VBM error 3.29 eV, worse than a constant.

**Design study** (`literature_advay/scripts/edge_model_study.py`, findings F19): candidates scored with
out-of-fold VBMs on the slab reference, 327 DFT interface offsets and 21 measured offsets; the gap fed
in is the pipeline's own (B2). Fitting slab VBMs better made offsets worse: small LightGBM (comp. or
+structure, B4) and direct LightGBM reached 0.80–0.88 eV on slab VBMs but 1.09–1.11 eV on DFT
interface offsets and 0.70–0.84 eV on measured ones, vs Butler–Ginley's 1.00 / 0.58. A ridge
correction tied Butler–Ginley on offsets. Blending the slab VBM with Butler–Ginley beat both; the
weight 0.3 follows from inverse-variance weighting (slab offset errors ≈ 1.3–1.6× Butler–Ginley's),
consistent with the 0.25–0.35 optimum seen in the study. One global parameter, but it was informed by
the same benchmarks, so treat the gains below as slightly optimistic.

**Changes**
- New `materialstack/edges.py`: `estimate_edges(gap, chi, looked_up)` — measured edges as given;
  else VBM = 0.3 × slab VBM + 0.7 × Butler–Ginley where slab data exists, else Butler–Ginley;
  CBM = VBM + Eg. Docstring records the evidence.
- `predict.py`: `resolve_layer` uses it (old ML-edge branch and the VBM/CBM "consistency" fix-ups
  removed); `_load_models` → `_load_gap_model` (no edge bundle); module docstring updated.
- Removed the old edge model: `EdgeCorrectionModel`, `load_edge_frame`, `_grouped_cv_edge`,
  `predict_edges`, `load_edge_model`, edge training in `train()` (models.py); `clean_edge_frame` and
  its config (clean.py); dead Borlido-holdout block in `train()`; `models/edge_model.joblib` deleted.
  `metrics.json` now records the edge rule instead.
- `cli.py`, `api.py`: no edge-model output; `/api/metrics` tolerates its absence (UI keeps working).
- `evaluate.py`: edge benchmark compares constant / Butler–Ginley / pipeline; junction pairs grouped
  by how their edges were made.
- 3 new tests (Butler–Ginley fallback, blend keeps CBM − VBM = Eg, measured edges as given); 17 pass.

**Result (vacuum-alignment path, i.e. pairs without interface data)**

| | before | after |
|---|---|---|
| 321 DFT interfaces: ΔEv MAE | 1.31 eV | **0.79 eV** (zero baseline 0.86) |
| junction type agreement / sign | 44% / 71% | **64% / 76%** |
| 21 measured offsets | 0.91 eV | **0.48 eV** (paper's 8: 0.82 → 0.34; paper's own 0.47) |

Absolute VBMs now match approximate literature ionization potentials for many layers (Si −5.22,
CdTe −5.84, ZnO −7.80, CsPbI3 −5.93 eV); the spurious Type III junctions (ZnO|Cu2O, CdS|CuInSe2) are gone.

**Known failure (new critique A10).** Butler–Ginley puts VBMs formed by cation d-states or s² lone
pairs far too deep: Cu2O −6.41, CuI −6.44, NiO −7.76, CsSnI3 −5.56 eV vs roughly 5.0–5.4 / 4.9 eV
from the literature. These are the common inorganic hole-transport layers; the benchmarks (III-V and
II-VI semiconductors) cannot detect it. Also, MAPbI3 still uses the SOC-blind HSE06 gap (A3, on
hold), which moves SnO2|MAPbI3 to Type I (VBO −0.05 eV).

**Requires:** nothing (old `edge_model.joblib` no longer used).

## #14 · 2026-10-03 · All models retrained on the new labels (plan phase 3)

**Changes**
- `train` run with the consensus labels (#8): composition-only and structure-aware band-gap models,
  DFT-gap proxies unchanged (#2), XGBoost baseline. No edge model (#13).
- `structure.py`: the GLLB-SC stored-gap feature removed. GLLB-SC only comes from the quarantined
  hypothetical datasets, so no training material had a value (training warned of an all-empty column).

**Result** (grouped CV on the training labels — labels changed in #8, so compare with care; the fixed
yardstick is the Borlido benchmark)
- Composition-only: MAE 0.461 eV, R² 0.760 (before: 0.451 / 0.745).
- Structure-aware: MAE 0.426 eV, R² 0.805 (before: 0.419 / 0.784). XGBoost baseline 0.459 eV.
- Borlido benchmark unchanged: formula only 0.62 eV, structure known 0.57 eV.
- `models/metrics.json` and `train_clean_audit.json` regenerated (tracked files: expected diff).

**Requires:** `python -m materialstack train`.

## #15 · 2026-10-03 · Junction-type probabilities and a "check with DFT" flag (D1/S1, plan phase 4)

**Problem.** "Uncertain" was a fixed rule (offset margin < 0.3 eV, or < 0.6 eV with untrusted
inputs), unrelated to how accurate the inputs actually are.

**Changes**
- New `materialstack/uncertainty.py`: σ for every input according to how it was obtained, each value
  traced to a benchmark (measured gap 0.2, HSE06/ML gap 0.75, ambiguous gap 0.8; VBM 0.2 measured /
  0.7 slab blend / 0.9 Butler–Ginley; interface-DFT offset 0.3 eV). `junction_type_probabilities()`
  samples gaps and the offset (Monte Carlo, fixed seed) and classifies each sample with the same rule
  as `junction_type`.
- `predict.classify_junction`: new fields `type_probabilities`, `confidence` (probability of the
  reported type) and `vbo_sigma_ev`; `uncertain` now means "most likely type < 80% → check with DFT";
  the reason says so. The old 0.3 eV margin rule (`JUNCTION_BOUNDARY_EV`) removed; `margin_ev` kept.
- `cli.py`: prints probabilities, VBO ± σ and "CHECK WITH DFT".
- `evaluate.py`: calibration of the probabilities on the 321 DFT interfaces (Brier score, reliability
  table, accuracy when confident vs flagged).
- 2 tests (clear vs coin-flip cases; probabilities in the junction output); 19 pass.

**Verification (DFT interfaces, vacuum-alignment path)**
- Brier 0.468 vs 0.539 for always predicting the most common type.
- Reliability, stated → actual: 44 → 30%, 57 → 57%, 72 → 60%, 85 → 77%, 94 → 92% (slightly
  overconfident in the middle; the reference itself has ~0.3 eV DFT error).
- Confident (≥ 80%) for 28% of junctions and right 83% of the time; the other 72% are flagged for DFT
  and are right only 55% of the time, so flagging them is warranted. Pairs with interface data get
  σ = 0.3 eV and are usually confident.

**Requires:** nothing.

## #16 · 2026-10-03 · The UI and API show what the pipeline knows (D3, plan phase 5)

**Changes**
- `api.py`: `GET /api/evaluation` serves summaries of `models/evaluation.json` (band gap on Borlido
  with DFT context; offsets vs measured and DFT interfaces; probability calibration).
- `predict.py`: junction labels use the names the user typed ("TiO2 | MAPbI3", not the internal
  formula "H6PbCI3N").
- Frontend (`api.ts`, `App.tsx`, `App.css`):
  - junction cards: type with its probability, a stacked Type I / II / III probability bar,
    VBO ± σ, where the offset came from, and "check with DFT" when confidence < 80%;
  - layer cards: the name typed plus the parsed formula, and the slab VBM spread when > 0.3 eV
    (gap-ambiguity and other notes were already shown);
  - performance report: the old edge-model panel and its feature list replaced by the edge rule and
    a "honest benchmarks" section (offsets vs measured and DFT interfaces, confident-call accuracy,
    band gap on all Borlido materials vs PBE/HSE06 on the same materials); obsolete
    "Borlido-only holdout" callout removed; header stat is now "band offset MAE vs measured".
- `npm run build` and `npm run lint` pass; 19 Python tests pass.

**Verification:** the server serves the new bundle; `/api/predict` returns probabilities and offset
sources; `/api/evaluation` and `/api/metrics` respond. Not checked visually (no browser in this
environment) — open http://127.0.0.1:8001 to review.

**Requires:** `cd frontend && npm run build`, restart `serve`.

## #17 · 2026-10-03 · Cleanup and docs (E1, plan phase 6, partial)

- Removed dead code: unused `Iterable` import and `transaction()` helper (db.py); unused `source_tag`
  parameter (jarvis_sources.py); duplicate `borlido_expt` loader entry; obsolete Borlido-holdout
  plumbing (`held_out`, `hold_eg`) in models.py/evaluate.py.
- README: prediction flow, band edges, junction probabilities, evaluate/test commands.
- Verified unchanged: 19 tests pass; retrain gives the same CV metrics; all benchmarks identical.
- Not done (user's call): leftover files `data/exports/`, `catboost_info/`, Vite template assets;
  `MaterialStack_EVALUATION.md` describes the old pipeline.

## #18 · 2026-10-03 · Spin–orbit rule restored (A3, taken off hold by the user)

- `predict.py`: `soc_sensitive()` — for compounds containing Pb, Bi, Tl or Hg only measured gaps
  (rank 1) are looked up; computed DFT gaps (calculated without spin–orbit coupling) are ignored, with
  a note. Test added (20 pass).
- Effect: MAPbI3 no longer takes the HSE06 2.64 eV gap; the ML fallback gives 2.46 eV, still far
  from ~1.6 eV measured, because the structure-aware model is itself fed SOC-blind DFT gaps. Measured
  values for the key perovskites are added in the A7 curation (#20).

## #19 · 2026-10-03 · Unused files removed (E1, user decision)

- Deleted: `data/exports/` (old TiO2 / Cs2AgBiBr6 CSV exports, not produced or read by any code),
  `catboost_info/` (logs from CatBoost, which the code does not use), Vite template assets
  `frontend/public/{favicon,icons}.svg` and `frontend/src/assets/hero.png` (not referenced).
- Kept by decision: `MaterialStack_EVALUATION.md` (describes the old pipeline; to be updated by the user).
- Frontend still builds.

## #20 · 2026-10-03 · Hybrid-functional oxide surfaces added (plan step 2.2; partly fixes A10)

**Source.** Kiyohara, Hinuma & Oba, JACS 2024 (doi:10.1021/jacs.3c13574), SI Data S1: IP and EA for
2,912 oxide surfaces (322 formulas), dielectric-dependent hybrid functional. ACS blocks automated
download; obtained through Europe PMC's supplementary-files API.

**Changes**
- New loader `sources/kiyohara.py` (`kiyohara_oxides`, in the default build): one record per surface,
  method `DFT-surface-hybrid` (rank 3), VBM = −IP, CBM = −EA, bulk hybrid gap kept. Reads
  `data/cache/kiyohara2024_ja3c13574_si_002.xlsx` (gitignored; downloads it if missing).
- `config.py`: method rank and default source registered.
- `edges.py`: surface weight per method (`SLAB_WEIGHTS`): GGA JARVIS slabs 0.3 (validated);
  hybrid oxide surfaces **0.8** (decided with the user: inverse-variance weighting assuming ~0.4 eV
  hybrid accuracy vs ~0.9 eV for Butler–Ginley; not validated here, ZnO is the only oxide in the
  benchmarks).
- Test: hybrid data weighted above GGA slabs; Cu2O-like case stays shallow (21 tests pass).

**Effect**
- Cu2O VBM −6.41 → **−5.11 eV** (hybrid alone −4.78), fixing A10 for oxide hole-transport layers.
  ZnO −7.58, rutile/anatase TiO2 mix −7.76, SnO2 −8.95 eV now from hybrid data.
- Benchmarks: DFT interfaces 0.79 → 0.78 eV (type agreement 64%), confident calls right 84%;
  measured offsets unchanged.
- Not covered: NiO (magnetic oxides excluded from the dataset), CuI, CuSCN (not oxides) — A10 remains
  open for these.

**Requires:** `python -m materialstack build-db --only kiyohara_oxides`.

## #21 · 2026-10-03 · Curated measured band edges (A7, revived by the user); calibration made fair

**Data** — `data/literature/measured_band_edges.csv` (29 rows, every row cited; tracked in the repo):
- Tao et al., Nat. Commun. 10, 2560 (2019), Table 1: measured ionization energy, electron affinity and
  optical gap for all 18 Cs/MA/FA × Pb/Sn × I/Br/Cl perovskites (UPS/IPES).
- InterMat Table 1 experimental column: measured electron affinities for Si, Ge, GaAs, InAs, AlSb,
  GaSb, AlN, GaN, GaP, InP, ZnTe (original references cited there), paired with Borlido gaps
  (ZnTe: consensus of compilations).
- Not found as open, citable measurements: NiO, CuI, CuSCN, oxide transport layers (reported oxide
  values scatter strongly with preparation). Organic layers still need formula-free support.

**Changes**
- `sources/literature.py`: a row with only EA and gap now also gets its VBM (VBM = CBM − gap).
- `clean.py`: `literature_csv` joins Borlido as a preferred (curated) gap source in `consensus_gap`.
- `uncertainty.py`: measured-edge σ 0.2 → 0.25 eV, derived from the 7 measured offsets with measured
  edges on both sides (RMSE 0.33 eV / √2). `junction_uncertainty(..., reference_sigma)` lets the
  evaluation add the reference's own noise.
- `evaluate.py`: calibration against DFT interfaces includes their ~0.3 eV error (otherwise accurate
  inputs look overconfident whenever the DFT reference is off).

**Effect**
- MAPbI3: gap 2.64 → 1.59 eV, VBM −5.93, CBM −4.36 (measured); FAPbI3 1.51 eV; CsPbI3 1.72 eV (no
  longer ambiguous); GaN/GaAs edges from measured EAs.
- TiO2|MAPbI3: Type II with TiO2's CBM 0.24 eV below the perovskite's and VBM 1.8 eV deeper — the
  expected electron-transport alignment.
- Benchmarks: measured offsets, vacuum-alignment path 0.47 → 0.36 eV (full pipeline 0.41); DFT
  interfaces 0.81 eV, type right 65%; probabilities Brier 0.456 (before 0.468), confident calls right
  83%, 67% flagged; band gap on Borlido 0.62 / 0.60 eV (formula / structure; was 0.62 / 0.57).
- Still wrong: NiO (Butler–Ginley only, VBM −7.76 eV) makes MAPbI3|NiO look hole-blocking; CuI −6.44 eV.

**Requires:** `build-db --only literature_csv`.

## #22 · 2026-10-03 · UI fixes found by driving it in a headless browser

Checked with Playwright (headless Chromium) at desktop, dark-scheme and phone sizes; new script
`literature_advay/scripts/ui_screenshots.py` (enters a stack, predicts, saves screenshots, reports
console errors and page width).

- Probability bar was broken: it inherited the `.junction > div` header-row layout (space-between,
  baseline alignment), so segments were spread apart, collapsed to a thin line, and the Type III
  segment showed as a blank gap. `App.css` now overrides that layout for `.prob-bar`.
- Long source strings ran into neighbouring cards → `.source` wraps (`overflow-wrap: anywhere`).
- Duplicate notes: the UI's own "slab VBM varies" line repeated the backend note → removed.
  `predict.py` no longer adds trivial alias notes ("ZnO → ZnO").
- The band-edge rule text was out of date (said 0.3 only): now generated from `edges.SLAB_WEIGHTS`
  by `edges.describe_rule()` and served by `/api/metrics` (also used in `metrics.json`).
- Honest benchmarks moved to the top of the performance report.
- Phone width: the nav overlapped ("MaterialStack" over "Lab", "Pipeline" cut off) → links wrap
  below the brand; the hero title made the page 498 px wide on a 390 px screen (sideways scroll)
  → title sized to the viewport and hero columns allowed to shrink. Page width now equals the screen
  at 360, 390, 768 and 1400 px.
- No console errors. The app has no dark theme (it renders the light design in dark mode too).

**Requires:** `cd frontend && npm run build`, restart `serve`. Playwright is a dev-only tool, not in
requirements.txt.

## #23 · 2026-10-03 · Professional, tool-first UI

Reviewed full-page screenshots (headless Chromium) and rebuilt the page around the tool; page height
5,418 → ~2,450 px at desktop width.

**Removed:** the marketing hero (giant title, decorative band art, "Open the lab" buttons), the
"Pipeline" step cards and catalogue-by-family bars, the feature-importance and label-cleaning panels,
the outdated footer ("Models frozen for this build"), the wide display font (Syne) and the
`framer-motion` animation dependency.

**New / changed (`App.tsx`, `App.css`, `index.css`, `index.html`):**
- Compact header (name, one-line purpose, links: Predict · Validation · Method); the input form is
  the first thing on the page, with example-stack chips that fill and run a stack in one click and a
  plain "Crystal phase" dropdown.
- Results: a **Layers table** (Eg, VBM, CBM to 2 decimals, a Measured / Computed / ML estimate /
  Ambiguous badge, provenance in plain language — "Measured · Borlido 2019", "Hybrid-DFT surfaces +
  electronegativity" — with the raw source on hover) and an **energy-level diagram** (each layer's
  gap as a box on a common vacuum scale, CBM/VBM labelled, legend).
- Junctions: ΔEv ± σ, ΔEc, offset source, type probabilities, and a plain-language reading
  ("Electrons collect in TiO2, holes in MAPbI3 (charge separation)").
- Validation: four headline numbers, offset and band-gap benchmark tables, and model internals in a
  collapsible "Model details". Short "Method" section and a data-sources footer.
- Phones: layer rows stack (numbers, then provenance); no sideways scroll at 390 px.

**Backend support:** `db.gap_summary` credits only the sources behind the number (Borlido alone when
it decided the value, not every compilation); `LayerResult.display_formula` gives the formula as a
chemist writes it (CH3NH3PbI3 rather than the reduced H6PbCI3N); the redundant alias note is hidden
in the UI.

**Verification:** build and lint pass; 21 tests pass; no console errors; page width equals the screen
at 390 and 1280 px.

## #24 · 2026-10-05 · Verification against the user's multilayer gold standard

- New `verif/verify_gold_standard.py` scores `verif/multilayer_gold_standard.csv` (50 scorable stacks,
  62 distinct layers, 102 interfaces; row 51 has no edges) in two scopes: **entire tool** (full
  pipeline) and **model only** (ML band gap + Butler–Ginley edges + vacuum alignment, no database
  values), with trivial baselines. Device abbreviations (CBTS, IGZO, i-ZnO, CZTSSe, CIT, CGS, CdZnS,
  MASnBr3, mixed Cs/FA/MA perovskite) are mapped to formulas inside the script, not in the tool;
  organic layers are counted as unsupported.
- Outputs in `verif/results/`: `verification_summary.csv` (presentable), `verification_by_tier.csv`,
  `verification_metrics_long.csv`, `layer_details.csv`, `junction_details.csv`.
- No pipeline change. Results in findings F20.

## #25 · 2026-10-05 · Database audit: measured levels, organic layers, measured offsets, sanity rules

**Data (only peer-reviewed measurements, each value read in the paper's text; citation in every row):**
- `data/literature/measured_band_edges.csv`: +19 rows — PCBM, C60, Spiro-OMeTAD, MoO3, TiO2 (anatase),
  ZnO and SnO2 (two surface terminations each), In2O3, NiOx (two UPS values), CuI, CuSCN, SnS,
  CuSbS2, Cs2SnI6. Rows giving only an ionization energy store the VBM; the CBM then comes from the
  material's band gap.
- New `data/literature/offsets/measured_band_offsets.csv` + source `literature_offsets`: measured
  interface offsets (CdS/CuInSe2, Morkel 2001; CdS/ZnO, Klein 2010). `db.interface_offset` prefers
  these over JARVIS DFT interfaces and returns the measured CBO, which `classify_junction` uses as is
  (σ 0.2 eV).

**Organic layers:** Spiro-OMeTAD, PTAA, PEDOT:PSS, P3HT, PCBM/PC61BM, PC71BM, C60, BCP map to
`organic:<name>` identities (no more "C60 → carbon"). They use measured IE/EA only (no ML, no
Butler–Ginley) and are excluded from featurising and training. Without a measured value the layer is
reported as unresolved. New abbreviations: i-ZnO, IGZO, CBTS, CGS, CIT, MASnBr3, FASnI3.

**Sanity rules** (`build_db.sanitize`, run after every build; removed rows written to
`data/sanitize_removed_records.csv`): negative gaps; gaps > 15 eV except noble-gas solids; measured
gaps > 3× (and > 3 eV above) the median of the other measured values (decimal slips such as InSe
13.2 eV). 15 records removed.

**Code:** `db.edge_summary` combines IP-only and full records (CBM = median VBM + median measured
CBM − VBM); `edges.estimate_edges` handles VBM-only measurements; `predict._resolve_organic`.
Verification script scores organic layers for the tool and adds a clear-junction metric (both gold
offsets ≥ 0.2 eV). 26 tests pass (5 new).

**Results:** findings F21, F22. Gold set (tool): type 52 → 59%, ΔEc MAE 0.65 → 0.53 eV, ΔEv 0.61 →
0.56 eV, coverage 91 → 100%. Independent measured-offset benchmark unchanged (0.41 eV).

## #26 · 2026-10-06 · Simplified: CSV data, one model, three scripts

**Why:** the project had a 400 MB SQLite database (11 tables/views), data decisions in seven places,
five trained models (composition, structure-aware, three DFT proxies) plus an XGBoost baseline, and
~7,150 lines of Python. Hard to read, debug and explain. Now ~1,070 lines.

**Data** (decided with the user): CSV files only.
- `data/curated/` (hand-edited, cited): `measured_band_edges.csv`, `measured_band_offsets.csv`,
  `aliases.csv` (moved out of Python).
- Built by `build_data.py`: `band_gaps.csv` (4,769 materials), `band_gaps_rejected.csv` (288 reports,
  rules R1–R6 with reasons), `dft_gaps.csv` (67,895 formulas; GGA and hybrid; JARVIS polymorphs within
  50 meV/atom only), `band_edges.csv` (43 measured, 322 hybrid-DFT surface materials).
- Sources dropped: Foundry-ML gaps (unclear provenance), Kingsbury (duplicate of Zhuo), Materials
  Project, hypothetical sets (Castelli, Wolverton, double perovskites), JARVIS 2D, dielectric set,
  JARVIS GGA surfaces and interfaces, all crystal structures.
- New rule R2: materials with Tc, Pm or actinides are not used.

**Model:** one LightGBM regressor on log(1+Eg) for semiconductors (`materialstack/model.py`): Magpie
composition features + χ + n_elements + DFT gap hint; missing values left to LightGBM (no median
imputation). Removed: metal classifier (metals are looked up, or recognised as only-metallic-element
formulas), structure-aware model, DFT-proxy models, XGBoost, perovskite/family features.

**Scripts:** `build_data.py`, `train.py` (5 s), `validate.py` (12 s; band-gap CV, band edges with the
measurement hidden, 21 measured offsets, gold stacks with and without measured edges) →
`results/metrics.csv` + `metrics.json` + two detail tables. Replaces `evaluate` and
`verif/verify_gold_standard.py`.

**Prediction** (`predict.py`, absorbs `edges.py`, `uncertainty.py`): unchanged physics; σ values now
set from `validate.py` (ML gap 0.5, VBM measured 0.25 / hybrid surface 0.65 / Butler–Ginley 1.3 eV).

**API/UI:** endpoints `/api/health`, `/api/metrics`, `POST /api/predict` (no `/api/stats`,
`/api/evaluation`); UI without the polymorph selector and model-ablation panels; Validation shows four
headline numbers and the full metrics table. Build, lint, headless check (desktop, dark, phone: no
console errors, no sideways scroll) pass.

**Removed files:** `materialstack/{db,build_db,models,clean,structure,evaluate,edges,uncertainty,features}.py`,
`materialstack/sources/`, `verif/verify_gold_standard.py`, `verif/results/`, old model files, the
SQLite database, obsolete `literature_advay/scripts/*` (results kept in findings), `catboost_info/`.

**Tests:** rewritten, 15 pass. **Results:** findings F23. Band gap 0.40 eV CV; offsets 0.45 eV
(old 0.41, the cost of dropping JARVIS GGA slabs); gold stacks type 61 %, ΔEc/ΔEv sign 80/83 %.
