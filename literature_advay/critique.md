# Project critique

A review of MaterialStack from a materials-science and ML point of view, written 2026-10-03.
Each issue has an ID, a severity, and a status. Fixes are recorded in [`../CHANGELOG.md`](../CHANGELOG.md)
with before/after numbers. Evidence for most issues is in [findings.md](findings.md).

Severity: **critical** = gives wrong answers today · **high** = makes accuracy claims or results
unreliable · **medium** = limits accuracy or usefulness · **low** = hygiene.

Status: `open` · `in progress` · `fixed (date)` · `won't fix (why)`.

---

## Intended use (agreed 2026-10-03)

MaterialStack is a **screening tool**: it should cut down how many candidates need DFT by ranking
them and ruling out clear mismatches, while being honest about close calls. So the priorities are:
not wrongly rejecting good candidates, sensible ranking, honest uncertainty, and handling compounds
that have not been made yet. Exact accuracy on single materials matters less.

## Summary

The overall design (look up trusted values first, fall back to ML, derive the junction from physics)
is right. The problems are concentrated in three places:

1. **Band-edge data.** Every band edge in the database is either on the wrong energy scale (JARVIS
   surfaces) or generated from an electronegativity formula (Castelli). The edge model learns from
   these, and the app looks them up directly.
2. **Evaluation.** The headline edge error (0.18 eV) mostly measures how well the model reproduces
   a formula; the "held-out" set is 41 materials; nothing checks the final output (junction type).
3. **Physics shortcuts in lookups.** Crystal phases are pooled under one formula, ties are broken
   by load order, and spin-orbit-sensitive compounds trust DFT gaps that are about 1 eV too large.

The band-gap side (two-stage LightGBM on about 5k experimental gaps, grouped cross-validation, DFT
proxy features) is reasonable and should be kept.

---

## A. Data correctness

| ID | Issue | Severity | Status |
|---|---|---|---|
| A1 | **JARVIS surface edges stored without vacuum alignment.** Loader stores `surf_vbm`/`surf_cbm` instead of subtracting the vacuum level `avg_max`. Si VBM shows +0.72 eV (expected about −5.2); Si/GaAs comes out Type III. Work function stored as `−efermi` has the same error. About 120 of 607 surfaces have broken vacuum levels and need filtering. (F1) | critical | fixed 2026-10-03 (CHANGELOG #1) |
| A2 | **Castelli band edges are the Butler–Ginley formula, used as data.** Median 0.03 eV from the formula. They are 85% of edge training labels (circular) and are accepted as *trusted* lookups (`trusted_edges = rank ≤ 4`). Hypothetical cubic "InInO3"/"GaGaO3" supply the edges shown for In2O3 and Ga2O3. (F2) | critical | fixed 2026-10-03 (CHANGELOG #2): edges removed; hypothetical sources quarantined from real materials |
| A3 | **Spin-orbit coupling ignored in lookups.** HSE06 gaps (rank 2, trusted) for Pb/Bi/Tl/Sn halides are computed without SOC, which overestimates gaps by roughly 0.5–1 eV for Pb iodides. MAPbI3 returns 2.64 eV (expt about 1.6). | high | fixed 2026-10-03 (CHANGELOG #18): measured gaps only for Pb/Bi/Tl/Hg; measured perovskite values via A7 |
| A4 | **Lookup picks the first-loaded record and pools polymorphs.** `materials_best` orders by `method_rank, record_id`, so equal-rank values are chosen by ingest order. CsPbI3 has experimental 1.67 eV (black phase) and 2.76 eV (yellow phase); which is returned depends on load order, with no warning. | high | fixed 2026-10-03 (CHANGELOG #3) |
| A8 | **Looked-up DFT edges inconsistent with the displayed gap.** Slab CBMs carry the functional's gap, so CBM − VBM ≠ Eg (ZnO: 2.37 vs 3.44 eV). | medium | fixed 2026-10-03 (CHANGELOG #3): CBM = VBM + best gap |
| A9 | **Surface choice for lookups is too crude.** Median VBM over all of a material's surfaces mixes in polar/unstable slabs. Against 21 measured band offsets: our vacuum alignment 0.91 eV vs the paper's 0.47 eV using specific non-polar (110) surfaces, and worse than plain Butler–Ginley (0.58 eV). (C3, F17) | high | investigated 2026-10-03 (F18): no selection policy beats the median on both benchmarks; kept median; combining slab VBMs with Butler–Ginley/ML moves to plan step 2.1 |
| A10 | **Butler–Ginley fails for d-/s²-derived valence bands.** VBMs formed by Cu 3d, Ni 3d or Sn 5s² states come out ~1–2 eV too deep (Cu2O −6.41, CuI −6.44, NiO −7.76 eV vs ~5.0–5.4 eV literature IPs; CsSnI3 −5.56 vs ~4.9). These are the common inorganic HTLs; the offset benchmarks (III-V, II-VI) cannot see it. Fix: measured IPs for these layers (A7) or a validated chemistry-specific correction. (CHANGELOG #13) | high | partly fixed 2026-10-03 (CHANGELOG #20, #21): Cu2O via hybrid data (−5.11 eV), perovskites/Sn halides measured; NiO, CuI, CuSCN now measured (CHANGELOG #25: −5.3, −5.2, −5.3 eV) |
| A5 | **168 of 593 JARVIS interface offsets dropped** because the `offset` field is a dict in those rows and the loader only accepts numbers. | medium | won't fix (2026-10-03): all 168 are empty dicts `{}` in JARVIS, no offset was computed, nothing is lost |
| A6 | **Synthetic TBmBJ surface records look like data.** `DFT-surface-TBmBJ` rows are our construction (surface VBM + bulk TBmBJ gap) but are ranked 3 and preferred over the real OptB88vdW slab result in lookups. Acceptable if labelled as derived; currently indistinguishable. | low | fixed 2026-10-03 (CHANGELOG #4): derived rows dropped, slab VBM trusted explicitly |
| A7 | **No experimental band-edge data at all.** `literature_csv` is empty, so no layer ever gets a measured VBM/CBM. Organic transport layers (Spiro, PTAA, PCBM) cannot be handled at all. | high | partly fixed 2026-10-03 (CHANGELOG #21): 29 cited measured values; 2026-10-05 (CHANGELOG #25): +19 incl. oxides, Cu HTLs, NiO, organics (Spiro, PCBM, C60) and 2 measured interface offsets; PTAA, P3HT, Cu2O, CdS, CIGS, CZTS, Sb2Se3, Cs2AgBiBr6 still missing |

## B. Modelling

| ID | Issue | Severity | Status |
|---|---|---|---|
| B1 | **Edge model target is ill-posed.** It predicts a correction to Butler–Ginley, mostly from labels that *are* Butler–Ginley (A2) or on the wrong scale (A1). On real surface data it does not beat a constant (0.61 vs 0.54 eV, F3). Better target: VBM vs vacuum (−IP) directly; CBM = VBM + best Eg. | critical | fixed 2026-10-03 (CHANGELOG #13): rule-based blend; ML did not beat Butler–Ginley on offsets |
| B2 | **Train/inference gap mismatch.** Edge features include the label's own DFT gap at training time but an experimental/ML gap at prediction time (mean difference 1.18 eV). Scored the way the app uses it, VBM MAE is about 0.69 eV vs 0.29 eV reported. (F4) | high | fixed 2026-10-03 (CHANGELOG #13): pipeline gap fed everywhere |
| B3 | **OptB88 and TBmBJ surface rows mixed in the edge aggregation.** Medians are taken over both methods for the same material, giving inconsistent CBM/gap pairs that the cleaning step then drops (only 126 of 322 surface materials survive). (F6) | medium | fixed (#4, #13): one record per surface; edge training removed |
| B4 | **Edges ignore crystal structure**, although structure and surface drive most of their variation. Structure features already exist in the DB. (F10) | medium | tested 2026-10-03 (#13): structure features did not improve offsets; not used |
| B5 | **Edges are bulk properties in the model but surface properties in reality.** Same material, different surfaces: VBM range median about 1.4 eV (uncorrected data). No orientation input, no range reported. (F5) | medium | open |
| B7 | **Training drops well-measured materials over single bad records.** `clean.py` removes any material whose experimental gaps span > 1.0 eV (or IQR > 0.5 eV), so Si, GaAs and ZnO are excluded from band-gap training because of one stray value each (e.g. GaAs 15.11 eV). Should use the median and drop outlier records, as the lookup now does (CHANGELOG #5). | high | fixed 2026-10-03 (CHANGELOG #8): shared `consensus_gap`; Borlido MAE 0.71 → 0.62 eV |
| B8 | **Directly computed interface offsets are not used.** JARVIS has DFT offsets for 239 semiconductor pairs that agree with experiment to 0.22 eV, far better than any vacuum alignment, but predictions never look them up. (C3) | high | fixed 2026-10-03 (CHANGELOG #12): measured-offset MAE 0.91 → 0.41 eV where interface data exists |
| B6 | **Junction type uses Anderson's rule only.** Interface dipoles are ignored; even with perfect surface data the expected error is about 0.45 eV (InterMat). A learned pairwise correction could use interface data. | medium | open |

## C. Evaluation

| ID | Issue | Severity | Status |
|---|---|---|---|
| C1 | **Edge metrics have no baselines and mix data sources.** The reported 0.18 eV is dominated by Castelli, where Butler–Ginley alone scores 0.08 eV. Metrics should be per source, against a constant and a Butler–Ginley baseline, using the gap the app actually feeds in. | high | fixed 2026-10-03 (CHANGELOG #6): `evaluate edges`; deployed model MAE 3.29 eV vs Butler–Ginley 1.37 |
| C2 | **Borlido "holdout" is 41 materials.** 407 of 453 Borlido materials are also in the training sources, so only Borlido-only materials are held out. A real independent test needs all Borlido materials excluded from the evaluation model's training. | high | fixed 2026-10-03 (CHANGELOG #7): grouped 5-fold on all 453; formula-only MAE 0.71 eV vs HSE06 0.61, PBE 1.13 |
| C3 | **No end-to-end check of junction types.** Nothing measures whether Type I/II/III or offsets are right. JARVIS interface offsets (and later a curated experimental set) should be a standing benchmark. | high | fixed 2026-10-03 (CHANGELOG #9): 321 DFT interfaces + 21 measured offsets; pipeline 0.91 eV vs experiment |
| C4 | **No automated tests.** Physics helpers (Butler–Ginley, junction classification, vacuum alignment, formula parsing) have no tests, so regressions like A1 go unnoticed. | medium | fixed 2026-10-03 (CHANGELOG #11): 12 tests; caught a quarantine-filter bug on day one |

## D. Tool and usability

| ID | Issue | Severity | Status |
|---|---|---|---|
| D1 / S1 | **Uncertainty is a fixed 0.3 eV flag.** It should come from predicted per-layer errors and be reported as a junction-type probability, so close calls go to DFT. Promoted to high for screening (agreed 2026-10-03). | high | fixed 2026-10-03 (CHANGELOG #15): calibrated type probabilities; confident calls right 83% |
| D2 | **Formula-only input.** No CIF / database-ID input, so users can't specify the phase. (F10) | medium | open |
| D3 | **Ambiguity isn't shown.** When a formula has conflicting values (polymorphs, A4) or edges vary strongly by surface (B5), the UI shows one number with no range. | medium | fixed 2026-10-03 (CHANGELOG #16): notes, VBM spread, probabilities and offset source shown |

### Screening-specific items considered (2026-10-03)

| ID | Item | Decision |
|---|---|---|
| S2 | "Unfamiliar chemistry" warning (applicability domain) | deferred by user: not needed now |
| S3 | Batch screening mode (candidate list against a fixed partner, ranked output) | deferred by user: not needed now |
| S4 | Device-aware scoring against target offset windows (e.g. ETL CBO 0–0.3 eV) | deferred by user: not needed now |

## E. Code hygiene

| ID | Issue | Severity | Status |
|---|---|---|---|
| E2 | **Slow lookups.** The `materials_best` view ranks the whole records table (263k rows) on every query: about 1.5 s per material lookup, already present in the original code. Matters when screening many candidates. | medium | fixed 2026-10-03 (CHANGELOG #10): ~3 ms per lookup |
| E1 | Unused code and leftover files (F11): unused import/helper/parameter, duplicate `borlido_expt` loader entry, redundant `GroupKFold`, `catboost_info/`, `data/exports/`, Vite template assets. | low | code done 2026-10-03 (CHANGELOG #17); leftover files await the user |

---

## Fix order (revised for screening, 2026-10-03)

1. Database: A1 ✓ → A2 (quarantine hypothetical sources rather than delete) → A4/A3 (confirm rules) → A5 → A6; A7 curated in parallel
2. Evaluation scaffolding before the new model: C1 honest edge metrics, C2 Borlido benchmark, C3 junction benchmark
3. Edge model rebuild: B1–B3
4. D1/S1 uncertainty and junction-type probabilities
5. Later: B4/D2 structure input, B5, B6 interface correction, C4 tests, E1 cleanup
