# Improvement plan

Goal: **a band-alignment screening tool for devices that is accurate where it can be and honest where
it can't.** Written 2026-10-03, after the database fixes (CHANGELOG #1–#5), the evaluation suite
(#6, #7, #9) and consensus labels (#8). Issue IDs refer to [critique.md](critique.md); every finished
step gets a CHANGELOG entry.

## How junction accuracy is decided

A junction's valence-band offset comes from the two layers' VBMs. In order of accuracy (vs the 21
measured offsets in `evaluate junctions`):

| Source of the offset | Accuracy today | Coverage |
|---|---|---|
| Directly computed interface (JARVIS ASJ) | 0.22 eV | 239 semiconductor pairs; unused (B8) |
| Vacuum alignment of slab VBMs, good surfaces | 0.47 eV (paper) | 256 materials |
| Vacuum alignment, our median over all surfaces | 0.91 eV | same (A9) |
| Butler–Ginley (−χ − Eg/2) | 0.58 eV | every material |
| Deployed ML edge model | worse than a constant (3.29 eV on VBM) | every material (B1–B3) |

So the work is: use the best source available for each pair, fix the weak tiers, and turn the
remaining uncertainty into a probability that tells the user when to run DFT.

## Benchmarks every step is judged on

- `evaluate gap`: Borlido, formula-only MAE (now 0.62 eV) — must not get worse.
- `evaluate edges`: slab VBM reference (deployed model 3.29 eV, Butler–Ginley 1.37 eV).
- `evaluate junctions`: 21 measured offsets (pipeline 0.91 eV, Butler–Ginley 0.58, ASJ 0.22) and
  321 DFT interfaces (pipeline 1.31 eV, zero baseline 0.86, type agreement 44%).
- Snapshot panel (`snapshot_predictions.py`) for before/after on 27 device materials.

Offsets for the ML path are always scored with **out-of-fold** VBMs (materials never seen in
training), because screened candidates are new materials.

## Phases

| # | Step | Issues | Done when |
|---|---|---|---|
| 0.1 | Fast lookups: filter by material before ranking, instead of ranking the whole table | E2 | a lookup takes < 0.1 s (was ~1.5 s) |
| 0.2 | Small test suite for the physics and data rules (consensus, vacuum alignment, junction types, quarantine) | C4 | `pytest` passes; runs in seconds |
| 1.1 | Representative surface per material: test median / lowest surface energy / other policies against measured and DFT offsets; pick one on physical grounds the data supports | A9 | vacuum-alignment offsets beat Butler–Ginley on measured offsets |
| 1.2 | Use directly computed interface offsets for pairs that have them; junction code reads offsets from one clearly named place | B8 | those pairs reproduce ASJ; source shown in output |
| 2.1 | Rebuild the edge model: target = representative slab VBM; CBM = VBM + best gap; gap fed exactly as at prediction time; small, regularised model; structure features as a variant | B1, B2, B3, B4 | out-of-fold offsets beat Butler–Ginley on both junction benchmarks; else ship Butler–Ginley |
| 2.2 | (optional data) Kiyohara/Oba hybrid-functional oxide surfaces, if the supporting files can be obtained — **blocked**: ACS returns 403 to automated download and no public mirror exists; needs a manual download from the paper page | B1, A7, A10 | oxide transport layers covered by hybrid-level data |
| 3 | Retrain all models once (gap models with consensus labels, new edge model); refresh `metrics.json`; full `evaluate`; snapshot | B7, B1 | all benchmarks recorded in CHANGELOG |
| 4 | Junction-type probabilities from per-source error estimates measured in the benchmarks; "run DFT" flag for close calls | D1/S1 | probabilities calibrated on the DFT interface set |
| 5 | Show it in the tool: API/UI show sources, ranges, ambiguity notes, probabilities | D3 | UI displays everything `predict` knows |
| 6 | Cleanup and docs | E1, docs | dead code gone; README/CLAUDE.md current |

Not in this plan (by decision): A3 spin–orbit (on hold), A7 experimental band edges (deferred),
S2–S4 (deferred), B5/D2 orientation and CIF input, B6 learned interface-dipole correction.

## Principles

- One rule, one place: e.g. `consensus_gap` serves lookup and training; junction types come from one
  function.
- Every number the tool shows says where it came from.
- A step that doesn't beat its baseline is not shipped; the baseline is shipped instead.
- Plain code over clever code: small functions with docstrings that say *why*.

## Status (2026-10-03, end of day)

| # | Step | Status |
|---|---|---|
| 0.1 | Fast lookups | done (#10) |
| 0.2 | Test suite | done (#11; 19 tests) |
| 1.1 | Surface selection | studied, no change (F18) |
| 1.2 | Interface offsets | done (#12) |
| 2.1 | Edge rebuild | done (#13): slab + Butler–Ginley blend; no ML edge model |
| 2.2 | Kiyohara oxide data | done (#20) via Europe PMC; weight 0.8 chosen with user |
| 3 | Retrain | done (#14) |
| 4 | Junction probabilities | done (#15) |
| 5 | UI / API | done (#16); not checked visually |
| 6 | Cleanup and docs | done except leftover files awaiting the user (#17) |

Headline numbers: band offsets vs 21 measured 0.91 → 0.41 eV (0.48 without interface data);
vs 321 DFT interfaces 1.31 → 0.79 eV, type right 44 → 64%; confident calls (≥ 80%) right 83%;
band gap on Borlido 0.71 → 0.62 eV (formula only).

Later the same day: A3 restored (#18), unused files removed (#19), Kiyohara oxides (#20), curated
measured edges (#21). Still open: NiO and CuI edges (A10), organic layers, oxide measured values (A7).

## Status (2026-10-06): simplified pipeline (CHANGELOG #25, #26)

Measured levels for NiO, CuI, CuSCN, oxides and organics were curated (#25). The project was then
rebuilt around CSV files, one LightGBM model and three scripts (#26). Current numbers
(`results/metrics.csv`): band gap 0.40 eV (grouped CV); offsets vs 21 measured 0.45 eV; gold stacks
ΔEc/ΔEv sign 80/83 %, confident calls 82 %. Next levers, in order: more measured IE/EA (CdS, Cu2O,
CIGS, CZTS, Sb2Se3, Cs2AgBiBr6, PTAA, P3HT), more measured interface offsets, then polymorph-aware gaps.
