# Datasets

Numbers below come from `scripts/source_credibility.py` run on the database as of 2026-10-03.
"Error vs experiment" = mean absolute error against the median experimental gap of the same formula
(non-metals only). Formula-level matching ignores polymorphs, which inflates errors for sources built
on hypothetical structures.

## 1. In use today

### Experimental band gaps (training labels for the Eg model)

| Source key | Rows / materials | Origin | Credibility | Notes |
|---|---|---|---|---|
| `expt_gap` | 6,353 / 4,920 | Zhuo, Mansouri Tehrani & Brgoch 2018, via matminer | **Medium-High** | Composition only, no phase (anatase = rutile). 39% are metals (gap 0). 462 formulas have several values; median spread 0.17 eV. |
| `expt_gap_kingsbury` | 4,604 / 4,604 | Kingsbury et al. 2022, cleaned Zhuo set matched to Materials Project IDs, via matminer | **Medium-High** | **95% identical to `expt_gap`** on 2,153 shared materials: a duplicate, not independent evidence. |
| `foundry_ml_exp_bandgaps` | 2,069 / 1,439 | Foundry-ML aggregation (via JARVIS loader) | **Medium** | Agrees with Zhuo 67–79% of the time (MAE 0.14 eV); exact provenance unclear. |
| `borlido_expt` | 472 / 453 | Borlido et al. 2019, curated benchmark for DFT functionals | **High** | Best-curated experimental set. Differs from Zhuo by MAE 0.22 eV, i.e. experimental labels themselves carry about 0.2 eV noise. Held out of training. |

Net: about 5–6k unique experimental gaps, mostly from one compilation.

### Computed band gaps (lookup + model features)

| Source key | Method | Materials | Error vs expt (n) | Bias | Credibility |
|---|---|---|---|---|---|
| `snumat` | HSE06 | 8,730 | 0.62 eV (1,004) | −0.05 | **High** for computed |
| `jarvis_dft_3d` | HSE06 | 56 | 0.59 (23) | −0.16 | High, tiny |
| `jarvis_dft_3d` | TBmBJ | 16,624 | 0.81 (772) | −0.34 | **Medium-High**, largest good set |
| `snumat` | PBE | 8,730 | 0.98 (1,004) | −0.90 | Medium (systematic underestimate) |
| `mp_nostruct_20181018` | PBE | 56,750 | 1.10 (1,391) | −0.98 | Medium (feature only) |
| `jarvis_dft_3d` | OptB88vdW | 65,321 | 1.15 (1,313) | −1.07 | Medium (feature only) |
| `dielectric_constant` | PBE | 964 | 1.24 (226) | −1.13 | Medium |
| `jarvis_halide_perovskites` | HSE06 / PBE | 229 | n/a | n/a | **Low for hybrid perovskites**: gives MAPbI3 2.64 eV vs about 1.6 eV experimentally |
| `wolverton_oxides` | PBE | 2,646 | 1.90 (136) | −1.83 | **Low**: hypothetical ABO3 perovskites |
| `double_perovskites_gap` | GLLB-SC | 1,216 | n/a | n/a | Low-Medium: hypothetical double perovskites |
| `castelli_perovskites` | GLLB-SC | 9,646 | 1.96 (89) | −1.96 | **Low**: hypothetical cubic ABX3; 96% metallic |

### Band edges (VBM / CBM vs vacuum): the core of band alignment

| Source key | Rows | What it really is | Credibility |
|---|---|---|---|
| `castelli_perovskites` | 18,928 | Edges **derived from the Butler–Ginley electronegativity formula** + GLLB-SC gap (our CBM matches the formula to a median 0.03 eV, corr 0.95). Hypothetical cubic perovskites. | **Low**: circular as ML labels; also supplies the edges shown for In2O3 and Ga2O3 via fake "InInO3"/"GaGaO3" perovskites. |
| `jarvis_surfacedb` | 995 (607 surfaces, 322 materials) | OptB88vdW slab calculations (InterMat). Paper: work function MAE 0.29 eV, EA MAE 0.39 eV vs experiment. TBmBJ rows are *our* construction: CBM = surface VBM + bulk TBmBJ gap. | **High source, broken copy**: loader stores `surf_vbm` without subtracting the vacuum level `avg_max` (see findings 2026-10-03). About 120 surfaces have nonsensical vacuum levels (`avg_max` up to 167 eV). |
| `jarvis_interfacedb` | 593 (425 with numeric offset) | Directly computed interface (ASJ) valence-band offsets. Paper: MAE 0.22 eV vs experiment (vacuum-alignment rule: 0.45 eV). Convention: positive ΔEv at A/B means VBM higher in B. | **High**, but **unused** by the pipeline. Best available end-to-end validation set. |
| `literature_csv` | 0 | User-supplied experimental edges | **Empty**: no experimental edge data in the DB. |

### Not loaded

- `materials_project` (live API; needs key): PBE/r2SCAN gaps, no vacuum-referenced edges. Low added value.
- `matbench_mp_gap`, `mp_all_20181018` (`--heavy`): more PBE. Redundant.

## 2. Candidate sources

| Source | What it gives | Why it matters | Access | Status |
|---|---|---|---|---|
| **Kiyohara, Hinuma & Oba, JACS 2024** | IP + EA for 2,195 binary + 718 ternary **oxide surfaces** (about 127 + 344 oxides), dielectric-dependent hybrid functional; also formula, space group, Miller index, surface energy, bulk gap | Best computed edge training data for oxides (most ETLs/HTLs); hybrid-level, surface-resolved. Replace Castelli with this. | Supporting-information Excel/ZIP on the ACS page (ACS website, may need browser download) | idea |
| **Experimental IP/EA (UPS/IPES) curated by us** | Measured VBM/CBM vs vacuum for device layers | The only way to get *trusted* edge lookups and an honest test set. Even 150–300 values beats all current edge data. | Manual curation into `data/literature/*.csv` (loader already exists) | idea |
| **Xu & Schoonen 2000** | CB/VB positions of about 50 oxide + 50 sulfide minerals | Classic table, easy to digitise. **Caveat:** many entries are themselves electronegativity estimates; keep only measured ones. | Paper (Am. Mineral.) | idea |
| **JARVIS surfacedb / interfacedb (fixed)** | Correct IP/EA per surface; 425 DFT band offsets | Fix vacuum alignment; use interfaces as junction-type test set | Already downloaded via `jarvis-tools` | idea |
| **C2DB** | Vacuum-referenced VBM/CBM (PBE, HSE06, G0W0) for about 1,500+ 2D monolayers | Clean edges for pre-training; 2D ≠ bulk thin films, so not direct labels | c2db.fysik.dtu.dk (ASE db download) | idea |
| **Perovskite Database Project** | About 42k published perovskite devices: full stacks, measured perovskite gaps, performance | Realistic stacks to test on; experimental perovskite gaps (fixes MAPbI3-type errors). Energy levels per layer not confirmed. | perovskitedatabase.com (open) | idea |
| **HOPV15** | Experimental HOMO/LUMO for 350 OPV molecules | Organic layers can't be handled by composition ML; mostly donor polymers, not Spiro/PTAA/PCBM | figshare (open) | idea |
| **BandgapDatabase1** (Dong & Cole, Sci. Data 2022) | About 100k experimental band-gap records text-mined from about 129k papers, with temperature; 84% precision | Much larger and more recent than Zhuo; covers device materials. Noisy, so filter and cross-check against Borlido before use. | Open: figshare, github.com/QingyangDong-qd220/BandgapDatabase1 | idea |
| **Nair, Foppa & Scheffler, Sci. Data 2025** | HSE06 + PBEsol (all-electron FHI-aims) for 7,024 materials | Second high-fidelity gap source next to SNUMAT; recent | Open: figshare ASE databases | idea |
| **Shef-AIRE bandgap benchmark** (ICLR 2025 workshop) | 60,218 computed + 1,183 experimental gaps, curated split | Ready-made benchmark to compare our gap model against published ones | Open: github.com/Shef-AIRE/bandgap-benchmark | idea |
| **Ye et al., Materials Futures 2025** | HSE CBM/VBM/gap for halide perovskites | Edge data for absorber layers; check whether the SI includes the dataset and whether edges are vacuum-referenced | Open access + SI | idea |
| "Full prediction of band potentials" (2024) | Transfer learning from 2D data to measured bulk band potentials | May contain a compiled set of measured bulk band potentials; check SI | ScienceDirect (paywalled) | idea |
