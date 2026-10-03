# Papers

Grouped by topic. Each entry: what it is, then the takeaway for MaterialStack. BibTeX keys refer to
[references.bib](references.bib).

## Band edges and band alignment

**InterMat: Accelerating band offset prediction in semiconductor interfaces with DFT and deep learning**
(Choudhary et al., Digital Discovery 2024) `choudhary2024intermat`
- Source of our `jarvis_surfacedb` and `jarvis_interfacedb`. 607 OptB88vdW surfaces, 593 interface (ASJ) calculations.
- Definitions: IP = E_vac − VBM, work function φ = E_vac − E_F, EA = IP − bulk gap. E_vac is the plateau of the planar-averaged potential (`avg_max` in the dataset).
- Accuracy vs experiment: work function MAE 0.29 eV, EA 0.39 eV; band offsets from vacuum alignment (independent unit, Anderson's rule) 0.45 eV vs direct interface calculation (ASJ) 0.22 eV.
- Convention: positive valence-band offset at A/B means the VBM is higher in B.
- Notes that bulk eigenvalues (`scf_vbm`/`scf_cbm`) are relative to the cell-average potential and **not** comparable to vacuum.
- **Takeaway:** confirms our loader bug; gives a ceiling of about 0.45 eV for vacuum-alignment junctions even with perfect surface DFT, and shows interface data is the right validation target.

**Band alignment of oxides by learnable structural-descriptor-aided neural network and transfer learning**
(Kiyohara, Hinuma & Oba, JACS 2024) `kiyohara2024oxides`
- About 2,900 oxide surfaces with IP/EA from a non-self-consistent dielectric-dependent hybrid functional (PBE0-type with mixing = 1/ε∞).
- Their neural network uses bulk structure + surface termination plane: IP MAE 0.22 eV, EA MAE 0.23 eV (vs their DFT).
- **Takeaway:** best available training data for oxide edges; shows that adding the surface plane as input is what gets errors down to about 0.2 eV.

**Band alignment of semiconductors from density-functional theory and many-body perturbation theory**
(Hinuma, Grüneis, Kresse & Oba, PRB 2014) `hinuma2014alignment`
- IPs/EAs of about 30 semiconductors with GW corrections, compared with experiment.
- **Takeaway:** reference for how band-gap corrections split between VBM and CBM (our Butler–Ginley baseline assumes a symmetric ½/½ split).

**The absolute energy positions of conduction and valence bands of selected semiconducting minerals**
(Xu & Schoonen, Am. Mineral. 2000) `xu2000absolute`
- Table of CB/VB positions for about 100 oxide and sulfide minerals.
- **Takeaway:** quick source of literature edges, but many values are electronegativity-based estimates; only use those backed by measurement.

**Prediction of flatband potentials at semiconductor-electrolyte interfaces from atomic electronegativities**
(Butler & Ginley, J. Electrochem. Soc. 1978) `butler1978`
- Origin of CBM = −(χ − Eg/2) used as our baseline.
- **Takeaway:** a decent baseline, but the Castelli edge data is generated from essentially this formula, so it can't be used to correct it.

**Machine learning for energy band prediction of halide perovskites**
(Ye, Li, Qu et al., Materials Futures 4, 035601, 2025) `ye2025perovskite`
- XGBoost beats other shallow models, Transformers and MLPs on HSE-computed halide perovskite CBM/VBM (MAE 0.15 eV each) and gaps (0.28 eV); SHAP analysis of key features.
- **Takeaway:** gradient boosting is still competitive (supports our LightGBM choice); their errors are vs DFT within one family, so not directly comparable to a general experimental benchmark.

## Band gap data and DFT accuracy

**Benchmarking band gap prediction for semiconductor materials using multimodal and multi-fidelity data**
(Wang, Liu, Jungbluth, Ramadan, Oliver & Lu, ICLR 2025 AI4Mat workshop) `wang2025benchmark`
- 60,218 computed + 1,183 experimental gaps, 7 models, leave-one-material-out validation; public code and data.
- **Takeaway:** a recent, public benchmark to report our gap model against.

**Auto-generated database of semiconductor band gaps using ChemDataExtractor** (Dong & Cole, Sci. Data 2022) `dong2022bandgapdb`
- About 100k text-mined experimental gaps (84% precision, 65% recall).
- **Takeaway:** largest open experimental gap source; needs cleaning.

**Materials database from all-electron hybrid functional DFT calculations** (Nair, Foppa & Scheffler, Sci. Data 12, 1518, 2025) `nair2025hybrid`
- HSE06 for 7,024 materials with FHI-aims.
- **Takeaway:** a second recent HSE06 source next to SNUMAT.


**Predicting the band gaps of inorganic solids by machine learning** (Zhuo, Mansouri Tehrani & Brgoch, JPCL 2018) `zhuo2018`: source of `expt_gap`.

**Performance comparison of r2SCAN and SCAN metaGGA density functionals for solid materials** (Kingsbury et al., Phys. Rev. Mater. 2022) `kingsbury2022`: source cited for `expt_gap_kingsbury` (verify that this is the right citation for the dataset).

**Large-scale benchmark of exchange-correlation functionals for the determination of electronic band gaps of solids** (Borlido et al., JCTC 2019) `borlido2019`: curated 472-material experimental set; our cleanest held-out test.

**A database of HSE06 band gaps (SNUMAT)** (Kim et al., Sci. Data 2020) `kim2020snumat`: best large computed gap source in our check (MAE 0.62 eV vs experiment).

**Computational screening of high-performance optoelectronic materials using OptB88vdW and TB-mBJ formalisms** (Choudhary et al., Sci. Data 2018) `choudhary2018tbmbj`: JARVIS TBmBJ gaps.

**The joint automated repository for various integrated simulations (JARVIS)** (Choudhary et al., npj Comput. Mater. 2020) `choudhary2020jarvis`.

**New cubic perovskites for one- and two-photon water splitting** (Castelli et al., EES 2012) `castelli2012` and **New light-harvesting materials using accurate and efficient bandgap calculations** (Castelli et al., AEM 2015 / EES 2012; verify) `castelli2012b`: source of `castelli_perovskites`; hypothetical cubic structures, edges from electronegativity.

**Machine learning bandgaps of double perovskites** (Pilania et al., Sci. Rep. 2016) `pilania2016`.

**High-throughput DFT calculations of formation energy, stability and oxygen vacancy formation energy of ABO3 perovskites** (Emery & Wolverton, Sci. Data 2017) `emery2017`: `wolverton_oxides`.

**High-throughput screening of inorganic compounds for the discovery of novel dielectric and optical materials** (Petousis et al., Sci. Data 2017) `petousis2017`.

**Commentary: The Materials Project** (Jain et al., APL Mater. 2013) `jain2013mp`.

## Device and stack data

**An open-access database and analysis tool for perovskite solar cells based on the FAIR data principles**
(Jacobsson et al., Nature Energy 2022) `jacobsson2022`
- About 42k devices with layer stacks and measured perovskite gaps.
- **Takeaway:** source of realistic stacks and experimental perovskite gaps for testing.

**The Harvard organic photovoltaic dataset (HOPV15)** (Lopez et al., Sci. Data 2016) `lopez2016hopv`
- 350 molecules with experimental HOMO/LUMO.
- **Takeaway:** a possible route for organic layers, which composition-based ML cannot handle.

## 2D materials

**The Computational 2D Materials Database (C2DB)** (Haastrup et al., 2D Mater. 2018) `haastrup2018c2db`; **Recent progress of the C2DB** (Gjerding et al., 2D Mater. 2021) `gjerding2021c2db`
- Vacuum-referenced VBM/CBM with PBE, HSE06, G0W0 for about 1,500+ monolayers.
- **Takeaway:** clean edge data for pre-training; monolayer ≠ bulk film.
