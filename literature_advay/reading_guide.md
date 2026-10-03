# Reading guide

What to understand before working on MaterialStack, in the order it builds up. Each concept says
**why it matters for this project**. Readings are tiered:
**★ = read first** (core, short), **◆ = read when you work on that part**, **○ = reference / optional**.
Citations marked *(verify)* are from memory and not yet checked against the publisher.

---

## Part 1: Energy levels of a single material

### Concepts
- **Band structure, VBM, CBM, band gap.** Direct vs indirect gaps.
  *Why:* VBM/CBM are the two numbers we predict per layer; Eg = CBM − VBM ties them together.
- **Fundamental vs optical gap; exciton binding energy.** Optical absorption measures the optical gap, slightly below the fundamental (transport) gap. Small for inorganics, large (0.3–1 eV) for organics.
  *Why:* many "experimental gaps" in our data are optical; organic HOMO/LUMO gaps aren't directly comparable to inorganic Eg.
- **Vacuum level, ionization potential (IP), electron affinity (EA), work function (φ), Fermi level.** IP = E_vac − VBM, EA = E_vac − CBM, φ = E_vac − E_F.
  *Why:* our "VBM vs vacuum" is just −IP. Bug F1 was exactly confusing these references.
- **IP and EA are surface properties, not bulk ones.** Surface dipoles, termination, reconstruction and orientation shift them by tenths of an eV to over 1 eV.
  *Why:* the main irreducible error of composition-only models (F5).
- **Electronegativity and the Butler–Ginley estimate**: midgap ≈ −χ_Mulliken (geometric mean).
  *Why:* our ML predicts a correction on top of this baseline.
- **Doping and Fermi-level position.** Shifts E_F (and work function) but not, to first order, IP/EA.
  *Why:* explains why experimental work functions scatter more than IPs.
- **Polymorphs.** Same formula, different structure, different gap and edges (anatase vs rutile TiO₂).
  *Why:* composition-only data can't distinguish them; this is what the `--polymorph` option deals with.
- **Temperature, strain and spin–orbit coupling.** SOC lowers the CBM of Pb/Bi/Sn halides by about 1 eV.
  *Why:* computed gaps of halide perovskites without SOC are unreliable (e.g. our MAPbI3 problem).

### Readings
- ★ **Kahn, "Fermi level, work function and vacuum level", *Materials Horizons* 3, 7 (2016).** Four pages; the clearest explanation of the energy references used everywhere in this project.
- ★ Kittel, *Introduction to Solid State Physics*, chapters on energy bands and semiconductor crystals. Skim for band structure and VBM/CBM if rusty.
- ◆ Butler & Ginley, *J. Electrochem. Soc.* 125, 228 (1978). Origin of the baseline formula.
- ○ Xu & Schoonen, *Am. Mineral.* 85, 543 (2000). Practical table of band positions; good to see how the field reports edges.

---

## Part 2: Heterojunctions and band alignment

### Concepts
- **Band offsets (CBO/VBO) and alignment types**: Type I (straddling), II (staggered), III (broken gap).
  *Why:* the output of the tool.
- **Anderson's rule (electron-affinity rule) / vacuum-level alignment**: align isolated materials at a common vacuum level.
  *Why:* this is exactly what `classify_junction` does; know its limits.
- **Interface dipoles, charge-neutrality level, Fermi-level pinning** (Tersoff). Real interfaces form dipoles that shift offsets away from Anderson's prediction.
  *Why:* even with perfect IP/EA, vacuum alignment is off by about 0.3–0.5 eV (InterMat: 0.45 eV vs direct interface calculations at 0.22 eV).
- **Transitivity of band offsets**: ΔE(A/C) ≈ ΔE(A/B) + ΔE(B/C). Holds for many semiconductor families, fails with strong interface chemistry.
- **Band bending and depletion regions.** Offsets are defined at the interface; band bending comes from doping/Fermi-level equilibration and is a separate effect.
  *Why:* don't confuse a device band diagram with the band offset we predict.
- **How offsets are measured**: XPS core-level method (Kraut), UPS for IP, inverse photoemission (IPES) or LEIPS for EA. Each has an uncertainty of about 0.1–0.2 eV, and reported values for the same interface often differ more than that.
  *Why:* sets the realistic accuracy target.

### Readings
- ★ **Sze & Ng, *Physics of Semiconductor Devices*, section on heterojunctions.** Standard band-diagram treatment, Anderson model.
- ★ Anderson, "Experiments on Ge–GaAs heterojunctions", *Solid-State Electronics* 5, 341 (1962). The original vacuum-alignment rule; short.
- ★ Tersoff, "Theory of semiconductor heterojunctions: the role of quantum dipoles", *Phys. Rev. B* 30, 4874 (1984). Why Anderson's rule fails; charge-neutrality levels.
- ◆ Kraut, Grant, Waldrop & Kowalczyk, *Phys. Rev. Lett.* 44, 1620 (1980). XPS method for measuring VBO *(verify)*.
- ◆ Choudhary & Garrity, InterMat, *Digital Discovery* (2024). The source of our surface and interface data; read the methods and benchmark tables.
- ○ Schulz, Cahen & Kahn, "Halide perovskites: is it all about the interfaces?", *Chem. Rev.* 119, 3349 (2019) *(verify)*; arXiv:1812.04908. Energy-level alignment in perovskite cells; why measured values scatter.

---

## Part 3: Solar-cell device context

### Concepts
- **Layer roles**: absorber, electron transport layer (ETL), hole transport layer (HTL); n-i-p vs p-i-n.
- **Desired alignment**: ETL CBM slightly below the absorber CBM and VBM much deeper (blocks holes); mirror image for the HTL. Type II at each contact, with the right direction.
  *Why:* "Type II" alone isn't enough; the *direction and size* of the offsets decide whether a stack works.
- **Spike vs cliff**: a small positive CBO "spike" (≈0–0.3 eV) is tolerable or helpful; a "cliff" increases recombination and lowers Voc (classic CIGS/CdS result).
  *Why:* this is what ~0.1–0.3 eV accuracy is needed *for*; explains the 0.3 eV "uncertain" threshold.
- **Shockley–Queisser limit, Voc loss.** Why band gap and offsets map to efficiency.
- **Organic layers** (Spiro-OMeTAD, PTAA, PCBM): described by HOMO/LUMO, measured by UPS/CV, no crystal formula.
  *Why:* our ML can't handle them; they need literature values.

### Readings
- ★ Nelson, *The Physics of Solar Cells* (or Würfel, *Physics of Solar Cells*). Chapters on p-n junctions and heterojunctions.
- ◆ Minemoto et al., "Theoretical analysis of the effect of conduction band offset of window/CIS layers on performance of CIS solar cells", *Sol. Energy Mater. Sol. Cells* 67, 83 (2001) *(verify)*. The spike/cliff result.
- ◆ Tao et al., "Absolute energy level positions in tin- and lead-based halide perovskites", *Nat. Commun.* 10, 2560 (2019) *(verify)*. Measured IP/EA for perovskites; candidate data source.
- ◆ Endres et al., "Valence and conduction band densities of states of metal halide perovskites", *J. Phys. Chem. Lett.* (2016) *(verify)*. Why perovskite IPs from UPS depend on how the band edge is extracted.
- ○ Jacobsson et al., Perovskite Database Project, *Nature Energy* 7, 107 (2022). Real device stacks.

---

## Part 4: How the computed data is made (DFT)

### Concepts
- **Kohn–Sham eigenvalues and the band-gap problem.** LDA/GGA (PBE, OptB88vdW) underestimate gaps by about 30–50% (derivative discontinuity).
  *Why:* explains the −1 eV bias of PBE/OptB88 sources (F8) and why they're features, not labels.
- **Fidelity ladder**: PBE / OptB88vdW < SCAN / r2SCAN < GLLB-SC, TB-mBJ < HSE06 / dielectric-dependent hybrids < GW. Know cost vs accuracy.
  *Why:* `METHOD_RANK` in `config.py` encodes this ladder.
- **How the gap correction splits**: going from PBE to hybrid/GW, the VBM typically moves down *and* the CBM moves up, not all on one side.
  *Why:* our Butler–Ginley baseline assumes a symmetric ½/½ split around −χ; whether that's right is worth checking (F4).
- **Slab calculations and the vacuum level**: planar-averaged electrostatic potential, vacuum plateau, slab thickness convergence, polar surfaces (spurious fields; dipole corrections).
  *Why:* this is how surfacedb's IP/EA are made; `avg_max` is the vacuum plateau.
- **Bulk + surface lineup (Van de Walle & Martin)**: bulk eigenvalues are relative to the average potential; a slab gives that average relative to vacuum.
  *Why:* the reason `scf_vbm` (bulk) can't be used against vacuum without a slab.
- **Hypothetical vs experimentally known structures; energy above hull.**
  *Why:* Castelli/Wolverton compute structures that don't exist for real materials (F2, F8).

### Readings
- ★ Borlido et al., "Large-scale benchmark of exchange–correlation functionals for the determination of electronic band gaps of solids", *JCTC* 15, 5069 (2019). Best single overview of which methods get gaps right.
- ★ Hinuma, Grüneis, Kresse & Oba, "Band alignment of semiconductors from density-functional theory and many-body perturbation theory", *PRB* 90, 155405 (2014) *(verify)*. How IP/EA are computed properly; how corrections split between edges.
- ◆ Van de Walle & Martin, "Theoretical study of band offsets at semiconductor interfaces", *PRB* 35, 8154 (1987). Bulk + interface lineup method.
- ◆ Kiyohara, Hinuma & Oba, *JACS* (2024). High-quality oxide IP/EA data + ML on surfaces.
- ○ Tran & Blaha, TB-mBJ, *PRL* 102, 226401 (2009); Heyd, Scuseria & Ernzerhof, HSE, *J. Chem. Phys.* 118, 8207 (2003). Only if you want the functionals themselves.
- ○ Perdew & Levy, *PRL* 51, 1884 (1983). The derivative discontinuity, i.e. why DFT gaps are too small *(verify)*.

---

## Part 5: Machine learning for materials

### Concepts
- **Composition descriptors** (Magpie: element-property statistics) vs **structure descriptors** vs **graph neural networks** (CGCNN, MEGNet, ALIGNN).
  *Why:* we use Magpie + LightGBM; know what GNNs add (structure) and need (a crystal structure per query).
- **Data leakage and grouped cross-validation**: random splits overestimate accuracy when similar compounds sit in train and test. Grouping by element set, leave-one-cluster-out.
  *Why:* `GroupKFold` by element set is a core design decision here.
- **Label quality beats model choice**: noise floor (≈0.2 eV between experimental compilations), duplicates, formula-derived labels (Castelli).
  *Why:* our biggest accuracy problems are in the data, not the model.
- **Multi-fidelity / transfer learning**: learn from plentiful low-fidelity (PBE) data, correct with scarce high-fidelity data (experiment/HSE).
  *Why:* the DFT-proxy features and the Δ-learning edge model are both forms of this.
- **Δ-learning (residual learning)**: predict the correction to a physical baseline (here Butler–Ginley) instead of the raw value.
- **Two-stage models**: metal/non-metal classifier, then a gap regressor (what `TwoStageGapModel` does).
- **Uncertainty quantification**: ensembles, quantile regression, conformal prediction.
  *Why:* the junction "UNCERTAIN" flag should come from predicted error, not a fixed 0.3 eV.
- **Metrics**: MAE vs RMSE, R², and comparing against trivial baselines (constant, formula-only).
  *Why:* F3 found the edge model doesn't beat a constant on real data. Always report a baseline.

### Readings
- ★ Wang, Kauwe, Murdock & Sparks, "Machine learning for materials scientists: an introductory guide toward best practices", *Chem. Mater.* 32, 4954 (2020). Practical pitfalls; very relevant.
- ★ Ward, Agrawal, Choudhary & Wolverton, "A general-purpose machine learning framework for predicting properties of inorganic materials", *npj Comput. Mater.* 2, 16028 (2016). The Magpie features we use.
- ★ Zhuo, Mansouri Tehrani & Brgoch, *JPCL* 9, 1668 (2018). ML band gaps from composition; source of our labels.
- ◆ Meredig et al., "Can machine learning identify the next high-temperature superconductor? Examining extrapolation performance for materials discovery", *Mol. Syst. Des. Eng.* 3, 819 (2018) *(verify)*. Leave-one-cluster-out CV.
- ◆ Dunn, Wang, Ganose, Dopp & Jain, "Benchmarking materials property prediction methods: the Matbench test set and Automatminer reference algorithm", *npj Comput. Mater.* 6, 138 (2020). How gap prediction is benchmarked.
- ◆ Chen, Zuo, Ye, Li & Ong, "Learning properties of ordered and disordered materials from multi-fidelity data", *Nat. Comput. Sci.* 1, 46 (2021). Multi-fidelity band gaps.
- ○ Ward et al., matminer, *Comput. Mater. Sci.* 152, 60 (2018). The library behind our datasets and features.
- ○ Xie & Grossman, CGCNN, *PRL* 120, 145301 (2018); Choudhary & DeCost, ALIGNN, *npj Comput. Mater.* 7, 185 (2021). If moving to structure-based models.

---

## Part 6: Recent work (2022–2025), the current state of the field

Parts 1–4 are old on purpose: the physics of band alignment was settled in 1962–1990 and textbooks
still teach it from those papers. Recent work changes the **data and ML** side, so read these to know
where the field is now and what to compare MaterialStack against. Verified 2026-10-03.

- ★ **Choudhary & Garrity, InterMat, *Digital Discovery* (2024)**, arXiv:2401.02021. Most relevant recent paper: DFT surfaces + interfaces + graph-neural-network band edges (0.26 eV). The closest published system to MaterialStack; treat it as the benchmark to beat.
- ★ **Kiyohara, Hinuma & Oba, *JACS* (2024)**. Surface-aware neural network for oxide IP/EA (about 0.22 eV) on about 2,900 hybrid-functional surfaces. Shows the surface plane as input is what gets edge errors near 0.2 eV.
- ◆ **Ye et al., "Machine learning for energy band prediction of halide perovskites", *Materials Futures* 4, 035601 (2025)**, doi:10.1088/2752-5724/adeead. XGBoost on HSE-computed halide perovskite CBM/VBM: MAE 0.15 eV (CBM) / 0.15 eV (VBM), gap 0.28 eV. Directly relevant to your absorber layers; labels are DFT, not experiment. Open access with supplementary data (check whether the dataset is included).
- ◆ **Wang et al., "Benchmarking band gap prediction for semiconductor materials using multimodal and multi-fidelity data", ICLR 2025 AI4Mat workshop**. 60,218 computed + 1,183 experimental gaps, seven models, leave-one-material-out validation. Public code/data: github.com/Shef-AIRE/bandgap-benchmark. Good template for honest evaluation.
- ◆ **Dong & Cole, "Auto-generated database of semiconductor band gaps using ChemDataExtractor", *Sci. Data* 9, 193 (2022)** *(verify volume/page)*. BandgapDatabase1: about 100k text-mined experimental gap records from about 129k papers, 84% precision. Much larger than our experimental sets but noisier.
- ◆ "Accurate prediction of experimental band gaps from large language model-based data extraction", arXiv:2311.13778 (2023). Uses LLMs to extract experimental gaps from papers; the modern route to building experimental datasets such as IP/EA tables.
- ○ **Nair, Foppa & Scheffler, "Materials database from all-electron hybrid functional DFT calculations", *Sci. Data* 12, 1518 (2025)**. HSE06 (FHI-aims) for 7,024 materials; open ASE database on figshare. Another high-fidelity gap source next to SNUMAT.
- ○ Liu et al., "A simple denoising approach to exploit multi-fidelity data for machine learning materials properties", *npj Comput. Mater.* (2022). How to combine cheap and expensive gap data.

**Takeaway:** recent ML papers report edge errors of about 0.15–0.26 eV, but against **DFT labels** for
**one material family** or with **surface information**. Nobody has a general, experimentally validated
band-alignment predictor; that's the gap your project can address honestly.

---

## Suggested order (about two weeks part-time)

0. **Before anything (½ day):** skim Part 6 (InterMat, Kiyohara 2024, Ye 2025) to see where the field is now; come back to them properly in step 3.
1. **Days 1–2:** Kahn 2016 → Sze heterojunction section → Anderson 1962 → Tersoff 1984. *Goal:* explain IP/EA/φ and why Anderson's rule fails.
2. **Days 3–4:** Nelson or Würfel heterojunction chapters → Minemoto spike/cliff. *Goal:* say what alignment a good ETL/HTL needs.
3. **Days 5–7:** Borlido 2019 → Hinuma 2014 → InterMat. *Goal:* know where every number in our database comes from and how wrong it can be.
4. **Days 8–10:** Wang 2020 best practices → Ward 2016 → Zhuo 2018 → Meredig 2018. *Goal:* defend our CV setup and metrics.
5. **Then:** Kiyohara 2024 and the perovskite energy-level papers when you start on new edge data.

## Self-check: can you answer these?

- Why is "VBM vs vacuum" a surface property, and how much can it vary for one material?
- Given IP and EA of two materials, classify their junction. What does the result ignore?
- Why are PBE gaps too small, and why doesn't that mean the PBE VBM is wrong by the same amount?
- Why does random K-fold CV overstate accuracy for materials, and what do we use instead?
- Why can't the Castelli edges be used to train a correction to Butler–Ginley?
- For a perovskite absorber, what CBO to the ETL is acceptable, and why does ~0.3 eV error matter?
