# Demo stacks and expected outputs

Recorded 2026-10-03 from the current pipeline (after CHANGELOG #21). Enter the layers top-first, one per
line, in the UI Lab (http://127.0.0.1:8001) or run `python -m materialstack predict A B C`.
Energies in eV vs vacuum. Probabilities use a fixed random seed, so they are reproducible.
If a number here stops matching, something in the pipeline changed — check CHANGELOG.md.

## 1. Perovskite n-i-p cell — `TiO2`, `MAPbI3`, `Cu2O`

| Layer | Eg | VBM | CBM | Where it comes from |
|---|---|---|---|---|
| TiO2 | 3.17 | −7.76 | −4.60 | gap: Borlido; edges: hybrid oxide surfaces (Kiyohara) + Butler–Ginley |
| MAPbI3 | 1.59 | −5.93 | −4.36 | measured (Tao 2019) |
| Cu2O | 2.17 | −5.11 | −2.94 | gap: Borlido; edges: hybrid oxide surfaces + Butler–Ginley |

- TiO2 | MAPbI3: **Type II**, 58% → check with DFT. CBO +0.26 (TiO2 CBM 0.26 eV below the perovskite's →
  electrons extracted), VBO −1.83 (holes blocked). The textbook ETL alignment.
- MAPbI3 | Cu2O: **Type II**, 75% → check with DFT. VBO −0.82, CBO +1.40 (electrons blocked).

Talking points: three different data sources in one stack; MAPbI3 note that computed gaps are not
trusted for Pb compounds; TiO2 note that its surface VBM varies by ~5 eV over 259 surfaces.

## 2. SnO2 ETL — `SnO2`, `FAPbI3`

- SnO2: Eg 3.60, VBM −8.95, CBM −5.35 (hybrid surfaces). FAPbI3: Eg 1.51, VBM −6.24, CBM −4.74 (measured).
- **Type II**, 67% → check with DFT. VBO −2.71, CBO +0.62 (large conduction-band cliff into SnO2).

## 3. All-inorganic perovskite — `TiO2`, `CsPbBr3`

- CsPbBr3: Eg 2.31, VBM −6.53, CBM −4.17 (measured edges).
- **Type II**, 72% → check with DFT. VBO −1.23, CBO +0.38.

## 4. CdTe thin film — `CdS`, `CdTe`

- CdS: Eg 2.48, VBM −7.00, CBM −4.52. CdTe: Eg 1.48, VBM −5.84, CBM −4.36 (slab + Butler–Ginley).
- **Type II**, 57% → check with DFT. VBO −1.16, CBO +0.16 (a small offset near the I/II boundary,
  hence the low confidence).

## 5. CIS cell — `ZnO`, `CdS`, `CuInSe2` (shows both junction routes)

- ZnO | CdS: **Type II, 95%, confident** — offset from a directly computed JARVIS interface
  (VBO +0.48, CBO −1.44).
- CdS | CuInSe2: **Type II, only 39%** → check with DFT (VBO −1.81, CBO +0.37); CuInSe2 edges are
  Butler–Ginley only, so the probabilities are spread (I 33%, II 39%, III 28%).

Talking point: the same stack contains a confident call (interface data) and an honest "don't know".

## 6. Interface DFT lookup — `Si`, `GaAs`

- Both layers have measured edges (electron affinities). Offset from 2 JARVIS interface calculations:
  VBO +0.25 (measured ≈ 0.23), CBO +0.10.
- **Type I at 42%** (II 58%) → check with DFT: both offsets are a few tenths of an eV, right at the
  Type I/II boundary. Good example of the tool refusing to over-claim.

## 7. Confident interface call — `GaN`, `ZnO`

- **Type II, 89%, confident.** VBO +0.40 (measured ≈ 0.70), CBO −0.46, from 3 interface calculations.

## 8. Lead-free absorber — `TiO2`, `Cs2AgBiBr6`

- Cs2AgBiBr6: Eg 2.19 (experiment), edges Butler–Ginley only (VBM −6.37, CBM −4.18).
- **Type II**, 67% → check with DFT. VBO −1.40, CBO +0.42.

## 9. A material nobody has measured — `TiO2`, `Cs2AgSbBr6`

- Cs2AgSbBr6 is **not in the database**: gap from the composition-only ML model (2.21 eV, metal
  probability ~0), edges from Butler–Ginley (VBM −6.46, CBM −4.25).
- **Type II**, 64% → check with DFT. This is the screening use case: instant estimate, honest confidence.
- Variant: `K2AgSbBr6` has a JARVIS structure but no trusted gap → structure-aware ML (2.33 eV);
  note shows the space group used.

## 10. Phase ambiguity — `ZrO2`

- Experimental reports disagree (0.56, 0.65, 3.80, 4.99 eV): gap shown 4.39 but **not trusted**, with a
  note asking for the phase. (CsPbI3 used to show this; the curated measured value now resolves it.)

## Known limitations (good to show honestly)

- `MAPbI3`, `NiO`: NiO gets VBM −7.76 (Butler–Ginley only), so the junction comes out **Type I, 69%**
  (NiO would block holes). In real cells NiO is a working hole-transport layer: this is wrong
  (critique A10, needs a citable measured value). CuI has the same problem.
- `MAPbI3`, `Spiro-OMeTAD`: organic layers have no formula → "missing band gap", no junction.
