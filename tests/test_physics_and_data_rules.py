"""Tests for the physics rules and the data-cleaning rules.

Run from the repo root:  .venv/bin/python -m pytest -q
Most tests use small hand-made inputs; the last ones need the built data (python build_data.py) and the
trained model (python train.py), and are skipped without them.
"""
import pandas as pd
import pytest

from build_data import clean_gaps, consensus
from materialstack.chem import butler_ginley_vbm, formula_key
from materialstack.config import BAND_GAPS, MODEL
from materialstack.predict import Layer, classify_junction, junction_type, type_probabilities


def reports(values, sources=None, formula="GaAs"):
    return pd.DataFrame({"name": formula, "formula": formula, "gap_ev": values,
                         "source": sources or ["Zhuo 2018"] * len(values), "reference": ""})


# --------------------------------------------------------------------------- band-gap cleaning (build_data.py)

def test_consensus_ignores_a_single_bad_report():
    label, outliers, _ = consensus(reports([1.42, 1.43, 1.44, 15.11, 1.40, 1.52]))
    assert label == pytest.approx(1.43, abs=0.02) and outliers.sum() == 1


def test_consensus_rejects_two_phases():
    # CsPbI3: black phase 1.67 eV vs yellow phase 2.76 eV, no majority
    assert consensus(reports([1.67, 2.76]))[0] is None


def test_consensus_counts_copied_values_once():
    # one number copied by three compilations is one measurement: 1.05 vs 0.25 has no majority
    assert consensus(reports([1.05, 1.05, 1.05, 0.25]))[0] is None


def test_consensus_prefers_curated_source():
    label, _, basis = consensus(reports([0.6, 1.12, 1.17, 4.19], ["Zhuo 2018", "Zhuo 2018", "Borlido 2019", "Zhuo 2018"]))
    assert label == pytest.approx(1.17) and basis == "Borlido 2019"


def test_cleaning_rules_reject_impossible_values_only():
    r = pd.DataFrame({"name": ["InSe", "InSe", "InSe", "UO2", "Ne", "MnO"], "gap_ev": [1.25, 1.3, 13.2, 2.1, 21.5, 3.6],
                      "source": "Zhuo 2018", "reference": ""})
    dft = pd.DataFrame({"formula": ["MnO"], "gap_gga": [0.3], "gap_hybrid": [2.9]})   # GGA underestimates: no conflict
    gaps, rejected = clean_gaps(r, dft)
    assert set(gaps.formula) == {"InSe", "MnO"}
    assert gaps.set_index("formula").gap_ev["InSe"] == pytest.approx(1.275)
    assert dict(zip(rejected.name, rejected.rule)) == {"InSe": "R4", "UO2": "R2", "Ne": "R3"}


def test_dft_conflict_rejects_metal_vs_insulator_only():
    r = pd.DataFrame({"name": ["BaMnO3", "CsF"], "gap_ev": [0.0, 10.0], "source": "Zhuo 2018", "reference": ""})
    dft = pd.DataFrame({"formula": ["BaMnO3", "CsF"], "gap_gga": [1.5, 6.0], "gap_hybrid": [3.04, 7.47]})
    gaps, rejected = clean_gaps(r, dft)
    assert dict(zip(rejected.name, rejected.rule)) == {"BaMnO3": "R6"}   # "0.00 eV" for an insulator: a typo
    assert list(gaps.formula) == ["CsF"]                                # 10 eV is right; DFT underestimates it


def test_formula_key_normalises_spelling_and_phase_words():
    assert formula_key("O2Ti") == formula_key("anatase TiO2") == formula_key("TiO2 film") == "TiO2"
    assert formula_key("Spiro-OMeTAD") is None


# --------------------------------------------------------------------------- physics (predict.py)

def test_butler_ginley_puts_the_gap_symmetrically_around_minus_chi():
    assert butler_ginley_vbm(5.0, 2.0) == pytest.approx(-6.0)


@pytest.mark.parametrize("top, bottom, expected", [
    ((-6.0, -3.0), (-5.5, -3.5), "I"),     # bottom gap inside top gap
    ((-6.0, -4.0), (-5.0, -3.0), "II"),    # staggered
    ((-5.0, -4.5), (-4.0, -3.0), "III"),   # bottom VBM above top CBM: broken gap
])
def test_junction_type(top, bottom, expected):
    assert junction_type(top[0], top[1], bottom[0], bottom[1]) == expected


def layer(vbm, gap, name="x", gap_kind="measured", edge_kind="measured"):
    return Layer(query=name, formula=name, display_formula=name, gap_ev=gap, gap_kind=gap_kind,
                 vbm_ev=vbm, cbm_ev=vbm + gap, edge_kind=edge_kind)


def test_offset_sign_conventions():
    # TiO2-like on MAPbI3-like: top VBM 2 eV deeper, top CBM 0.5 eV deeper → type II, electrons collect on top
    j = classify_junction(layer(-8.0, 3.2), layer(-6.0, 1.7), use_measured_edges=False)
    assert j["type"] == "II"
    assert j["vbo_ev"] == pytest.approx(-2.0)        # VBM(top) − VBM(bottom)
    assert j["cbo_ev"] == pytest.approx(0.5)         # CBM(bottom) − CBM(top) > 0: electrons go to the top layer


def test_confidence_is_high_far_from_a_boundary_and_low_on_it():
    assert type_probabilities(3.0, 1.5, -2.5, 0.2, 0.2, 0.2)["II"] > 0.95   # both offsets ≥ 1 eV
    p = type_probabilities(2.0, 1.0, 0.0, 0.2, 0.2, 0.3)      # VBMs equal: type I/II boundary
    assert 0.2 < p["I"] < 0.8


def test_metal_layer_has_no_junction_type():
    j = classify_junction(layer(-5.0, 1.5), layer(-5.1, 0.0), use_measured_edges=False)
    assert j["type"] is None and "metal" in j["reason"]


# --------------------------------------------------------------------------- on the built data

built = pytest.mark.skipif(not (BAND_GAPS.exists() and MODEL.exists()), reason="run build_data.py and train.py")


@built
def test_aliases_organics_and_metals():
    from materialstack.predict import resolve_layer
    assert resolve_layer("MAPbI3").display_formula == "CH3NH3PbI3"
    spiro = resolve_layer("Spiro-OMeTAD")
    assert spiro.edge_kind == "measured" and spiro.vbm_ev < -4.5 and spiro.gap_ev > 2.5
    assert resolve_layer("C60").formula == "organic:C60"           # not carbon
    assert resolve_layer("Au").gap_ev == 0.0


@built
def test_measured_interface_offset_in_both_orders():
    from materialstack.predict import predict_stack
    a = predict_stack(["CdS", "CuInSe2"])["junctions"][0]
    b = predict_stack(["CuInSe2", "CdS"])["junctions"][0]
    assert a["offset_source"].startswith("measured")
    assert a["vbo_ev"] == pytest.approx(-0.8) and b["vbo_ev"] == pytest.approx(0.8)   # CuInSe2 VBM 0.8 eV above CdS
    assert a["cbo_ev"] == pytest.approx(0.0, abs=1e-9) and b["cbo_ev"] == pytest.approx(0.0, abs=1e-9)


@built
def test_lowercase_formulas_and_offset_uncertainties():
    from materialstack.predict import predict_stack, resolve_layer
    assert resolve_layer("tio2").formula == "TiO2"
    assert resolve_layer("sio2").formula == "SiO2"                 # not the DFT-only SIO2 (sulfur iodine oxide)
    assert resolve_layer("Xyz").formula is None
    j = predict_stack(["TiO2", "MAPbI3"])["junctions"][0]
    assert j["cbo_sigma_ev"] > j["vbo_sigma_ev"] > 0               # ΔEc also carries both gap errors


def test_cv_fold_is_fixed_by_element_system_alone():
    from materialstack.model import element_group
    from validate import fold_of
    assert fold_of(element_group("CsPbI3")) == fold_of(element_group("Cs4PbI6"))   # same elements, same fold
    assert fold_of("As-Ga") == fold_of("As-Ga")                                    # no randomness, no other data
    assert {fold_of(f"X{i}") for i in range(200)} == {0, 1, 2, 3, 4}
