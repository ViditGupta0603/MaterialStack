"""Loaders for datasets hosted by matminer (figshare / matbench)."""
from __future__ import annotations

from typing import Iterator

import pandas as pd

from materialstack.chem import to_float

REF_ZHUO = "Zhuo, Mansouri Tehrani, Brgoch, J. Phys. Chem. Lett. 9, 1668 (2018)"
REF_CASTELLI = "Castelli et al., Energy Environ. Sci. 5, 5814 (2012); CMR cubic perovskites"
REF_PILANIA = "Pilania et al., Sci. Rep. 6, 19375 (2016)"
REF_MP2018 = "Materials Project snapshot 2018-10-18 (Jain et al., APL Mater. 1, 011002 (2013))"
REF_WOLVERTON = "Emery & Wolverton, Sci. Data 4, 170153 (2017)"
REF_PETOUSIS = "Petousis et al., Sci. Data 4, 160134 (2017)"


def _rows(name: str) -> Iterator[dict]:
    from matminer.datasets import load_dataset
    df: pd.DataFrame = load_dataset(name)
    cols = list(df.columns)
    for tup in df.itertuples(index=False, name=None):
        yield dict(zip(cols, tup))


def _str(x) -> str | None:
    return x if isinstance(x, str) and x else None


def expt_gap(ctx: dict) -> Iterator[tuple[str, dict]]:
    for d in _rows("expt_gap"):
        yield "record", {"formula": d["formula"], "method": "experiment", "gap": to_float(d["gap expt"]),
                         "reference": REF_ZHUO}


def expt_gap_kingsbury(ctx: dict) -> Iterator[tuple[str, dict]]:
    for d in _rows("expt_gap_kingsbury"):
        yield "record", {"formula": d["formula"], "method": "experiment", "gap": to_float(d["expt_gap"]),
                         "external_id": _str(d.get("likely_mpid")),
                         "reference": REF_ZHUO + "; cleaned by Kingsbury (matbench_expt_gap)"}


def castelli_perovskites(ctx: dict) -> Iterator[tuple[str, dict]]:
    for i, d in enumerate(_rows("castelli_perovskites")):
        vbm, cbm = to_float(d["vbm"]), to_float(d["cbm"])
        # CMR stores the edges as positive energies below vacuum; convert to a signed vacuum scale.
        yield "record", {"formula": d["formula"], "method": "GLLB-SC", "gap": to_float(d["gap gllbsc"]),
                         "vbm": -abs(vbm) if vbm is not None else None,
                         "cbm": -abs(cbm) if cbm is not None else None,
                         "edge_reference": "vacuum", "is_direct": bool(d["gap is direct"]),
                         "structure": d.get("structure"), "external_id": f"castelli-{i}", "reference": REF_CASTELLI,
                         "extra": {"e_form": to_float(d.get("e_form")), "fermi_level": to_float(d.get("fermi level")),
                                   "edge_note": "cubic ABX3 prototype; alignment per CMR"}}


def double_perovskites_gap(ctx: dict) -> Iterator[tuple[str, dict]]:
    for i, d in enumerate(_rows("double_perovskites_gap")):
        yield "record", {"formula": d["formula"], "method": "GLLB-SC", "gap": to_float(d["gap gllbsc"]),
                         "external_id": f"dp-{i}", "reference": REF_PILANIA,
                         "extra": {k: d[k] for k in ("a_1", "b_1", "a_2", "b_2")}}


def mp_nostruct_20181018(ctx: dict) -> Iterator[tuple[str, dict]]:
    for d in _rows("mp_nostruct_20181018"):
        yield "record", {"formula": d["formula"], "method": "PBE", "gap": to_float(d["gap pbe"]),
                         "external_id": _str(d.get("mpid")), "e_above_hull": to_float(d.get("e_hull")),
                         "reference": REF_MP2018, "extra": {"e_form": to_float(d.get("e_form"))}}


def mp_all_20181018(ctx: dict) -> Iterator[tuple[str, dict]]:
    for d in _rows("mp_all_20181018"):
        yield "record", {"formula": d["formula"], "method": "PBE", "gap": to_float(d["gap pbe"]),
                         "external_id": _str(d.get("mpid")), "structure": d.get("structure"),
                         "e_above_hull": to_float(d.get("e_hull")), "reference": REF_MP2018}


def matbench_mp_gap(ctx: dict) -> Iterator[tuple[str, dict]]:
    for i, d in enumerate(_rows("matbench_mp_gap")):
        s = d["structure"]
        yield "record", {"formula": s.composition.reduced_formula, "method": "PBE", "gap": to_float(d["gap pbe"]),
                         "structure": s, "external_id": f"mbgap-{i}", "reference": "matbench_mp_gap (MP 2019)"}


def wolverton_oxides(ctx: dict) -> Iterator[tuple[str, dict]]:
    for i, d in enumerate(_rows("wolverton_oxides")):
        yield "record", {"formula": d["formula"], "method": "PBE", "gap": to_float(d["gap pbe"]),
                         "e_above_hull": to_float(d.get("e_hull")), "external_id": f"wolv-{i}",
                         "reference": REF_WOLVERTON, "extra": {"lowest_distortion": _str(d.get("lowest distortion"))}}


def dielectric_constant(ctx: dict) -> Iterator[tuple[str, dict]]:
    for d in _rows("dielectric_constant"):
        spg = d.get("space_group")
        yield "record", {"formula": d["formula"], "method": "PBE", "gap": to_float(d["band_gap"]),
                         "structure": d.get("structure"), "spg": int(spg) if spg is not None else None,
                         "external_id": _str(d.get("material_id")), "reference": REF_PETOUSIS,
                         "extra": {"refractive_index": to_float(d.get("n"))}}
