"""Loaders for datasets distributed through jarvis-tools (NIST JARVIS figshare)."""
from __future__ import annotations

import re
from typing import Iterator

from materialstack.chem import jarvis_atoms_to_structure, to_float
from materialstack.config import CACHE_DIR

REF_JARVIS = "Choudhary et al., npj Comput. Mater. 6, 173 (2020); JARVIS-DFT"
REF_SNUMAT = "Kim et al., Sci. Data 7, 387 (2020); SNUMAT HSE06"
REF_HALIDE = "JARVIS halide perovskites (doi:10.1039/D1EE02971A)"
REF_FOUNDRY = "Foundry-ML experimental band gaps (doi:10.18126/wg3u-g8vu)"
REF_INTERMAT = "Choudhary & Garrity, Digital Discovery 3, 1365 (2024); InterMat / JARVIS surface & interface DB"

_NA = {"na", "n/a", "none", "", None}


def _f(x):
    if x is None or isinstance(x, (dict, list, tuple, set)):
        return None
    if isinstance(x, str) and x.strip().lower() in _NA:
        return None
    return to_float(x)


def _data(name: str) -> list[dict]:
    from jarvis.db.figshare import data
    return data(name, store_dir=str(CACHE_DIR))


def _formula_from_atoms(atoms: dict) -> str | None:
    try:
        from collections import Counter
        c = Counter(atoms["elements"])
        return "".join(f"{el}{n if n > 1 else ''}" for el, n in sorted(c.items()))
    except Exception:
        return None


def _bulk(name: str, source_tag: str, ctx: dict) -> Iterator[tuple[str, dict]]:
    keep_structures = not ctx.get("no_structures", False)
    jid_gap: dict[str, dict] = ctx.setdefault("jarvis_jid_gap", {})
    for d in _data(name):
        jid = d.get("jid")
        formula = d.get("formula") or _formula_from_atoms(d.get("atoms", {}))
        if not formula:
            continue
        opt, mbj, hse = _f(d.get("optb88vdw_bandgap")), _f(d.get("mbj_bandgap")), _f(d.get("hse_gap"))
        jid_gap[jid] = {"OptB88vdW": opt, "TBmBJ": mbj, "HSE06": hse, "formula": formula}
        struct = jarvis_atoms_to_structure(d["atoms"]) if keep_structures and d.get("atoms") else None
        spg = d.get("spg_number")
        spg = int(spg) if isinstance(spg, (int, float)) and spg == spg else None
        ehull = _f(d.get("ehull"))
        common = {"formula": formula, "structure": struct, "spg": spg, "e_above_hull": ehull, "external_id": jid,
                  "reference": REF_JARVIS, "extra": {"dimensionality": d.get("dimensionality"), "source_db": name}}
        first = True
        for method, gap in (("OptB88vdW", opt), ("TBmBJ", mbj), ("HSE06", hse)):
            if gap is None:
                continue
            row = dict(common, method=method, gap=gap)
            if not first:
                row["structure"] = None  # structure is inserted once; later records re-use it by external_id
            yield "record", row
            first = False


def dft_3d(ctx: dict) -> Iterator[tuple[str, dict]]:
    yield from _bulk("dft_3d", "jarvis_dft_3d", ctx)


def dft_2d(ctx: dict) -> Iterator[tuple[str, dict]]:
    for kind, row in _bulk("dft_2d", "jarvis_dft_2d", ctx):
        row["phase_tag"] = "2D"
        row["material_class"] = "inorganic_2d"
        yield kind, row


def snumat(ctx: dict) -> Iterator[tuple[str, dict]]:
    keep_structures = not ctx.get("no_structures", False)
    for d in _data("snumat"):
        formula = _formula_from_atoms(d.get("atoms", {}))
        if not formula:
            continue
        struct = jarvis_atoms_to_structure(d["atoms"]) if keep_structures else None
        spg = d.get("Space_group_rlx")
        spg = int(spg) if spg not in _NA else None
        sid = d.get("SNUMAT_id")
        extra = {"icsd": d.get("ICSD_number"), "soc": d.get("SOC"), "magnetic_ordering": d.get("Magnetic_ordering"),
                 "hse_optical_gap": _f(d.get("Band_gap_HSE_optical"))}
        hse = _f(d.get("Band_gap_HSE"))
        gga = _f(d.get("Band_gap_GGA"))
        if hse is not None:
            yield "record", {"formula": formula, "method": "HSE06", "gap": hse, "structure": struct, "spg": spg,
                             "is_direct": (d.get("Direct_or_indirect_HSE") == "Direct"), "external_id": sid,
                             "reference": REF_SNUMAT, "extra": extra}
        if gga is not None:
            yield "record", {"formula": formula, "method": "PBE", "gap": gga, "structure": None, "spg": spg,
                             "is_direct": (d.get("Direct_or_indirect") == "Direct"), "external_id": sid,
                             "reference": REF_SNUMAT, "extra": extra}


def halide_perovskites(ctx: dict) -> Iterator[tuple[str, dict]]:
    for d in _data("halide_peroskites"):
        formula = _formula_from_atoms(d.get("atoms", {}))
        if not formula:
            continue
        struct = jarvis_atoms_to_structure(d["atoms"]) if not ctx.get("no_structures") else None
        extra = {"tolerance_factor_reported": _f(d.get("Tol")), "hse_decomp_energy": _f(d.get("HSE_decomp_energy"))}
        for method, key in (("HSE06", "HSE_gap"), ("PBE", "PBE_gap")):
            gap = _f(d.get(key))
            if gap is None:
                continue
            yield "record", {"formula": formula, "method": method, "gap": gap,
                             "structure": struct if method == "HSE06" else None, "external_id": d.get("id"),
                             "reference": REF_HALIDE, "extra": extra}


def foundry_exp_gaps(ctx: dict) -> Iterator[tuple[str, dict]]:
    for d in _data("foundry_ml_exp_bandgaps"):
        yield "record", {"formula": d["composition"], "method": "experiment", "gap": _f(d.get("exp_gap")),
                         "external_id": d.get("id"), "reference": REF_FOUNDRY}


_SURF = re.compile(r"Surface-(JVASP-\d+)_miller_(\d+)_(\d+)_(\d+)")
_INTF = re.compile(r"Interface-(JVASP-\d+)_(JVASP-\d+)_film_miller_(\d+)_(\d+)_(\d+)_sub_miller_(\d+)_(\d+)_(\d+)")


def surfacedb(ctx: dict) -> Iterator[tuple[str, dict]]:
    """Vacuum-referenced VBM/CBM from slab calculations (OptB88vdW). Adds a TBmBJ-corrected CBM when the
    bulk TBmBJ gap is known (requires jarvis_dft_3d to have been loaded earlier in the same build)."""
    jid_gap: dict[str, dict] = ctx.get("jarvis_jid_gap", {})
    for d in _data("surfacedb"):
        m = _SURF.search(d.get("name", ""))
        jid = m.group(1) if m else None
        miller = "".join(m.groups()[1:]) if m else None
        vbm, cbm = _f(d.get("surf_vbm")), _f(d.get("surf_cbm"))
        wf = _f(d.get("efermi"))
        formula = d.get("formula")
        if not formula or vbm is None:
            continue
        gap_opt = (cbm - vbm) if (cbm is not None) else None
        base = {"formula": formula, "edge_reference": "vacuum", "miller": miller, "external_id": d.get("name"),
                "reference": REF_INTERMAT, "ip": -vbm, "work_function": -wf if wf is not None else None,
                "extra": {"jid": jid, "surface_energy": _f(d.get("surf_en"))}}
        yield "record", dict(base, method="DFT-surface-OptB88vdW", gap=gap_opt, vbm=vbm, cbm=cbm,
                             ea=-cbm if cbm is not None else None)
        mbj = (jid_gap.get(jid) or {}).get("TBmBJ") if jid else None
        if mbj is not None and mbj > 0:
            cbm_corr = vbm + mbj
            yield "record", dict(base, method="DFT-surface-TBmBJ", gap=mbj, vbm=vbm, cbm=cbm_corr, ea=-cbm_corr,
                                 extra=dict(base["extra"], note="CBM = surface VBM + bulk TBmBJ gap"))


def interfacedb(ctx: dict) -> Iterator[tuple[str, dict]]:
    jid_gap: dict[str, dict] = ctx.get("jarvis_jid_gap", {})
    for d in _data("interfacedb"):
        m = _INTF.search(d.get("jid", ""))
        if not m:
            continue
        ja, jb = m.group(1), m.group(2)
        vbo = _f(d.get("offset"))
        yield "interface", {"source": "jarvis_interfacedb", "external_id": d["jid"], "material_a": ja, "material_b": jb,
                            "formula_a": (jid_gap.get(ja) or {}).get("formula"),
                            "formula_b": (jid_gap.get(jb) or {}).get("formula"),
                            "miller_a": "".join(m.groups()[2:5]), "miller_b": "".join(m.groups()[5:8]),
                            "vbo_ev": vbo, "cbo_ev": None, "junction_type": None, "method": "DFT-ASJ-OptB88vdW",
                            "extra": {"interface_gap_optb88vdw": _f(d.get("optb88vdw_bandgap")),
                                      "interface_vbm": _f(d.get("optb88vdw_vbm")),
                                      "interface_cbm": _f(d.get("optb88vdw_cbm")),
                                      "gap_a_tbmbj": (jid_gap.get(ja) or {}).get("TBmBJ"),
                                      "gap_b_tbmbj": (jid_gap.get(jb) or {}).get("TBmBJ")}}
