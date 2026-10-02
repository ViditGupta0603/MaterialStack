"""User-supplied literature CSVs and the alias table for common solar-cell layer names."""
from __future__ import annotations

from typing import Iterator

import pandas as pd

from materialstack.chem import clean_material_name, to_float
from materialstack.config import LITERATURE_DIR

# Accepted column names (case-insensitive) -> canonical.
COLMAP = {
    "material": "material", "name": "material", "material_name": "material", "raw_name": "material",
    "formula": "formula", "formula_clean": "formula", "composition": "formula",
    "eg": "gap", "gap": "gap", "band_gap": "gap", "bandgap": "gap", "eg_ev": "gap",
    "vbm": "vbm", "evbm": "vbm", "vb": "vbm", "homo": "vbm", "ip": "ip", "ionization_potential": "ip",
    "cbm": "cbm", "ecbm": "cbm", "cb": "cbm", "lumo": "cbm", "ea": "ea", "electron_affinity": "ea", "chi": "ea",
    "method": "method", "technique": "method", "reference": "reference", "ref": "reference", "doi": "reference",
    "phase": "phase_tag", "phase_tag": "phase_tag", "role": "role", "material_class": "material_class",
    "edge_reference": "edge_reference", "scale": "edge_reference",
}

TEMPLATE_HEADER = ("material,formula,phase_tag,material_class,role,gap,vbm,cbm,ip,ea,method,edge_reference,"
                   "reference\n")


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    ren = {c: COLMAP.get(c.strip().lower().replace(" ", "_"), None) for c in df.columns}
    ren = {k: v for k, v in ren.items() if v}
    return df.rename(columns=ren)


def literature_csv(ctx: dict) -> Iterator[tuple[str, dict]]:
    files = sorted(LITERATURE_DIR.glob("*.csv"))
    template = LITERATURE_DIR / "TEMPLATE_literature.csv"
    if not template.exists():
        template.write_text(TEMPLATE_HEADER, encoding="utf-8")
    files = [f for f in files if f.name != template.name]
    if not files:
        raise RuntimeError(f"literature_csv: no CSV files in {LITERATURE_DIR} (template written); skipped")
    for f in files:
        df = _norm_cols(pd.read_csv(f))
        for d in df.to_dict(orient="records"):
            name = str(d.get("material") or d.get("formula") or "").strip()
            formula = d.get("formula")
            cn = clean_material_name(name) if name else None
            if not isinstance(formula, str) or not formula.strip():
                formula = cn.formula_clean if cn else None
            vbm, cbm = to_float(d.get("vbm")), to_float(d.get("cbm"))
            ip, ea = to_float(d.get("ip")), to_float(d.get("ea"))
            gap = to_float(d.get("gap"))
            if vbm is None and ip is not None:
                vbm = -abs(ip)
            if cbm is None and ea is not None:
                cbm = -abs(ea)
            if gap is None and vbm is not None and cbm is not None:
                gap = cbm - vbm
            if cbm is None and vbm is not None and gap is not None:
                cbm = vbm + gap
            cls = d.get("material_class") if isinstance(d.get("material_class"), str) else (cn.material_class if cn else None)
            phase = d.get("phase_tag") if isinstance(d.get("phase_tag"), str) else (cn.phase_tag if cn else "")
            row = {"formula": formula, "name": name or formula, "phase_tag": phase or "", "material_class": cls,
                   "method": "literature", "gap": gap, "vbm": vbm, "cbm": cbm,
                   "edge_reference": (d.get("edge_reference") if isinstance(d.get("edge_reference"), str) else "vacuum"),
                   "ip": ip if ip is not None else (-vbm if vbm is not None else None),
                   "ea": ea if ea is not None else (-cbm if cbm is not None else None),
                   "external_id": f"{f.stem}:{name}", "reference": d.get("reference"),
                   "extra": {"file": f.name, "measurement": d.get("method"), "role": d.get("role")}}
            if not formula:
                # Organic / composite: keep as alias so the run step can at least identify it.
                yield "alias", {"alias": name, "formula": None, "material_class": cls or "organic",
                                "role_hint": d.get("role"), "note": f"from {f.name}; no stoichiometric formula"}
                row["unparseable"] = True
            yield "record", row


# Common layer names in perovskite / thin-film solar cells. Values here are identities only (no band data).
ALIASES: list[tuple[str, str | None, str, str | None, str | None]] = [
    ("MAPbI3", "CH3NH3PbI3", "inorganic_bulk", "absorber", "methylammonium lead iodide"),
    ("MAPbBr3", "CH3NH3PbBr3", "inorganic_bulk", "absorber", None),
    ("MAPbCl3", "CH3NH3PbCl3", "inorganic_bulk", "absorber", None),
    ("FAPbI3", "CH5N2PbI3", "inorganic_bulk", "absorber", "formamidinium lead iodide"),
    ("FAPbBr3", "CH5N2PbBr3", "inorganic_bulk", "absorber", None),
    ("MASnI3", "CH3NH3SnI3", "inorganic_bulk", "absorber", None),
    ("FASnI3", "CH5N2SnI3", "inorganic_bulk", "absorber", None),
    ("CsPbI3", "CsPbI3", "inorganic_bulk", "absorber", None),
    ("CsPbBr3", "CsPbBr3", "inorganic_bulk", "absorber", None),
    ("CsSnI3", "CsSnI3", "inorganic_bulk", "absorber", None),
    ("Cs2AgBiBr6", "Cs2AgBiBr6", "inorganic_bulk", "absorber", "lead-free double perovskite"),
    ("CIGS", "CuIn0.7Ga0.3Se2", "inorganic_bulk", "absorber", "typical Ga fraction 0.3"),
    ("CIS", "CuInSe2", "inorganic_bulk", "absorber", None),
    ("CZTS", "Cu2ZnSnS4", "inorganic_bulk", "absorber", None),
    ("CZTSe", "Cu2ZnSnSe4", "inorganic_bulk", "absorber", None),
    ("c-TiO2", "TiO2", "inorganic_bulk", "ETL", "compact TiO2 (anatase)"),
    ("TiO2-c", "TiO2", "inorganic_bulk", "ETL", "compact TiO2 (anatase)"),
    ("mp-TiO2", "TiO2", "inorganic_bulk", "ETL", "mesoporous TiO2 (anatase)"),
    ("anatase", "TiO2", "inorganic_bulk", "ETL", None),
    ("rutile", "TiO2", "inorganic_bulk", "ETL", None),
    ("ZnO", "ZnO", "inorganic_bulk", "ETL", None),
    ("SnO2", "SnO2", "inorganic_bulk", "ETL", None),
    ("CdS", "CdS", "inorganic_bulk", "ETL", "buffer layer in CIGS/CdTe cells"),
    ("ZnS", "ZnS", "inorganic_bulk", "ETL", None),
    ("Zn(O,S)", "ZnO0.5S0.5", "inorganic_bulk", "ETL", "approximate composition"),
    ("NiO", "NiO", "inorganic_bulk", "HTL", None),
    ("NiOx", "NiO", "inorganic_bulk", "HTL", "non-stoichiometric NiO"),
    ("CuI", "CuI", "inorganic_bulk", "HTL", None),
    ("CuSCN", "CuSCN", "inorganic_bulk", "HTL", None),
    ("Cu2O", "Cu2O", "inorganic_bulk", "HTL", None),
    ("MoO3", "MoO3", "inorganic_bulk", "HTL", None),
    ("MoOx", "MoO3", "inorganic_bulk", "HTL", None),
    ("WO3", "WO3", "inorganic_bulk", "ETL", None),
    ("V2O5", "V2O5", "inorganic_bulk", "HTL", None),
    ("Nb2O5", "Nb2O5", "inorganic_bulk", "ETL", None),
    ("ITO", "In1.8Sn0.2O3", "doped_composite", "contact", "In2O3:Sn, ~10% Sn"),
    ("FTO", "SnO2", "doped_composite", "contact", "SnO2:F"),
    ("AZO", "ZnO", "doped_composite", "contact", "ZnO:Al"),
    ("MoS2", "MoS2", "inorganic_2d", "HTL", None),
    ("WS2", "WS2", "inorganic_2d", "HTL", None),
    ("MoSe2", "MoSe2", "inorganic_2d", None, None),
    ("WSe2", "WSe2", "inorganic_2d", None, None),
    ("Spiro-OMeTAD", None, "organic", "HTL", "C81H68N4O8; values only from literature CSV"),
    ("spiro", None, "organic", "HTL", "Spiro-OMeTAD"),
    ("PTAA", None, "organic", "HTL", "poly(triarylamine)"),
    ("PEDOT:PSS", None, "organic", "HTL", None),
    ("P3HT", None, "organic", "HTL", None),
    ("PCBM", None, "organic", "ETL", "PC61BM"),
    ("PC61BM", None, "organic", "ETL", None),
    ("PC60BM", None, "organic", "ETL", None),
    ("PC71BM", None, "organic", "ETL", None),
    ("C60", "C60", "organic", "ETL", "fullerene"),
    ("BCP", None, "organic", "buffer", "bathocuproine"),
    ("Au", "Au", "inorganic_bulk", "contact", None),
    ("Ag", "Ag", "inorganic_bulk", "contact", None),
    ("Al", "Al", "inorganic_bulk", "contact", None),
    ("Cu", "Cu", "inorganic_bulk", "contact", None),
    ("Mo", "Mo", "inorganic_bulk", "contact", None),
    ("Si", "Si", "inorganic_bulk", "absorber", None),
    ("a-Si", "Si", "inorganic_bulk", "absorber", "amorphous silicon"),
    ("CdTe", "CdTe", "inorganic_bulk", "absorber", None),
    ("GaAs", "GaAs", "inorganic_bulk", "absorber", None),
    ("Sb2Se3", "Sb2Se3", "inorganic_bulk", "absorber", None),
    ("Sb2S3", "Sb2S3", "inorganic_bulk", "absorber", None),
]


def aliases(ctx: dict) -> Iterator[tuple[str, dict]]:
    for alias, formula, cls, role, note in ALIASES:
        yield "alias", {"alias": alias, "formula": formula, "material_class": cls, "role_hint": role, "note": note}
