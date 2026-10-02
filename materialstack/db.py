"""SQLite access layer for the MaterialStack materials database."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from materialstack.chem import (classify_family, mulliken_chi, structure_summary, structure_to_cif,
                            try_composition)
from materialstack.config import DB_PATH, METHOD_RANK

SCHEMA = """
CREATE TABLE IF NOT EXISTS materials (
    material_id        INTEGER PRIMARY KEY,
    material_key       TEXT UNIQUE NOT NULL,      -- formula_reduced + '|' + phase_tag
    formula_reduced    TEXT NOT NULL,
    formula_raw        TEXT,
    material_name      TEXT,
    phase_tag          TEXT DEFAULT '',
    material_class     TEXT,
    anonymized_formula TEXT,
    n_elements         INTEGER,
    elements           TEXT,                      -- JSON list
    family             TEXT,
    is_perovskite_like INTEGER DEFAULT 0,
    tolerance_factor   REAL,
    octahedral_factor  REAL,
    mulliken_chi       REAL
);
CREATE INDEX IF NOT EXISTS idx_materials_formula ON materials(formula_reduced);

CREATE TABLE IF NOT EXISTS structures (
    structure_id       INTEGER PRIMARY KEY,
    material_id        INTEGER NOT NULL REFERENCES materials(material_id),
    source             TEXT NOT NULL,
    external_id        TEXT,
    cif                TEXT,
    space_group_number INTEGER,
    crystal_system     TEXT,
    density            REAL,
    volume_per_atom    REAL,
    n_sites            INTEGER,
    e_above_hull       REAL,
    UNIQUE(source, external_id)
);
CREATE INDEX IF NOT EXISTS idx_structures_material ON structures(material_id);

CREATE TABLE IF NOT EXISTS records (
    record_id          INTEGER PRIMARY KEY,
    material_id        INTEGER NOT NULL REFERENCES materials(material_id),
    structure_id       INTEGER REFERENCES structures(structure_id),
    source             TEXT NOT NULL,
    method             TEXT NOT NULL,
    method_rank        INTEGER NOT NULL,
    gap_ev             REAL,
    is_direct          INTEGER,
    is_metal           INTEGER,
    vbm_ev             REAL,
    cbm_ev             REAL,
    edge_reference     TEXT DEFAULT 'none',       -- vacuum | nhe | internal | none
    ip_ev              REAL,
    ea_ev              REAL,
    work_function_ev   REAL,
    surface_miller     TEXT,
    external_id        TEXT,
    reference          TEXT,
    extra              TEXT                       -- JSON
);
CREATE INDEX IF NOT EXISTS idx_records_material ON records(material_id);
CREATE INDEX IF NOT EXISTS idx_records_source ON records(source);

CREATE TABLE IF NOT EXISTS interfaces (
    interface_id       INTEGER PRIMARY KEY,
    source             TEXT NOT NULL,
    external_id        TEXT UNIQUE,
    material_a         TEXT,
    material_b         TEXT,
    formula_a          TEXT,
    formula_b          TEXT,
    miller_a           TEXT,
    miller_b           TEXT,
    vbo_ev             REAL,
    cbo_ev             REAL,
    junction_type      TEXT,
    method             TEXT,
    extra              TEXT
);

CREATE TABLE IF NOT EXISTS aliases (
    alias              TEXT PRIMARY KEY,
    formula            TEXT,
    material_class     TEXT,
    role_hint          TEXT,
    note               TEXT
);

CREATE TABLE IF NOT EXISTS build_log (
    source             TEXT PRIMARY KEY,
    status             TEXT,
    n_records          INTEGER,
    n_materials        INTEGER,
    n_structures       INTEGER,
    seconds            REAL,
    message            TEXT,
    built_at           TEXT
);

CREATE VIEW IF NOT EXISTS materials_best AS
WITH ranked AS (
    SELECT r.*, ROW_NUMBER() OVER (PARTITION BY material_id ORDER BY method_rank, record_id) AS rn_gap
    FROM records r WHERE gap_ev IS NOT NULL
), edges AS (
    SELECT r.*, ROW_NUMBER() OVER (PARTITION BY material_id ORDER BY method_rank, record_id) AS rn_edge
    FROM records r WHERE cbm_ev IS NOT NULL AND vbm_ev IS NOT NULL AND edge_reference IN ('vacuum','nhe')
), agg AS (
    SELECT material_id, COUNT(*) AS n_records, MIN(gap_ev) AS gap_min, MAX(gap_ev) AS gap_max,
           SUM(CASE WHEN structure_id IS NOT NULL THEN 1 ELSE 0 END) AS n_with_structure
    FROM records GROUP BY material_id
)
SELECT m.material_id, m.material_key, m.formula_reduced, m.material_name, m.phase_tag, m.material_class, m.family,
       g.gap_ev AS best_gap, g.method AS best_gap_method, g.source AS best_gap_source,
       e.cbm_ev AS best_cbm, e.vbm_ev AS best_vbm, e.method AS best_edge_method, e.source AS best_edge_source,
       a.n_records, a.gap_min, a.gap_max, a.n_with_structure
FROM materials m
LEFT JOIN ranked g ON g.material_id = m.material_id AND g.rn_gap = 1
LEFT JOIN edges  e ON e.material_id = m.material_id AND e.rn_edge = 1
LEFT JOIN agg    a ON a.material_id = m.material_id;
"""


def connect(path: Path | str = DB_PATH) -> sqlite3.Connection:
    con = sqlite3.connect(str(path))
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA foreign_keys=ON")
    return con


def init_db(path: Path | str = DB_PATH) -> sqlite3.Connection:
    con = connect(path)
    con.executescript(SCHEMA)
    return con


@contextmanager
def transaction(con: sqlite3.Connection):
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise


class Writer:
    """Insert helper with an in-memory material_key -> material_id cache."""

    def __init__(self, con: sqlite3.Connection):
        self.con = con
        self._mat_cache: dict[str, int] = {
            k: i for k, i in con.execute("SELECT material_key, material_id FROM materials")
        }

    # ---- materials
    def material_id(self, formula: str, *, phase_tag: str = "", name: str | None = None,
                    material_class: str | None = None) -> int | None:
        comp = try_composition(formula)
        if comp is None:
            return None
        red = comp.reduced_formula
        key = f"{red}|{phase_tag or ''}"
        if key in self._mat_cache:
            return self._mat_cache[key]
        fam = classify_family(comp)
        chi = mulliken_chi(comp)
        cls = material_class or ("organic" if fam["family"] == "organic" else "inorganic_bulk")
        cur = self.con.execute(
            """INSERT INTO materials(material_key, formula_reduced, formula_raw, material_name, phase_tag,
               material_class, anonymized_formula, n_elements, elements, family, is_perovskite_like,
               tolerance_factor, octahedral_factor, mulliken_chi)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (key, red, formula, name or red, phase_tag or "", cls, fam["anonymized_formula"], fam["n_elements"],
             json.dumps(sorted(e.symbol for e in comp)), fam["family"], fam["is_perovskite_like"],
             fam["tolerance_factor"], fam["octahedral_factor"], chi),
        )
        self._mat_cache[key] = cur.lastrowid
        return cur.lastrowid

    # ---- structures
    def structure(self, material_id: int, structure, *, source: str, external_id: str | None,
                  spg: int | None = None, e_above_hull: float | None = None, store_cif: bool = True) -> int | None:
        if structure is None:
            return None
        row = self.con.execute("SELECT structure_id FROM structures WHERE source=? AND external_id=?",
                               (source, external_id)).fetchone()
        if row:
            return row[0]
        summ = structure_summary(structure, spg)
        cif = structure_to_cif(structure) if store_cif else None
        cur = self.con.execute(
            """INSERT INTO structures(material_id, source, external_id, cif, space_group_number, crystal_system,
               density, volume_per_atom, n_sites, e_above_hull) VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (material_id, source, external_id, cif, summ["space_group_number"], summ["crystal_system"],
             summ["density"], summ["volume_per_atom"], summ["n_sites"], e_above_hull),
        )
        return cur.lastrowid

    # ---- records
    def record(self, material_id: int, *, source: str, method: str, gap: float | None = None,
               vbm: float | None = None, cbm: float | None = None, edge_reference: str = "none",
               is_direct: bool | None = None, is_metal: bool | None = None, structure_id: int | None = None,
               ip: float | None = None, ea: float | None = None, work_function: float | None = None,
               miller: str | None = None, external_id: str | None = None, reference: str | None = None,
               extra: dict | None = None) -> int:
        if gap is None and vbm is None and cbm is None and work_function is None:
            return -1
        if is_metal is None and gap is not None:
            is_metal = gap <= 1e-6
        cur = self.con.execute(
            """INSERT INTO records(material_id, structure_id, source, method, method_rank, gap_ev, is_direct,
               is_metal, vbm_ev, cbm_ev, edge_reference, ip_ev, ea_ev, work_function_ev, surface_miller,
               external_id, reference, extra) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (material_id, structure_id, source, method, METHOD_RANK.get(method, 8), gap,
             None if is_direct is None else int(is_direct), None if is_metal is None else int(is_metal),
             vbm, cbm, edge_reference, ip, ea, work_function, miller, external_id, reference,
             json.dumps(extra, default=str) if extra else None),
        )
        return cur.lastrowid

    def interface(self, **kw) -> None:
        cols = ["source", "external_id", "material_a", "material_b", "formula_a", "formula_b", "miller_a",
                "miller_b", "vbo_ev", "cbo_ev", "junction_type", "method", "extra"]
        vals = [kw.get(c) for c in cols]
        if isinstance(vals[-1], dict):
            vals[-1] = json.dumps(vals[-1], default=str)
        self.con.execute(
            f"INSERT OR REPLACE INTO interfaces({','.join(cols)}) VALUES ({','.join('?' * len(cols))})", vals)

    def alias(self, alias: str, formula: str | None, material_class: str, role_hint: str | None = None,
              note: str | None = None) -> None:
        self.con.execute("INSERT OR REPLACE INTO aliases(alias, formula, material_class, role_hint, note) "
                         "VALUES (?,?,?,?,?)", (alias, formula, material_class, role_hint, note))

    def log(self, source: str, status: str, n_records: int, n_materials: int, n_structures: int,
            seconds: float, message: str = "") -> None:
        self.con.execute(
            "INSERT OR REPLACE INTO build_log VALUES (?,?,?,?,?,?,?,datetime('now'))",
            (source, status, n_records, n_materials, n_structures, round(seconds, 1), message))

    def delete_source(self, source: str) -> None:
        self.con.execute("DELETE FROM records WHERE source=?", (source,))
        self.con.execute("DELETE FROM structures WHERE source=?", (source,))
        self.con.execute("DELETE FROM interfaces WHERE source=?", (source,))
        self.con.execute("DELETE FROM build_log WHERE source=?", (source,))


# --------------------------------------------------------------------------- reads

def resolve_alias(con: sqlite3.Connection, name: str) -> dict | None:
    row = con.execute("SELECT alias, formula, material_class, role_hint, note FROM aliases WHERE lower(alias)=lower(?)",
                      (name,)).fetchone()
    if not row:
        return None
    return dict(zip(["alias", "formula", "material_class", "role_hint", "note"], row))


def lookup(con: sqlite3.Connection, formula: str) -> dict[str, Any]:
    """All records for a formula (any phase) plus the best-ranked summary rows."""
    comp = try_composition(formula)
    red = comp.reduced_formula if comp else formula
    mats = pd.read_sql_query("SELECT * FROM materials WHERE formula_reduced=?", con, params=(red,))
    if mats.empty:
        return {"formula": red, "materials": mats, "records": pd.DataFrame(), "best": pd.DataFrame(),
                "structures": pd.DataFrame()}
    ids = tuple(int(i) for i in mats.material_id)
    q = f"({','.join('?' * len(ids))})"
    recs = pd.read_sql_query(
        f"""SELECT r.record_id, m.material_key, r.source, r.method, r.method_rank, r.gap_ev, r.vbm_ev, r.cbm_ev,
                   r.edge_reference, r.ip_ev, r.ea_ev, r.work_function_ev, r.surface_miller, r.is_direct,
                   r.is_metal, r.structure_id, r.external_id, r.reference
            FROM records r JOIN materials m USING(material_id)
            WHERE r.material_id IN {q} ORDER BY r.method_rank, r.source""", con, params=ids)
    best = pd.read_sql_query(f"SELECT * FROM materials_best WHERE material_id IN {q}", con, params=ids)
    structs = pd.read_sql_query(
        f"""SELECT structure_id, material_id, source, external_id, space_group_number, crystal_system, density,
                   volume_per_atom, n_sites, e_above_hull FROM structures WHERE material_id IN {q}""",
        con, params=ids)
    return {"formula": red, "materials": mats, "records": recs, "best": best, "structures": structs}


def stats(con: sqlite3.Connection) -> dict[str, Any]:
    out = {}
    for t in ("materials", "records", "structures", "interfaces", "aliases"):
        out[t] = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    out["by_source"] = pd.read_sql_query(
        """SELECT source, method, COUNT(*) AS n_records, COUNT(DISTINCT material_id) AS n_materials,
                  SUM(CASE WHEN cbm_ev IS NOT NULL AND edge_reference IN ('vacuum','nhe') THEN 1 ELSE 0 END)
                  AS n_vacuum_edges
           FROM records GROUP BY source, method ORDER BY source, method""", con)
    out["by_family"] = pd.read_sql_query(
        "SELECT family, COUNT(*) AS n_materials FROM materials GROUP BY family ORDER BY n_materials DESC", con)
    out["build_log"] = pd.read_sql_query("SELECT * FROM build_log ORDER BY source", con)
    return out
