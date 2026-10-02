"""Build the materials database from all configured sources."""
from __future__ import annotations

import logging
import time
import traceback
from pathlib import Path

import pandas as pd

from materialstack.config import ALL_SOURCES, DATA_DIR, DB_PATH, DEFAULT_SOURCES, HEAVY_SOURCES
from materialstack.db import Writer, init_db, stats
from materialstack.sources import LOADERS

log = logging.getLogger("materialstack.build_db")


def _ingest(writer: Writer, source: str, ctx: dict, commit_every: int = 5000) -> tuple[int, int, int]:
    n_rec = n_struct = 0
    mats: set[int] = set()
    structure_cache: dict[tuple[str, str | None], int | None] = {}
    loader = LOADERS[source]
    for kind, row in loader(ctx):
        if kind == "alias":
            writer.alias(row["alias"], row.get("formula"), row.get("material_class", "unknown"), row.get("role_hint"),
                         row.get("note"))
            continue
        if kind == "interface":
            writer.interface(**row)
            n_rec += 1
            continue
        if row.get("unparseable"):
            continue
        formula = row.get("formula")
        if not formula:
            continue
        mid = writer.material_id(formula, phase_tag=row.get("phase_tag", "") or "", name=row.get("name"),
                                 material_class=row.get("material_class"))
        if mid is None:
            continue
        mats.add(mid)
        ext = row.get("external_id")
        sid = None
        key = (source, ext)
        if row.get("structure") is not None:
            sid = writer.structure(mid, row["structure"], source=source, external_id=ext, spg=row.get("spg"),
                                   e_above_hull=row.get("e_above_hull"))
            structure_cache[key] = sid
            if sid is not None:
                n_struct += 1
        elif key in structure_cache:
            sid = structure_cache[key]
        rid = writer.record(mid, source=source, method=row["method"], gap=row.get("gap"), vbm=row.get("vbm"),
                            cbm=row.get("cbm"), edge_reference=row.get("edge_reference", "none"),
                            is_direct=row.get("is_direct"), is_metal=row.get("is_metal"), structure_id=sid,
                            ip=row.get("ip"), ea=row.get("ea"), work_function=row.get("work_function"),
                            miller=row.get("miller"), external_id=ext, reference=row.get("reference"),
                            extra=row.get("extra"))
        if rid > 0:
            n_rec += 1
        if n_rec % commit_every == 0:
            writer.con.commit()
            log.info("[%s] ... %d records, %d materials, %d structures", source, n_rec, len(mats), n_struct)
    writer.con.commit()
    return n_rec, len(mats), n_struct


def build(sources: list[str] | None = None, *, skip: list[str] | None = None, heavy: bool = False,
          rebuild: bool = False, mp_api_key: str | None = None, no_structures: bool = False,
          db_path: Path = DB_PATH) -> pd.DataFrame:
    if rebuild and Path(db_path).exists():
        Path(db_path).unlink()
        for suffix in ("-wal", "-shm"):
            p = Path(str(db_path) + suffix)
            if p.exists():
                p.unlink()
    con = init_db(db_path)
    writer = Writer(con)

    force = sources is not None  # --only always re-ingests the named sources
    if sources is None:
        sources = list(DEFAULT_SOURCES)
        if heavy:
            sources += sorted(HEAVY_SOURCES)
        if mp_api_key or (DATA_DIR / "cache" / "mp_summary.parquet").exists():
            sources.append("materials_project")
    skip = set(skip or [])
    sources = [s for s in sources if s not in skip]
    unknown = [s for s in sources if s not in LOADERS]
    if unknown:
        raise ValueError(f"unknown sources: {unknown}; known: {ALL_SOURCES}")

    ctx = {"mp_api_key": mp_api_key, "no_structures": no_structures}
    # Surfaces/interfaces reference JARVIS ids resolved by dft_3d, so keep dft_3d before them.
    order = sorted(sources, key=lambda s: (s not in ("jarvis_dft_3d", "jarvis_dft_2d"), sources.index(s)))

    done = {r[0] for r in con.execute("SELECT source FROM build_log WHERE status='ok'")}
    for src in order:
        if src in done and not rebuild and not force:
            log.info("[%s] already built, skipping (use --rebuild or --only to redo)", src)
            if src in ("jarvis_dft_3d",) and any(s in sources for s in ("jarvis_surfacedb", "jarvis_interfacedb")):
                _prime_jid_map(con, ctx)
            continue
        t0 = time.time()
        log.info("[%s] loading ...", src)
        try:
            writer.delete_source(src)
            n_rec, n_mat, n_struct = _ingest(writer, src, ctx)
            writer.log(src, "ok", n_rec, n_mat, n_struct, time.time() - t0)
            con.commit()
            log.info("[%s] ok: %d records, %d materials, %d structures in %.0fs", src, n_rec, n_mat, n_struct,
                     time.time() - t0)
        except Exception as e:  # keep going with the other sources
            con.rollback()
            msg = f"{type(e).__name__}: {e}"
            writer.log(src, "skipped" if "skipped" in str(e) else "error", 0, 0, 0, time.time() - t0, msg[:500])
            con.commit()
            (log.warning if "skipped" in str(e) else log.error)("[%s] %s", src, msg)
            if "skipped" not in str(e):
                log.debug(traceback.format_exc())

    con.execute("ANALYZE")
    con.commit()
    summary = stats(con)
    summary["by_source"].to_csv(DATA_DIR / "materials_db_summary.csv", index=False)
    con.close()
    return summary["build_log"]


def _prime_jid_map(con, ctx: dict) -> None:
    """Rebuild the JVASP-id -> gaps map from an already-ingested jarvis_dft_3d so surface loaders can use it."""
    jid_gap: dict[str, dict] = ctx.setdefault("jarvis_jid_gap", {})
    rows = con.execute("""SELECT r.external_id, r.method, r.gap_ev, m.formula_reduced FROM records r
                          JOIN materials m USING(material_id) WHERE r.source='jarvis_dft_3d'""")
    for jid, method, gap, formula in rows:
        d = jid_gap.setdefault(jid, {"formula": formula})
        d[method] = gap
