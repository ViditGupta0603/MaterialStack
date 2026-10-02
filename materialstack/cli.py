from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from materialstack.config import ALL_SOURCES, DB_PATH

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 200)


def _setup_logging(verbosity: int) -> None:
    level = logging.WARNING if verbosity == 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("matminer").setLevel(logging.WARNING)


def cmd_build_db(args: argparse.Namespace) -> None:
    from materialstack.build_db import build
    sources = args.only if args.only else None
    log = build(sources, skip=args.skip, heavy=args.heavy, rebuild=args.rebuild, mp_api_key=args.mp_api_key,
                no_structures=args.no_structures)
    print("\nBuild log:")
    print(log.to_string(index=False))


def cmd_stats(args: argparse.Namespace) -> None:
    from materialstack.db import connect, stats
    con = connect(DB_PATH)
    s = stats(con)
    print(f"materials={s['materials']}  records={s['records']}  structures={s['structures']}  "
          f"interfaces={s['interfaces']}  aliases={s['aliases']}")
    print("\nRecords by source / method:")
    print(s["by_source"].to_string(index=False))
    print("\nMaterials by family:")
    print(s["by_family"].to_string(index=False))


def cmd_featurize(args: argparse.Namespace) -> None:
    from materialstack.features import run
    print(run(n_jobs=args.n_jobs, structures=not args.no_structures))


def cmd_featurize_structures(args: argparse.Namespace) -> None:
    from materialstack.db import connect
    from materialstack.structure import featurize_structures
    con = connect(DB_PATH)
    n = featurize_structures(con, n_jobs=args.n_jobs, rebuild=args.rebuild)
    con.close()
    print(f"features_structure: {n} structures featurized")


def cmd_train_proxies(args: argparse.Namespace) -> None:
    from materialstack.structure import train_dft_proxies
    rep = train_dft_proxies(n_splits=args.n_splits)
    print("DFT proxy models (out-of-fold, GroupKFold by element set):")
    for key, r in rep["targets"].items():
        print(f"  {key:10s} {r['method']:10s} n={r['n_train']:6d}  MAE={r['oof_mae_all']:.3f} eV  "
              f"non-metal MAE={r['oof_mae_nonmetal']:.3f}  R2={r['oof_r2_nonmetal']:.3f}")
    print(f"Saved structure_proxy table and models/proxy_*.joblib ({rep['seconds']}s)")


def cmd_train(args: argparse.Namespace) -> None:
    from materialstack.clean import CleanConfig, parse_eg_methods
    from materialstack.models import train

    eg_methods = parse_eg_methods(args.eg_methods)
    cfg = CleanConfig(
        eg_methods=eg_methods,
        max_gap=args.max_gap,
        max_expt_spread=args.max_expt_spread,
        max_dft_conflict=args.max_dft_conflict,
    )
    result = train(
        run_xgb_baseline=not args.no_xgb_baseline,
        n_splits=args.n_splits,
        eg_methods=eg_methods,
        clean_config=cfg,
        use_structure=not args.no_structure,
    )
    m = result.metrics
    eg = m["eg_model"]["lightgbm"]
    edge = m["edge_model"]["lightgbm"]
    print("Primary backend: LightGBM")
    print(f"  Eg methods: {', '.join(m.get('eg_train_methods', []))}")
    if m.get("clean_summary"):
        cs = m["clean_summary"]
        print(f"  Clean audit: Eg {cs['n_eg_before']}→{cs['n_eg_after']}, "
              f"edge {cs['n_edge_before']}→{cs['n_edge_after']} "
              f"(SQLite unchanged; see train_clean_audit.json)")
    print(f"  Eg metal ACC={eg['metal_accuracy']:.3f}  F1={eg['metal_f1']:.3f}")
    print(f"  Eg nonmetal MAE={eg['gap_mae_nonmetal']:.3f} eV  RMSE={eg['gap_rmse_nonmetal']:.3f}  "
          f"R2={eg['gap_r2_nonmetal']:.3f}  (n={eg['n_nonmetals']})")
    if "xgboost_baseline" in m["eg_model"]:
        x = m["eg_model"]["xgboost_baseline"]
        print(f"  XGB baseline MAE={x['gap_mae_nonmetal']:.3f} eV")
    if m["eg_model"].get("borlido_holdout"):
        b = m["eg_model"]["borlido_holdout"]
        print(f"  Borlido holdout MAE={b.get('mae_nonmetal')} eV (n={b['n']})")
    h = m["eg_model"].get("structure_hybrid")
    if h:
        c, s = h["system"]["composition_only"], h["system"]["structure_aware"]
        print(f"  Structure-aware Eg: MAE={s['mae_nonmetal']:.3f} eV  R2={s['r2_nonmetal']:.3f}  "
              f"metal ACC={s['metal_accuracy']:.3f}  (composition-only {c['mae_nonmetal']:.3f} / "
              f"{c['r2_nonmetal']:.3f}; polymorph default={h['recommended_policy']})")
    print(f"  Edge CBM MAE={edge.get('cbm_mae'):.3f} eV  VBM MAE={edge.get('vbm_mae'):.3f} eV  "
          f"delta MAE={edge.get('delta_mae'):.3f} eV  (n={edge['n_train_materials']})")
    print(f"Saved: {result.eg_path}")
    print(f"       {result.edge_path}")
    print(f"       {result.metrics_path}  ({m['seconds']}s)")
    if m.get("paths", {}).get("clean_audit"):
        print(f"       {m['paths']['clean_audit']}")


def cmd_predict(args: argparse.Namespace) -> None:
    from materialstack.predict import resolve_layer, resolve_stack

    names = args.materials
    if len(names) == 1:
        r = resolve_layer(names[0], max_lookup_rank=args.max_lookup_rank, polymorph=args.polymorph)
        print(f"query={r.query}  formula={r.formula}  material_id={r.material_id}")
        print(f"  Eg  = {r.gap_ev} eV   [{r.gap_source}]  trusted={r.trusted_gap}")
        print(f"  CBM = {r.cbm_ev} eV   [{r.edge_source}]  trusted={r.trusted_edges}")
        print(f"  VBM = {r.vbm_ev} eV")
        if r.chi is not None:
            print(f"  χ   = {r.chi:.4f} eV (Mulliken)")
        if r.p_metal is not None:
            print(f"  p_metal = {r.p_metal:.3f}")
        for s in r.structures_used or []:
            erel = f"{s['e_rel']:.3f}" if s["e_rel"] is not None else "n/a"
            print(f"  structure {s['structure_id']} ({s['source']}, SG {s['space_group']}, "
                  f"ΔE={erel} eV/atom) → Eg={s['gap_ev']:.3f} eV")
        for n in r.notes or []:
            print(f"  note: {n}")
        return

    stack = resolve_stack(names, max_lookup_rank=args.max_lookup_rank, polymorph=args.polymorph)
    print(f"Lookup-first stack (max_lookup_rank={stack['max_lookup_rank']}, polymorph={stack['polymorph']})")
    for L in stack["layers"]:
        print(f"\n{L['query']} → {L['formula']}")
        print(f"  Eg={L['gap_ev']} [{L['gap_source']}] trusted={L['trusted_gap']}")
        print(f"  CBM={L['cbm_ev']}  VBM={L['vbm_ev']} [{L['edge_source']}] "
              f"trusted={L['trusted_edges']}")
        for n in L.get("notes") or []:
            print(f"  note: {n}")
    if stack["junctions"]:
        print("\nJunctions (deterministic from edges):")
        for j in stack["junctions"]:
            unc = " UNCERTAIN" if j.get("uncertain") else ""
            print(f"  {j['interface']}: Type {j.get('type')}{unc}  "
                  f"CBO={j.get('cbo_ev'):.3f} VBO={j.get('vbo_ev'):.3f}  ({j.get('reason')})")


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn
    uvicorn.run(
        "materialstack.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def cmd_lookup(args: argparse.Namespace) -> None:
    from materialstack.chem import clean_material_name
    from materialstack.db import connect, lookup, resolve_alias
    con = connect(DB_PATH)
    name = args.formula
    alias = resolve_alias(con, name)
    if alias:
        print(f"alias '{name}' -> formula={alias['formula']} class={alias['material_class']} "
              f"role={alias['role_hint']} note={alias['note']}")
        if not alias["formula"]:
            print("No stoichiometric formula: values must come from the literature CSV.")
            return
        name = alias["formula"]
    cn = clean_material_name(name)
    print(f"cleaned: formula={cn.formula_clean} phase='{cn.phase_tag}' class={cn.material_class}")
    if not cn.formula_clean:
        print("Formula could not be parsed.")
        return
    res = lookup(con, cn.formula_clean)
    if res["materials"].empty:
        print(f"No records for {res['formula']}.")
        return
    cols = ["material_key", "family", "anonymized_formula", "mulliken_chi", "tolerance_factor", "material_class"]
    print("\nMaterial:")
    print(res["materials"][cols].to_string(index=False))
    print("\nBest-ranked values:")
    print(res["best"][["material_key", "best_gap", "best_gap_method", "best_gap_source", "best_cbm", "best_vbm",
                       "best_edge_method", "n_records", "gap_min", "gap_max", "n_with_structure"]].to_string(index=False))
    print(f"\nAll records ({len(res['records'])}):")
    rc = ["source", "method", "gap_ev", "vbm_ev", "cbm_ev", "edge_reference", "surface_miller", "is_direct",
          "structure_id", "external_id"]
    print(res["records"][rc].head(args.limit).to_string(index=False))
    if not res["structures"].empty:
        print(f"\nStructures ({len(res['structures'])}):")
        print(res["structures"].head(args.limit).to_string(index=False))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="materialstack", description="MaterialStack materials DB and junction pipeline")
    p.add_argument("-v", "--verbose", action="count", default=1)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-db", help="download all sources and build data/materials_db.sqlite")
    b.add_argument("--only", nargs="*", metavar="SRC", help=f"build only these sources; choices: {ALL_SOURCES}")
    b.add_argument("--skip", nargs="*", default=[], metavar="SRC")
    b.add_argument("--heavy", action="store_true", help="include matbench_mp_gap and mp_all_20181018 (needs ~8 GB RAM)")
    b.add_argument("--rebuild", action="store_true", help="delete the DB and start over")
    b.add_argument("--mp-api-key", default=None, help="Materials Project API key (or set MP_API_KEY)")
    b.add_argument("--no-structures", action="store_true", help="skip storing CIFs (faster, smaller)")
    b.set_defaults(func=cmd_build_db)

    s = sub.add_parser("stats", help="print database statistics")
    s.set_defaults(func=cmd_stats)

    f = sub.add_parser("featurize", help="cache composition (and structure) features into the DB")
    f.add_argument("--n-jobs", type=int, default=1)
    f.add_argument("--no-structures", action="store_true")
    f.set_defaults(func=cmd_featurize)

    fs = sub.add_parser("featurize-structures", help="cache fast crystal-structure features for every CIF")
    fs.add_argument("--n-jobs", type=int, default=1)
    fs.add_argument("--rebuild", action="store_true", help="drop and recompute features_structure")
    fs.set_defaults(func=cmd_featurize_structures)

    tp = sub.add_parser("train-proxies", help="train DFT-gap proxy models (transfer learning from JARVIS/SNUMAT)")
    tp.add_argument("--n-splits", type=int, default=5)
    tp.set_defaults(func=cmd_train_proxies)

    t = sub.add_parser("train", help="train LightGBM Eg + band-edge models (XGBoost baseline comparison)")
    t.add_argument("--n-splits", type=int, default=5, help="GroupKFold splits (default 5)")
    t.add_argument("--no-xgb-baseline", action="store_true",
                   help="skip XGBoost comparison (faster; LightGBM still trains)")
    t.add_argument("--eg-methods", default="experiment,literature",
                   help="comma-separated Eg label methods (default: experiment,literature; "
                        "use experiment,HSE06,TBmBJ for the old mixed-fidelity run)")
    t.add_argument("--max-gap", type=float, default=12.0,
                   help="drop non-metal training rows with Eg above this (eV)")
    t.add_argument("--max-expt-spread", type=float, default=1.0,
                   help="drop materials whose experiment gaps span more than this (eV)")
    t.add_argument("--max-dft-conflict", type=float, default=2.0,
                   help="drop experiment materials whose nearest HSE/TBmBJ gap differs by more than this (eV)")
    t.add_argument("--no-structure", action="store_true",
                   help="skip the structure-aware hybrid Eg model")
    t.set_defaults(func=cmd_train)

    l = sub.add_parser("lookup", help="show all records for a formula or alias")
    l.add_argument("formula")
    l.add_argument("--limit", type=int, default=40)
    l.set_defaults(func=cmd_lookup)

    pr = sub.add_parser("predict", help="lookup-first Eg/CBM/VBM (ML only if DB rank too weak)")
    pr.add_argument("materials", nargs="+",
                    help="one material, or a stack top→bottom (junctions computed if ≥2)")
    pr.add_argument("--max-lookup-rank", type=int, default=2,
                    help="use DB gap if method_rank≤this (1=experiment/literature, 2=HSE06; default 2)")
    pr.add_argument("--polymorph", choices=["ground_state", "mean"], default=None,
                    help="structure choice for ML Eg: ground_state (lowest E_hull) or mean over polymorphs "
                         "(default: policy recommended at training time)")
    pr.set_defaults(func=cmd_predict)

    srv = sub.add_parser("serve", help="MaterialStack web UI + API (FastAPI)")
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--port", type=int, default=8000)
    srv.add_argument("--reload", action="store_true")
    srv.set_defaults(func=cmd_serve)

    args = p.parse_args(argv)
    _setup_logging(args.verbose)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
