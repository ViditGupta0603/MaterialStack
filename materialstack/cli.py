"""Command line:  python -m materialstack predict TiO2 MAPbI3 Spiro-OMeTAD
                 python -m materialstack serve [--port 8000]

Building the data, training and validating are the three scripts in the repo root:
build_data.py, train.py and validate.py.
"""
from __future__ import annotations

import argparse
import logging
import warnings


def _fmt(x, nd=2) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def cmd_predict(args: argparse.Namespace) -> None:
    from materialstack.predict import predict_stack
    out = predict_stack(args.layers)
    print(f"{'layer':16s} {'Eg':>6s} {'VBM':>7s} {'CBM':>7s}  source")
    for L in out["layers"]:
        print(f"{L['query']:16s} {_fmt(L['gap_ev']):>6s} {_fmt(L['vbm_ev']):>7s} {_fmt(L['cbm_ev']):>7s}  "
              f"gap: {L['gap_source']}; edges: {L['edge_source'][:70]}")
        for n in L["notes"]:
            print(f"{'':16s} note: {n}")
    print()
    for j in out["junctions"]:
        if j["type"] is None:
            print(f"{j['interface']:30s} —  ({j['reason']})")
            continue
        p = j["type_probabilities"]
        print(f"{j['interface']:30s} Type {j['type']:3s} ({j['confidence']:.0%}{', check with DFT' if j['uncertain'] else ''})"
              f"  ΔEv {j['vbo_ev']:+.2f} ± {j['vbo_sigma_ev']:.2f} eV, ΔEc {j['cbo_ev']:+.2f} ± {j['cbo_sigma_ev']:.2f} eV"
              f"  [I {p['I']:.0%} / II {p['II']:.0%} / III {p['III']:.0%}]  offset: {j['offset_source'][:45]}")


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn
    uvicorn.run("materialstack.api:app", host=args.host, port=args.port, reload=args.reload)


def main() -> None:
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.WARNING)
    p = argparse.ArgumentParser(prog="materialstack", description="Band alignment of device stacks")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("predict", help="layers from top to bottom, e.g. TiO2 MAPbI3 Spiro-OMeTAD")
    s.add_argument("layers", nargs="+")
    s.set_defaults(func=cmd_predict)
    s = sub.add_parser("serve", help="web UI + API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--reload", action="store_true")
    s.set_defaults(func=cmd_serve)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
