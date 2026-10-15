import { useEffect, useRef, useState } from "react";
import type { LayerResult } from "./api";
import { fmt } from "./format";

/** Each layer as a box from VBM to CBM on a common vacuum scale; edges that are not measured are dashed. Drawn at the card's real width, so the
 *  text keeps the same size on a phone and on a wide screen. */
export default function BandDiagram({ layers }: { layers: LayerResult[] }) {
  const box = useRef<HTMLDivElement>(null);
  const [W, setW] = useState(900);
  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(280, Math.round(e.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const ok = layers.filter((L) => L.vbm_ev != null && L.cbm_ev != null);
  return <div ref={box}>{ok.length ? draw(layers, ok, W) : <p className="muted">No band edges to draw.</p>}</div>;
}

function draw(layers: LayerResult[], ok: LayerResult[], W: number) {
  const H = Math.min(440, Math.max(300, W * 0.45));
  const pad = { l: 52, r: 14, t: 28, b: 44 };
  const hi = Math.max(...ok.map((L) => L.cbm_ev as number)) + 0.5;
  const lo = Math.min(...ok.map((L) => L.vbm_ev as number)) - 0.5;
  const y = (e: number) => pad.t + ((hi - e) / (hi - lo)) * (H - pad.t - pad.b);
  const colW = (W - pad.l - pad.r) / layers.length;
  const ticks: number[] = [];
  for (let e = Math.ceil(lo); e <= Math.floor(hi); e += 1) ticks.push(e);
  return (
    <svg className="band-diagram" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Band alignment diagram">
      {ticks.map((e) => (
        <g key={e}>
          <line x1={pad.l} x2={W - pad.r} y1={y(e)} y2={y(e)} className="grid" />
          <text x={pad.l - 8} y={y(e) + 5} className="axis" textAnchor="end">
            {e}
          </text>
        </g>
      ))}
      <text x={6} y={16} className="axis">
        eV
      </text>
      {layers.map((L, i) => {
        if (L.vbm_ev == null || L.cbm_ev == null) return null;
        const x = pad.l + i * colW + colW * 0.15;
        const w = colW * 0.7;
        const yc = y(L.cbm_ev);
        const yv = y(L.vbm_ev);
        const est = L.edge_kind === "measured" ? "" : " estimated"; // dashed: not measured
        return (
          <g key={`${L.query}-${i}`}>
            <rect x={x} y={yc} width={w} height={yv - yc} className="gap-box" />
            <line x1={x} x2={x + w} y1={yc} y2={yc} className={`cbm${est}`} />
            <line x1={x} x2={x + w} y1={yv} y2={yv} className={`vbm${est}`} />
            <text x={x + w / 2} y={yc - 7} className="level" textAnchor="middle">
              {fmt(L.cbm_ev)}
            </text>
            <text x={x + w / 2} y={yv + 18} className="level" textAnchor="middle">
              {fmt(L.vbm_ev)}
            </text>
            <text x={x + w / 2} y={H - 14} className="label" textAnchor="middle">
              {colW < 110 && L.query.length > 9 ? `${L.query.slice(0, 8)}…` : L.query}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
