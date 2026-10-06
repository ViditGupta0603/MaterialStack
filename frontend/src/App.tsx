import { useEffect, useState } from "react";
import type { FormEvent } from "react";
import { fetchMetrics, predictStack } from "./api";
import type { JunctionResult, LayerResult, Metric, MetricsResponse, PredictResponse } from "./api";
import "./App.css";

const EXAMPLES: { label: string; stack: string[] }[] = [
  { label: "Perovskite n-i-p", stack: ["TiO2", "MAPbI3", "Spiro-OMeTAD"] },
  { label: "Perovskite p-i-n", stack: ["PCBM", "MAPbI3", "NiO"] },
  { label: "CdS / CdTe", stack: ["CdS", "CdTe"] },
  { label: "CIS cell", stack: ["ZnO", "CdS", "CuInSe2"] },
  { label: "Si / GaAs", stack: ["Si", "GaAs"] },
];

function fmt(n: number | null | undefined, digits = 2) {
  if (n == null || Number.isNaN(n)) return "—";
  return n.toFixed(digits);
}

function pct(x: number | null | undefined) {
  return x == null ? "—" : `${Math.round(x * 100)}%`;
}

// ---------------------------------------------------------------- provenance in plain language

const EDGE_LABEL: Record<LayerResult["edge_kind"], string> = {
  measured: "Measured (photoemission)",
  surface: "Hybrid-DFT surfaces + electronegativity",
  estimate: "Electronegativity estimate (Butler–Ginley)",
  none: "Not available",
};

function gapTag(L: LayerResult): { text: string; tone: "ok" | "warn" | "muted" } {
  if (L.gap_kind === "measured") return { text: "Measured", tone: "ok" };
  if (L.gap_kind === "ML") return { text: "ML estimate", tone: "muted" };
  if (L.gap_kind === "metal") return { text: "Metal", tone: "muted" };
  return { text: "Not available", tone: "warn" };
}

// ---------------------------------------------------------------- app

export default function App() {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [input, setInput] = useState("TiO2\nMAPbI3\nSpiro-OMeTAD");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);

  useEffect(() => {
    fetchMetrics().then(setMetrics).catch(() => setMetrics(null));
  }, []);

  async function run(materials: string[]) {
    if (!materials.length) return;
    setBusy(true);
    setError(null);
    try {
      setResult(await predictStack(materials));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  function onSubmit(e: FormEvent) {
    e.preventDefault();
    run(input.split(/[\n,]+/).map((s) => s.trim()).filter(Boolean));
  }

  return (
    <div className="page">
      <header className="nav">
        <div className="nav-title">
          <span className="nav-brand">MaterialStack</span>
          <span className="nav-tag">Band-alignment screening for device stacks</span>
        </div>
        <nav className="nav-links">
          <a href="#predict">Predict</a>
          <a href="#validation">Validation</a>
          <a href="#method">Method</a>
        </nav>
      </header>

      <section className="section" id="predict">
        <form className="input-panel" onSubmit={onSubmit}>
          <label className="field-label" htmlFor="stack">
            Layers, top first — one formula or common name per line
          </label>
          <textarea
            id="stack"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            rows={4}
            placeholder={"TiO2\nMAPbI3\nSpiro-OMeTAD"}
            spellCheck={false}
          />
          <div className="examples">
            <span className="muted">Examples:</span>
            {EXAMPLES.map((ex) => (
              <button
                key={ex.label}
                type="button"
                className="chip"
                onClick={() => {
                  setInput(ex.stack.join("\n"));
                  run(ex.stack);
                }}
              >
                {ex.label}
              </button>
            ))}
          </div>
          <div className="actions">
            <button className="btn primary" type="submit" disabled={busy}>
              {busy ? "Predicting…" : "Predict alignment"}
            </button>
          </div>
          {error && <p className="error">{error}</p>}
        </form>

        {result && <Results result={result} />}
      </section>

      <section className="section" id="validation">
        <h2>Validation</h2>
        <p className="lede">
          Scored on data the model never trained on. Run <code>python validate.py</code> to reproduce;
          the full table is in <code>results/metrics.csv</code>.
        </p>
        {metrics ? <Validation m={metrics} /> : <p className="muted">Metrics not available: run validate.py.</p>}
      </section>

      <section className="section" id="method">
        <h2>Method</h2>
        <ol className="method">
          <li>
            <b>Band gap.</b> A measured value when one exists (Borlido 2019, curated literature, else the
            consensus of the Zhuo 2018 compilation). Otherwise one LightGBM model predicts it from the
            formula, using a DFT gap (JARVIS, SNUMAT) as an extra hint when the formula has one.
          </li>
          <li>
            <b>Band edges.</b> Measured ionization energy / electron affinity (photoemission papers) when
            available. Otherwise hybrid-DFT surface calculations for oxides (Kiyohara 2024), else the
            electronegativity estimate VBM = −χ − E<sub>g</sub>/2. CBM = VBM + E<sub>g</sub>.
          </li>
          <li>
            <b>Junctions.</b> A measured interface offset where one exists, otherwise both layers aligned to
            the vacuum level. Type I/II/III follows from the four band edges; its probability comes from the
            validated error of each input. Below 80% the tool asks for a DFT check.
          </li>
        </ol>
      </section>

      <footer className="footer">
        <span>MaterialStack</span>
        <span className="muted">Data: Zhuo 2018, Borlido 2019, JARVIS-DFT, SNUMAT, Kiyohara 2024, photoemission literature</span>
      </footer>
    </div>
  );
}

// ---------------------------------------------------------------- results

function Results({ result }: { result: PredictResponse }) {
  const layers = result.layers;
  return (
    <div className="results">
      <div className="results-grid">
        <div className="panel">
          <h3>Layers</h3>
          <div className="table-wrap">
            <table className="layers">
              <thead>
                <tr>
                  <th>Layer</th>
                  <th className="num">
                    E<sub>g</sub> (eV)
                  </th>
                  <th className="num">VBM (eV)</th>
                  <th className="num">CBM (eV)</th>
                  <th>Source</th>
                </tr>
              </thead>
              <tbody>
                {layers.map((L, i) => {
                  const tag = gapTag(L);
                  return (
                    <tr key={`${L.query}-${i}`}>
                      <td>
                        <div className="layer-name">{L.query}</div>
                        {L.display_formula && L.display_formula !== L.query && (
                          <div className="muted small">{L.display_formula}</div>
                        )}
                      </td>
                      <td className="num">{fmt(L.gap_ev)}</td>
                      <td className="num">{fmt(L.vbm_ev)}</td>
                      <td className="num">{fmt(L.cbm_ev)}</td>
                      <td>
                        <span className={`badge ${tag.tone}`}>{tag.text}</span>
                        <div className="small" title={L.gap_source}>
                          Gap: {L.gap_source || "not available"}
                        </div>
                        <div className="small" title={L.edge_source}>
                          Edges: {EDGE_LABEL[L.edge_kind]}
                        </div>
                        {L.notes.map((n) => (
                          <div className="small note" key={n}>
                            {n}
                          </div>
                        ))}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
        <div className="panel">
          <h3>Energy levels vs vacuum</h3>
          <BandDiagram layers={layers} />
          <div className="legend small">
            <span className="key cbm-key" /> CBM
            <span className="key vbm-key" /> VBM
            <span className="muted">· shaded: band gap</span>
          </div>
        </div>
      </div>

      {result.junctions.length > 0 && (
        <div className="panel">
          <h3>Junctions</h3>
          {result.junctions.map((j, i) => (
            <JunctionRow key={j.interface} j={j} top={layers[i]} bottom={layers[i + 1]} />
          ))}
        </div>
      )}
    </div>
  );
}

/** Where carriers go across one interface, in words. */
function carrierFlow(j: JunctionResult, top: LayerResult, bottom: LayerResult): string | null {
  if (j.type == null || j.cbo_ev == null || j.vbo_ev == null) return null;
  if (j.type === "III") return "Broken gap: the bands do not overlap (tunnelling / recombination junction).";
  // cbo = CBM(bottom) − CBM(top): > 0 → electrons settle in the top layer.
  const electrons = j.cbo_ev > 0 ? top.query : bottom.query;
  // vbo = VBM(top) − VBM(bottom): > 0 → holes settle in the top layer.
  const holes = j.vbo_ev > 0 ? top.query : bottom.query;
  return electrons === holes
    ? `Both electrons and holes collect in ${electrons} (carriers confined).`
    : `Electrons collect in ${electrons}, holes in ${holes} (charge separation).`;
}

function JunctionRow({ j, top, bottom }: { j: JunctionResult; top: LayerResult; bottom: LayerResult }) {
  const probs = j.type_probabilities;
  const flow = carrierFlow(j, top, bottom);
  return (
    <div className="junction">
      <div className="junction-head">
        <strong>
          {top.query} / {bottom.query}
        </strong>
        {j.type ? (
          <span className={`badge ${j.uncertain ? "warn" : "ok"}`}>
            Type {j.type} · {pct(j.confidence)}
            {j.uncertain ? " · check with DFT" : ""}
          </span>
        ) : (
          <span className="badge muted">{j.reason}</span>
        )}
      </div>
      {j.type && (
        <>
          <div className="junction-values">
            <span>
              ΔE<sub>v</sub> {fmt(j.vbo_ev)} ± {fmt(j.vbo_sigma_ev)} eV
            </span>
            <span>
              ΔE<sub>c</sub> {fmt(j.cbo_ev)} eV
            </span>
            <span className="muted">Offset: {j.offset_source}</span>
          </div>
          {flow && <p className="small">{flow}</p>}
          {probs && (
            <div className="prob-bar" aria-label="Junction type probabilities">
              {(["I", "II", "III"] as const).map((k) =>
                probs[k] > 0.005 ? (
                  <span key={k} className={`prob-${k}`} style={{ width: `${probs[k] * 100}%` }}>
                    {probs[k] > 0.12 ? `Type ${k} ${Math.round(probs[k] * 100)}%` : ""}
                  </span>
                ) : null,
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}

/** Each layer as a box from VBM to CBM on a common vacuum scale. */
function BandDiagram({ layers }: { layers: LayerResult[] }) {
  const ok = layers.filter((L) => L.vbm_ev != null && L.cbm_ev != null);
  if (!ok.length) return <p className="muted">No band edges to draw.</p>;
  const W = 520;
  const H = 300;
  const pad = { l: 44, r: 10, t: 22, b: 36 };
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
          <text x={pad.l - 6} y={y(e) + 4} className="axis" textAnchor="end">
            {e}
          </text>
        </g>
      ))}
      <text x={4} y={12} className="axis">
        eV
      </text>
      {layers.map((L, i) => {
        if (L.vbm_ev == null || L.cbm_ev == null) return null;
        const x = pad.l + i * colW + colW * 0.15;
        const w = colW * 0.7;
        const yc = y(L.cbm_ev);
        const yv = y(L.vbm_ev);
        return (
          <g key={`${L.query}-${i}`}>
            <rect x={x} y={yc} width={w} height={yv - yc} className="gap-box" />
            <line x1={x} x2={x + w} y1={yc} y2={yc} className="cbm" />
            <line x1={x} x2={x + w} y1={yv} y2={yv} className="vbm" />
            <text x={x + w / 2} y={yc - 5} className="level" textAnchor="middle">
              {fmt(L.cbm_ev)}
            </text>
            <text x={x + w / 2} y={yv + 14} className="level" textAnchor="middle">
              {fmt(L.vbm_ev)}
            </text>
            <text x={x + w / 2} y={H - 12} className="label" textAnchor="middle">
              {L.query}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

// ---------------------------------------------------------------- validation

function find(m: Metric[], name: string): Metric | undefined {
  return m.find((x) => x.metric === name);
}

function show(x: Metric | undefined) {
  if (!x || x.value == null) return "—";
  return x.unit === "%" ? `${Math.round(x.value)}%` : `${fmt(x.value)}${x.unit ? ` ${x.unit}` : ""}`;
}

function Validation({ m }: { m: MetricsResponse }) {
  const rows = m.metrics;
  const sections = [...new Set(rows.map((r) => r.section))];
  const gap = find(rows, "MAE");
  const offsets = find(rows, "|ΔEv| MAE, tool");
  const sign = find(rows, "ΔEc sign agreement (|gold| ≥ 0.1 eV), tool");
  const confident = find(rows, "Type accuracy when confident (≥ 80 %), tool");
  return (
    <>
      <div className="stat-row">
        <Stat label="Band-gap error, unseen chemistry" value={show(gap)} note={`${gap?.n ?? "—"} materials, 5-fold CV`} />
        <Stat label="Band-offset error vs experiment" value={show(offsets)} note={`${offsets?.n ?? "—"} measured interfaces`} />
        <Stat label="Spike vs cliff right (ΔEc sign)" value={show(sign)} note={`${sign?.n ?? "—"} gold device junctions`} />
        <Stat label="Confident junction calls correct" value={show(confident)} note={`${confident?.n ?? "—"} calls with ≥ 80%`} />
      </div>
      <details className="details">
        <summary>All metrics</summary>
        {sections.map((sec) => (
          <div className="panel" key={sec}>
            <h3>{sec}</h3>
            <div className="table-wrap">
              <table className="kv">
                <thead>
                  <tr>
                    <th>Metric</th>
                    <th className="num">Value</th>
                    <th className="num">Baseline</th>
                    <th className="num">n</th>
                  </tr>
                </thead>
                <tbody>
                  {rows
                    .filter((r) => r.section === sec)
                    .map((r) => (
                      <tr key={r.metric} title={[r.note, r.baseline_rule && `baseline: ${r.baseline_rule}`].filter(Boolean).join(" · ")}>
                        <td>{r.metric}</td>
                        <td className="num">{show(r)}</td>
                        <td className="num">{r.baseline == null ? "" : show({ ...r, value: r.baseline })}</td>
                        <td className="num">{r.n ?? ""}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </div>
        ))}
      </details>
    </>
  );
}

function Stat({ label, value, note }: { label: string; value: string; note: string }) {
  return (
    <div className="stat">
      <span className="stat-label">{label}</span>
      <strong>{value}</strong>
      <span className="muted small">{note}</span>
    </div>
  );
}
