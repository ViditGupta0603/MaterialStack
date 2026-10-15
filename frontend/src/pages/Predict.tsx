import type { FormEvent } from "react";
import type { JunctionResult, LayerResult, PredictResponse } from "../api";
import BandDiagram from "../BandDiagram";
import { fmt, pct, splitLayers } from "../format";

/** Stacks whose junction types the tool gets right with ≥ 80% confidence, checked against the published
 *  device stacks in data/multilayer_gold_standard.csv or textbook band alignments (GaN/AlN). */
const EXAMPLES: { label: string; stack: string[] }[] = [
  { label: "Perovskite cell", stack: ["TiO2", "MAPbI3", "Spiro-OMeTAD"] },
  { label: "Perovskite, NiO hole layer", stack: ["TiO2", "MAPbI3", "NiO"] },
  { label: "Inorganic perovskite", stack: ["CsPbI3", "CuSCN"] },
  { label: "Thin-film window", stack: ["ZnO", "CdS"] },
  { label: "Oxide heterojunction", stack: ["ZnO", "Cu2O"] },
  { label: "Nitride LED (Type I)", stack: ["GaN", "AlN"] },
];

const EDGE_TAG: Record<LayerResult["edge_kind"], { text: string; tone: "ok" | "warn" | "muted" }> = {
  measured: { text: "Measured edges", tone: "ok" },
  surface: { text: "DFT edges", tone: "muted" },
  estimate: { text: "Estimated edges", tone: "warn" },
  none: { text: "No edges", tone: "warn" },
};

const EDGE_LABEL: Record<LayerResult["edge_kind"], string> = {
  measured: "measured (photoemission)",
  surface: "hybrid-DFT surface calculations + electronegativity",
  estimate: "electronegativity estimate (Butler–Ginley), about ±1 eV",
  none: "not available",
};

function gapTag(L: LayerResult): { text: string; tone: "ok" | "warn" | "muted" } {
  if (L.gap_kind === "measured") return { text: "Measured gap", tone: "ok" };
  if (L.gap_kind === "ML") return { text: "ML gap", tone: "muted" };
  if (L.gap_kind === "metal") return { text: "Metal", tone: "muted" };
  return { text: "No gap", tone: "warn" };
}

/** The citation part of a measured edge source ("measured (photoemission): Tao et al. …" → "Tao et al. …"). */
function edgeReference(L: LayerResult): string | null {
  return L.edge_kind === "measured" ? L.edge_source.split(": ").slice(1).join(": ") || null : null;
}

type Props = {
  input: string;
  setInput: (s: string) => void;
  run: (materials: string[]) => void;
  busy: boolean;
  error: string | null;
  result: PredictResponse | null;
};

export default function Predict({ input, setInput, run, busy, error, result }: Props) {
  function onSubmit(e: FormEvent) {
    e.preventDefault();
    run(splitLayers(input));
  }

  return (
    <div className="predict">
      <aside className="predict-side">
        <form className="card input-card" onSubmit={onSubmit}>
          <h2 className="card-title">Device stack</h2>
          <label className="field-label" htmlFor="stack">
            Layers from top to bottom, one formula or common name per line
          </label>
          <textarea
            id="stack"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            rows={5}
            placeholder={"TiO2\nMAPbI3\nSpiro-OMeTAD"}
            spellCheck={false}
          />
          <button className="btn primary" type="submit" disabled={busy}>
            {busy ? "Predicting…" : "Predict alignment"}
          </button>
          {error && <p className="error">{error}</p>}
          <div className="examples">
            <span className="eyebrow">Examples</span>
            {EXAMPLES.map((ex) => (
              <button
                key={ex.label}
                type="button"
                className="example"
                disabled={busy}
                onClick={() => {
                  setInput(ex.stack.join("\n"));
                  run(ex.stack);
                }}
              >
                <span>{ex.label}</span>
                <span className="muted small">{ex.stack.join(" / ")}</span>
              </button>
            ))}
          </div>
        </form>
        <div className="card key-card small">
          <span className="eyebrow">Reading the result</span>
          <p>
            <span className="key cbm-key" /> CBM and <span className="key vbm-key" /> VBM vs vacuum; the shaded box
            is the band gap. Dashed lines are estimated, solid lines measured.
          </p>
          <p>
            <span className="badge ok">Type II · 91%</span> confident call.{" "}
            <span className="badge warn">check with DFT</span> below 80%.
          </p>
          <p>
            <a href="#/guide">How to use and notes →</a>
          </p>
        </div>
      </aside>

      <div className="predict-main">
        {result ? (
          <Results result={result} />
        ) : (
          <div className="card empty">{busy ? "Predicting…" : "Enter a stack and press Predict alignment."}</div>
        )}
      </div>
    </div>
  );
}

function Results({ result }: { result: PredictResponse }) {
  const layers = result.layers;
  return (
    <>
      <section className="card">
        <div className="card-head">
          <h2 className="card-title">Energy levels vs vacuum</h2>
          <div className="legend small">
            <span className="key cbm-key" /> CBM
            <span className="key vbm-key" /> VBM
            <span className="key dashed-key" /> estimated
            <span className="muted nowrap">· shaded: band gap</span>
          </div>
        </div>
        <BandDiagram layers={layers} />
        {result.junctions.some((j) => j.offset_source?.startsWith("measured")) && (
          <p className="small muted caption">
            Junctions marked <i>Measured offset</i> use the offset measured at the real interface, so their ΔE
            values differ from the gaps drawn here (each layer placed against vacuum on its own).
          </p>
        )}
      </section>

      {result.junctions.length > 0 ? (
        <section>
          <h2 className="section-title">Junctions</h2>
          <div className="junction-grid">
            {result.junctions.map((j, i) => (
              <JunctionCard key={j.interface} j={j} top={layers[i]} bottom={layers[i + 1]} />
            ))}
          </div>
        </section>
      ) : (
        <p className="card small muted">Add a second layer to get a junction type.</p>
      )}

      <section className="card">
        <div className="card-head">
          <h2 className="card-title">Layers</h2>
          <button className="btn ghost small" type="button" onClick={() => downloadCsv(result)}>
            Download CSV
          </button>
        </div>
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
                const edge = EDGE_TAG[L.edge_kind];
                const ref = edgeReference(L);
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
                      <div className="badges">
                        <span className={`badge ${tag.tone}`}>{tag.text}</span>
                        {L.gap_kind !== "none" && <span className={`badge ${edge.tone}`}>{edge.text}</span>}
                      </div>
                      {L.gap_source && <div className="small">Gap: {L.gap_source}</div>}
                      {L.gap_kind !== "none" && <div className="small">Edges: {EDGE_LABEL[L.edge_kind]}</div>}
                      {ref && <div className="small muted ref">{ref}</div>}
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
      </section>
    </>
  );
}

function JunctionCard({ j, top, bottom }: { j: JunctionResult; top: LayerResult; bottom: LayerResult }) {
  const probs = j.type_probabilities;
  const measured = j.offset_source?.startsWith("measured");
  return (
    <div className="card junction">
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
            <div>
              <span className="eyebrow">
                ΔE<sub>v</sub>
              </span>
              <b>
                {fmt(j.vbo_ev)} <span className="muted">± {fmt(j.vbo_sigma_ev)}</span> eV
              </b>
            </div>
            <div>
              <span className="eyebrow">
                ΔE<sub>c</sub>
              </span>
              <b>
                {fmt(j.cbo_ev)} <span className="muted">± {fmt(j.cbo_sigma_ev)}</span> eV
              </b>
            </div>
            {measured && (
              <span className="badge ok" title={j.offset_source}>
                Measured offset
              </span>
            )}
          </div>
          {probs && (
            <div className="prob-bar" aria-label="Junction type probabilities">
              {(["I", "II", "III"] as const).map((k) =>
                probs[k] > 0.005 ? (
                  <span key={k} className={`prob-${k}`} style={{ width: `${probs[k] * 100}%` }}>
                    {probs[k] > 0.3
                      ? `Type ${k} ${Math.round(probs[k] * 100)}%`
                      : probs[k] > 0.08
                        ? `${k} ${Math.round(probs[k] * 100)}%`
                        : ""}
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

/** Layers and junctions as one CSV file (opens in Excel). */
function downloadCsv(result: PredictResponse) {
  const cell = (v: unknown) => {
    const s = v == null ? "" : typeof v === "number" ? v.toFixed(3) : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const rows: unknown[][] = [
    ["layer", "formula", "gap_ev", "vbm_ev", "cbm_ev", "gap_source", "edge_source", "notes"],
    ...result.layers.map((L) => [
      L.query, L.display_formula, L.gap_ev, L.vbm_ev, L.cbm_ev, L.gap_source, L.edge_source, L.notes.join("; "),
    ]),
    [],
    ["junction", "type", "confidence", "dEv_ev", "dEv_sigma_ev", "dEc_ev", "dEc_sigma_ev", "offset_source"],
    ...result.junctions.map((j) => [
      j.interface.replace(" | ", " / "), j.type ?? j.reason, j.confidence, j.vbo_ev, j.vbo_sigma_ev, j.cbo_ev,
      j.cbo_sigma_ev, j.offset_source,
    ]),
  ];
  const blob = new Blob([rows.map((r) => r.map(cell).join(",")).join("\n")], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `materialstack_${result.layers.map((L) => L.query).join("_").replace(/[^A-Za-z0-9_-]+/g, "")}.csv`;
  a.click();
  URL.revokeObjectURL(a.href);
}
