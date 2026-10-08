import type { Metric, MetricsResponse } from "../api";
import { fmt, findMetric, showMetric } from "../format";

/** The sections shown as tables (A: band gap, B: band edges), with what each checks. */
const TABLES: Record<string, (rows: Metric[]) => string> = {
  A: (rows) => {
    const hybrid = findMetric(rows, "MAE where a hybrid DFT gap exists");
    return (
      "The ML band-gap model, scored by 5-fold cross-validation in which materials made of the same elements " +
      "stay in the same fold, so every score is for chemistry the model has not seen." +
      (hybrid?.baseline != null
        ? ` For comparison, hybrid DFT alone is off by ${fmt(hybrid.baseline)} eV on the same materials.`
        : "")
    );
  },
  B: () =>
    "For every material with a measured VBM, the measurement is hidden and the rule used for unmeasured layers " +
    "(hybrid-DFT surfaces, or the electronegativity estimate) is compared with it. Bias < 0: predicted too deep.",
};

export default function Validation({ metrics }: { metrics: MetricsResponse | null }) {
  if (!metrics) return <div className="card empty">Validation results are not available.</div>;
  const rows = metrics.metrics;
  const sections = [...new Set(rows.map((r) => r.section))].filter((sec) => sec.split(".")[0] in TABLES);
  const gap = findMetric(rows, "MAE");
  const offsets = findMetric(rows, "|ΔEv| MAE, tool");
  // several of these interfaces' measured edges come from the same paper (InterMat), so show the clean number too
  const offsetsClean = findMetric(rows, "|ΔEv| MAE, tool without measured edges");
  const sign = findMetric(rows, "ΔEc sign agreement (|gold| ≥ 0.1 eV), tool");
  const confident = findMetric(rows, "Type accuracy when confident (≥ 80 %), tool");

  return (
    <div className="stack-lg">
      <header className="page-head">
        <h1>Validation</h1>
        <p className="lede">
          Every number is scored on data the tool did not use for that answer: unseen materials for the band gap,
          hidden measurements for the band edges, and experimentally measured interfaces and published device stacks
          for the junctions.
        </p>
      </header>

      <div className="stat-row">
        <Stat label="Band-gap error, unseen chemistry" value={showMetric(gap)} note={`${gap?.n ?? "—"} materials, 5-fold CV`} />
        <Stat
          label="Band-offset error vs experiment"
          value={showMetric(offsets)}
          note={`ΔEv at ${offsets?.n ?? "—"} measured interfaces; ${showMetric(offsetsClean)} without our measured edges`}
        />
        <Stat label="Spike vs cliff right (ΔEc sign)" value={showMetric(sign)} note={`${sign?.n ?? "—"} junctions in published devices`} />
        <Stat label="Confident junction calls correct" value={showMetric(confident)} note={`${confident?.n ?? "—"} calls with ≥ 80% confidence`} />
      </div>

      <div className="metric-grid">
        {sections.map((sec) => (
          <section className="card" key={sec}>
            <h2 className="card-title">{sec}</h2>
            <p className="small muted about">{TABLES[sec.split(".")[0]](rows)}</p>
            <div className="table-wrap">
              <table className="kv">
                <thead>
                  <tr>
                    <th>Metric</th>
                    <th className="num">Value</th>
                    <th className="num">Materials</th>
                  </tr>
                </thead>
                <tbody>
                  {rows
                    .filter((r) => r.section === sec)
                    .map((r) => (
                      <tr key={r.metric} title={r.note || undefined}>
                        <td>{r.metric}</td>
                        <td className="num strong">{showMetric(r)}</td>
                        <td className="num muted">{r.n ?? ""}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </section>
        ))}
      </div>
    </div>
  );
}

function Stat({ label, value, note }: { label: string; value: string; note: string }) {
  return (
    <div className="card stat">
      <span className="stat-label">{label}</span>
      <strong>{value}</strong>
      <span className="muted small">{note}</span>
    </div>
  );
}
