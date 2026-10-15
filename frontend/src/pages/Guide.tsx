import type { MetricsResponse } from "../api";
import { findMetric, showMetric } from "../format";

export default function Guide({ metrics }: { metrics: MetricsResponse | null }) {
  const rows = metrics?.metrics;
  const gapMae = showMetric(findMetric(rows, "MAE"));
  const bgMae = showMetric(findMetric(rows, "VBM MAE, Butler–Ginley estimate"));
  const conf = findMetric(rows, "Type accuracy when confident (≥ 80 %), tool");

  const notes: { title: string; text: string }[] = [
    {
      title: "A screening tool",
      text: "It sorts stacks into likely-fine and needs-a-closer-look. It does not replace DFT or measurement for a final design.",
    },
    {
      title: "Band gaps",
      text: `A measured gap is used when one is known; otherwise the ML model predicts it (about ${gapMae} error on unseen chemistry). The model sees the formula, not the crystal structure, so polymorphs such as anatase vs rutile TiO2 or black vs yellow CsPbI3 get the same gap.`,
    },
    {
      title: "Band edges",
      text: `Measured for about 47 materials and computed with hybrid DFT for oxides. Everything else uses the electronegativity estimate (about ${bgMae} error; in hybrid perovskites the organic cation is counted as Cs). It comes out 2–2.5 eV too deep for Cu(I) and Ni hole layers such as CuI, CuSCN and NiO when they are not measured. Even measured oxide edges vary by up to 1 eV with surface and preparation.`,
    },
    {
      title: "Organic layers",
      text: "Only organics with a measured value are supported (Spiro-OMeTAD, PCBM, C60 and a few more); the model cannot predict molecules.",
    },
    {
      title: "Interfaces",
      text: "Layers are aligned to the vacuum level, which ignores interface dipoles. Measured interface offsets are used where they exist (CdS/CuInSe2, ZnO/CdS).",
    },
    {
      title: "Not modelled",
      text: "Doping, layer thickness, strain, temperature and defects. The edges are for ideal, undoped materials.",
    },
    {
      title: "Confidence",
      text: `The percentage is how often the junction type survives the typical error of every input (Monte Carlo). Calls at 80% or more were right ${showMetric(conf)} of the time on the device stacks (${conf?.n ?? "—"} calls); below 80% the tool says to check with DFT.`,
    },
    {
      title: "Validation reference",
      text: "The device stacks used for validation take their band edges from SCAPS simulation inputs, which themselves differ from photoemission by 0.4–0.8 eV for some layers, so they set a ceiling on the scores.",
    },
  ];

  return (
    <div className="stack-lg">
      <header className="page-head">
        <h1>Guide</h1>
        <p className="lede">How to use MaterialStack, and what to keep in mind when reading its answers.</p>
      </header>

      <div className="guide-grid">
        <section className="card">
          <h2 className="card-title">How to use</h2>
          <ol className="howto">
            <li>
              <b>Enter the stack</b> on the <a href="#/predict">Predict</a> page, top layer first, one per line. Use a
              formula (<code>TiO2</code>, <code>CuInSe2</code>) or a common name (<code>MAPbI3</code>, <code>CIGS</code>,{" "}
              <code>Spiro-OMeTAD</code>, <code>PCBM</code>). Commas also work.
            </li>
            <li>
              <b>Press Predict alignment</b>, or click one of the example stacks.
            </li>
            <li>
              <b>Read the diagram.</b> Each layer is a box from its VBM (orange) to its CBM (blue) on a common vacuum
              scale; the height of the box is the band gap. Solid lines are measured, dashed lines estimated.
            </li>
            <li>
              <b>Read the junction cards.</b> Each interface shows its type (I, II or III), the confidence, the
              offsets ΔE<sub>v</sub> and ΔE<sub>c</sub> with their uncertainty, and the probability of each type.
            </li>
            <li>
              <b>Check the sources</b> in the layer table: green badges are measured values (with the paper), grey
              are DFT or ML values, amber are rough estimates. An amber{" "}
              <span className="badge warn">check with DFT</span> on a junction means the type is below 80% confidence.
            </li>
            <li>
              <b>Keep the result.</b> <i>Download CSV</i> saves the layers and junctions; the address bar holds the
              stack, so the page can be bookmarked or the link shared.
            </li>
          </ol>
        </section>

        <section className="card">
          <h2 className="card-title">Signs and types</h2>
          <dl className="defs">
            <dt>ΔE<sub>v</sub></dt>
            <dd>VBM(top) − VBM(bottom). Positive: holes stay in the top layer.</dd>
            <dt>ΔE<sub>c</sub></dt>
            <dd>CBM(bottom) − CBM(top). Positive: electrons stay in the top layer.</dd>
            <dt>Type I</dt>
            <dd>Straddling: one layer holds both electrons and holes (confinement, LEDs).</dd>
            <dt>Type II</dt>
            <dd>Staggered: electrons and holes end up in different layers (charge separation, solar cells).</dd>
            <dt>Type III</dt>
            <dd>Broken gap: the bands do not overlap (tunnel / recombination junctions).</dd>
          </dl>
        </section>
      </div>

      <section>
        <h2 className="section-title">Notes</h2>
        <div className="note-grid">
          {notes.map((n) => (
            <div className="card note-card" key={n.title}>
              <h3>{n.title}</h3>
              <p>{n.text}</p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
