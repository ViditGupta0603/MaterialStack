const STEPS: { title: string; what: string; order: string[] }[] = [
  {
    title: "1 · Band gap",
    what: "Width of the gap of each layer.",
    order: [
      "Measured gap (Borlido 2019 or curated papers preferred, else the consensus of the Zhuo 2018 reports)",
      "Only metallic elements → metal",
      "LightGBM model from the formula (~150 Magpie features) plus the DFT gap of the formula as a hint when JARVIS / SNUMAT has one",
    ],
  },
  {
    title: "2 · Band edges",
    what: "Where the VBM and CBM sit relative to vacuum.",
    order: [
      "Measured ionization energy / electron affinity (photoemission papers): VBM = −IE, CBM = −EA",
      "Oxides: hybrid-DFT surface calculations (median over surfaces), 0.8 × surface + 0.2 × estimate",
      "Electronegativity estimate (Butler–Ginley): VBM = −χ − Eg/2",
      "CBM = VBM + Eg when no electron affinity is measured",
    ],
  },
  {
    title: "3 · Junction",
    what: "Band offsets and type at each interface.",
    order: [
      "Measured interface offset where one exists, otherwise both layers aligned to vacuum",
      "Type I / II / III follows from the four band edges (physics rule, not learned)",
      "Confidence: 4,000 Monte Carlo draws with the validated error of every input; below 80% → check with DFT",
    ],
  },
];

const SOURCES: { name: string; gives: string; size: string; ref: string }[] = [
  { name: "Zhuo et al. 2018", gives: "Measured band gaps (compilation)", size: "6,354 reports", ref: "J. Phys. Chem. Lett. 9, 1668" },
  { name: "Borlido et al. 2019", gives: "Measured band gaps (curated benchmark)", size: "472 materials", ref: "J. Chem. Theory Comput. 15, 5069" },
  { name: "JARVIS-DFT", gives: "GGA (OptB88vdW) and TBmBJ gaps, stable polymorphs", size: "part of 67,895 formulas", ref: "Choudhary et al., npj Comput. Mater. 2020" },
  { name: "SNUMAT", gives: "PBE and HSE06 gaps of ICSD structures", size: "part of 67,895 formulas", ref: "Kim et al., Sci. Data 2020" },
  { name: "Kiyohara, Hinuma & Oba 2024", gives: "Hybrid-DFT ionization potentials of oxide surfaces", size: "2,912 surfaces, 322 oxides", ref: "J. Am. Chem. Soc. 2024" },
  { name: "Photoemission papers", gives: "Measured IE / EA of device layers, measured interface offsets", size: "43 materials, 2 interfaces", ref: "cited per row in data/curated/" },
  { name: "InterMat (Table 2)", gives: "Measured band offsets, validation only", size: "21 interfaces", ref: "peer-reviewed compilation" },
];

export default function Method() {
  return (
    <div className="stack-lg">
      <header className="page-head">
        <h1>Method</h1>
        <p className="lede">
          Measured values are used whenever they exist; one ML model fills in the band gap when they do not. Each
          step falls back to the next rule only when the one above has no data.
        </p>
      </header>

      <div className="step-grid">
        {STEPS.map((s) => (
          <section className="card step" key={s.title}>
            <h2 className="card-title">{s.title}</h2>
            <p className="muted small">{s.what}</p>
            <ol className="fallback">
              {s.order.map((o) => (
                <li key={o}>{o}</li>
              ))}
            </ol>
          </section>
        ))}
      </div>

      <section className="card">
        <h2 className="card-title">Data sources</h2>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Source</th>
                <th>What it gives</th>
                <th>Size</th>
                <th>Reference</th>
              </tr>
            </thead>
            <tbody>
              {SOURCES.map((s) => (
                <tr key={s.name}>
                  <td className="layer-name">{s.name}</td>
                  <td>{s.gives}</td>
                  <td className="nowrap">{s.size}</td>
                  <td className="muted">{s.ref}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
