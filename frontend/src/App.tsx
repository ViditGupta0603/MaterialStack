import { motion } from "framer-motion";
import { useEffect, useMemo, useState } from "react";
import type { FormEvent } from "react";
import { fetchMetrics, fetchStats, predictStack } from "./api";
import type {
  GapMetrics,
  MetricsResponse,
  PolymorphPolicy,
  PredictResponse,
  StatsResponse,
} from "./api";
import "./App.css";

const ease = [0.22, 1, 0.36, 1] as const;

function fmt(n: number | null | undefined, digits = 3) {
  if (n == null || Number.isNaN(n)) return "—";
  return n.toFixed(digits);
}

export default function App() {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [input, setInput] = useState("TiO2\nMAPbI3");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [polymorph, setPolymorph] = useState<PolymorphPolicy>("ground_state");

  useEffect(() => {
    fetchMetrics().then(setMetrics).catch(() => setMetrics(null));
    fetchStats().then(setStats).catch(() => setStats(null));
  }, []);

  async function onPredict(e: FormEvent) {
    e.preventDefault();
    const materials = input
      .split(/[\n,]+/)
      .map((s) => s.trim())
      .filter(Boolean);
    if (!materials.length) return;
    setBusy(true);
    setError(null);
    try {
      setResult(await predictStack(materials, polymorph));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  const familyBars = useMemo(() => {
    if (!stats?.by_family?.length) return [];
    const max = Math.max(...stats.by_family.map((f) => f.n_materials));
    return stats.by_family.map((f) => ({
      ...f,
      pct: (100 * f.n_materials) / max,
    }));
  }, [stats]);

  return (
    <div className="page">
      <header className="nav">
        <a className="nav-brand" href="#top">
          MaterialStack
        </a>
        <nav className="nav-links">
          <a href="#lab">Lab</a>
          <a href="#report">Performance</a>
          <a href="#pipeline">Pipeline</a>
        </nav>
      </header>

      <section className="hero" id="top">
        <div className="hero-copy">
          <motion.p
            className="brand"
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, ease }}
          >
            MaterialStack
          </motion.p>
          <motion.h1
            initial={{ opacity: 0, y: 22 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.75, delay: 0.08, ease }}
          >
            Band edges for layered stacks.
          </motion.h1>
          <motion.p
            className="lede"
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.16, ease }}
          >
            Lookup trusted gaps first. Predict the rest. Classify junctions from
            physics — not a black-box label.
          </motion.p>
          <motion.div
            className="hero-cta"
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.65, delay: 0.24, ease }}
          >
            <a className="btn primary" href="#lab">
              Open the lab
            </a>
            <a className="btn ghost" href="#report">
              View training report
            </a>
          </motion.div>
        </div>
        <motion.div
          className="hero-visual"
          aria-hidden
          initial={{ opacity: 0, scale: 1.04 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 1.1, ease }}
        >
          <BandStackArt />
        </motion.div>
      </section>

      <section className="section lab" id="lab">
        <div className="section-head">
          <p className="eyebrow">Lab</p>
          <h2>Resolve a layer or a stack</h2>
          <p>
            Enter one formula per line. MaterialStack prefers experiment / HSE
            from the database, then Magpie + LightGBM.
          </p>
        </div>
        <form className="lab-form" onSubmit={onPredict}>
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            rows={5}
            placeholder={"TiO2\nMAPbI3\nNiO"}
            spellCheck={false}
          />
          <div className="lab-actions">
            <button className="btn primary" type="submit" disabled={busy}>
              {busy ? "Resolving…" : "Predict stack"}
            </button>
            <div className="seg" role="radiogroup" aria-label="Polymorph policy">
              <span className="seg-label">Polymorph</span>
              {(
                [
                  ["ground_state", "Ground state"],
                  ["mean", "Mean of polymorphs"],
                ] as const
              ).map(([value, label]) => (
                <button
                  key={value}
                  type="button"
                  role="radio"
                  aria-checked={polymorph === value}
                  className={polymorph === value ? "seg-btn active" : "seg-btn"}
                  onClick={() => setPolymorph(value)}
                >
                  {label}
                </button>
              ))}
            </div>
            {error && <p className="error">{error}</p>}
          </div>
        </form>

        {result && (
          <motion.div
            className="results"
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, ease }}
          >
            <div className="layer-grid">
              {result.layers.map((L, i) => (
                <motion.article
                  key={`${L.query}-${i}`}
                  className="layer"
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.06 * i, duration: 0.45, ease }}
                >
                  <header>
                    <h3>{L.formula ?? L.query}</h3>
                    <span className={L.trusted_gap ? "tag ok" : "tag warn"}>
                      {L.trusted_gap ? "trusted gap" : "ML / weak"}
                    </span>
                  </header>
                  <dl>
                    <div>
                      <dt>Eg</dt>
                      <dd>{fmt(L.gap_ev)} eV</dd>
                    </div>
                    <div>
                      <dt>CBM</dt>
                      <dd>{fmt(L.cbm_ev)} eV</dd>
                    </div>
                    <div>
                      <dt>VBM</dt>
                      <dd>{fmt(L.vbm_ev)} eV</dd>
                    </div>
                  </dl>
                  <p className="source">{L.gap_source}</p>
                  <p className="source muted">{L.edge_source}</p>
                  {L.structures_used.length > 0 && (
                    <ul className="chips">
                      {L.structures_used.map((s) => (
                        <li key={s.structure_id}>
                          SG {s.space_group ?? "?"} · {fmt(s.gap_ev, 2)} eV
                          {s.e_rel != null && ` · ΔE ${fmt(s.e_rel, 3)} eV/atom`}
                        </li>
                      ))}
                    </ul>
                  )}
                  {L.notes?.map((n) => (
                    <p className="note" key={n}>
                      {n}
                    </p>
                  ))}
                </motion.article>
              ))}
            </div>
            {result.junctions.length > 0 && (
              <div className="junctions">
                <h3>Junctions</h3>
                {result.junctions.map((j) => (
                  <div className="junction" key={j.interface}>
                    <div>
                      <strong>{j.interface}</strong>
                      <span className={j.uncertain ? "tag warn" : "tag ok"}>
                        Type {j.type ?? "?"}
                        {j.uncertain ? " · uncertain" : ""}
                      </span>
                    </div>
                    <p>
                      CBO {fmt(j.cbo_ev)} eV · VBO {fmt(j.vbo_ev)} eV ·{" "}
                      {j.reason}
                    </p>
                  </div>
                ))}
              </div>
            )}
          </motion.div>
        )}
      </section>

      <section className="section report" id="report">
        <div className="section-head">
          <p className="eyebrow">Performance</p>
          <h2>Full training report</h2>
          <p>
            Frozen LightGBM models on cleaned experiment / literature labels.
            GroupKFold by element set. Models unchanged for this UI release.
          </p>
        </div>

        {!metrics ? (
          <p className="muted">Loading metrics…</p>
        ) : (
          <>
            <div className="stat-row">
              <div className="stat">
                <span>Eg MAE (non-metal)</span>
                <strong>{fmt(metrics.eg.mae_nonmetal)} eV</strong>
              </div>
              <div className="stat">
                <span>Eg R²</span>
                <strong>{fmt(metrics.eg.r2_nonmetal)}</strong>
              </div>
              <div className="stat">
                <span>Metal accuracy</span>
                <strong>{fmt(metrics.eg.metal_accuracy * 100, 1)}%</strong>
              </div>
              <div className="stat">
                <span>Edge CBM MAE</span>
                <strong>{fmt(metrics.edge.cbm_mae)} eV</strong>
              </div>
            </div>

            <div className="report-grid">
              <article className="panel">
                <h3>Model A — band gap</h3>
                <ul className="kv">
                  <li>
                    <span>Backend</span>
                    <b>{metrics.primary_backend}</b>
                  </li>
                  <li>
                    <span>Train methods</span>
                    <b>{metrics.eg_train_methods.join(", ")}</b>
                  </li>
                  <li>
                    <span>Train materials</span>
                    <b>
                      {metrics.eg.n_train} ({metrics.eg.n_metals} metal /{" "}
                      {metrics.eg.n_nonmetals} non-metal)
                    </b>
                  </li>
                  <li>
                    <span>RMSE / R²</span>
                    <b>
                      {fmt(metrics.eg.rmse_nonmetal)} eV ·{" "}
                      {fmt(metrics.eg.r2_nonmetal)}
                    </b>
                  </li>
                  <li>
                    <span>XGB baseline MAE</span>
                    <b>{fmt(metrics.eg_xgb_baseline.mae_nonmetal)} eV</b>
                  </li>
                  <li>
                    <span>CV folds</span>
                    <b>GroupKFold × {metrics.eg.n_splits}</b>
                  </li>
                  <li>
                    <span>Train wall</span>
                    <b>{metrics.seconds}s</b>
                  </li>
                </ul>
                {metrics.borlido_holdout && (
                  <p className="callout">
                    Borlido-only holdout (n={metrics.borlido_holdout.n}): MAE{" "}
                    {fmt(metrics.borlido_holdout.mae_nonmetal)} eV — OOD stress
                    test, never used in fit.
                  </p>
                )}
              </article>

              <article className="panel">
                <h3>Model B — vacuum edges</h3>
                <ul className="kv">
                  <li>
                    <span>δCBM / CBM / VBM MAE</span>
                    <b>
                      {fmt(metrics.edge.delta_mae)} / {fmt(metrics.edge.cbm_mae)}{" "}
                      / {fmt(metrics.edge.vbm_mae)} eV
                    </b>
                  </li>
                  <li>
                    <span>Train materials</span>
                    <b>{metrics.edge.n_train}</b>
                  </li>
                  <li>
                    <span>XGB edge CBM MAE</span>
                    <b>{fmt(metrics.edge_xgb_baseline.cbm_mae)} eV</b>
                  </li>
                </ul>
                {metrics.edge.sources && (
                  <ul className="chips">
                    {Object.entries(metrics.edge.sources).map(([k, v]) => (
                      <li key={k}>
                        {k}: {v}
                      </li>
                    ))}
                  </ul>
                )}
                <p className="formula">
                  CBM<sub>BG</sub> = −(χ − E<sub>g</sub>/2) · δCBM = CBM −
                  CBM<sub>BG</sub> · VBM = CBM − E<sub>g</sub>
                </p>
              </article>
            </div>

            {metrics.structure_hybrid && (
              <StructurePanel h={metrics.structure_hybrid} />
            )}

            {metrics.clean_audit && (
              <article className="panel wide">
                <h3>Label cleaning audit</h3>
                <p className="muted">
                  Eg {metrics.clean_audit.n_eg_before} →{" "}
                  {metrics.clean_audit.n_eg_after} · edges{" "}
                  {metrics.clean_audit.n_edge_before} →{" "}
                  {metrics.clean_audit.n_edge_after} (SQLite unchanged)
                </p>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Rule</th>
                        <th>Dropped</th>
                        <th>Detail</th>
                      </tr>
                    </thead>
                    <tbody>
                      {metrics.clean_audit.rules
                        .filter((r) => r.n_dropped > 0)
                        .map((r) => (
                          <tr key={r.rule}>
                            <td>{r.rule}</td>
                            <td>{r.n_dropped}</td>
                            <td>{r.detail}</td>
                          </tr>
                        ))}
                    </tbody>
                  </table>
                </div>
              </article>
            )}

            <div className="report-grid">
              <article className="panel">
                <h3>Top Eg features</h3>
                <ol className="feat-list">
                  {metrics.eg.top_features.map((f) => (
                    <li key={f.feature}>
                      <span>{f.feature.replace("MagpieData ", "")}</span>
                      <b>{Math.round(f.importance)}</b>
                    </li>
                  ))}
                </ol>
              </article>
              <article className="panel">
                <h3>Top edge features</h3>
                <ol className="feat-list">
                  {metrics.edge.top_features.map((f) => (
                    <li key={f.feature}>
                      <span>{f.feature.replace("MagpieData ", "")}</span>
                      <b>{Math.round(f.importance)}</b>
                    </li>
                  ))}
                </ol>
              </article>
            </div>

            <p className="decision">{metrics.decision}</p>
          </>
        )}
      </section>

      <section className="section pipeline" id="pipeline">
        <div className="section-head">
          <p className="eyebrow">Pipeline</p>
          <h2>From catalog to junction</h2>
        </div>
        <ol className="steps">
          <li>
            <strong>Database</strong>
            <span>
              {stats
                ? `${stats.materials.toLocaleString()} materials · ${stats.records.toLocaleString()} records`
                : "104k+ materials"}
            </span>
          </li>
          <li>
            <strong>Clean labels</strong>
            <span>Experiment-first · scatter / DFT conflict filters</span>
          </li>
          <li>
            <strong>LightGBM</strong>
            <span>Metal classifier + log1p Eg · δCBM edge correction</span>
          </li>
          <li>
            <strong>Lookup-first</strong>
            <span>Trusted DB before ML · Type I/II/III from offsets</span>
          </li>
        </ol>
        {familyBars.length > 0 && (
          <div className="family">
            <h3>Catalog by family</h3>
            <ul>
              {familyBars.map((f) => (
                <li key={f.family}>
                  <span>{f.family}</span>
                  <div className="bar">
                    <i style={{ width: `${f.pct}%` }} />
                  </div>
                  <b>{f.n_materials.toLocaleString()}</b>
                </li>
              ))}
            </ul>
          </div>
        )}
      </section>

      <footer className="footer">
        <strong>MaterialStack</strong>
        <span>Models frozen for this build</span>
      </footer>
    </div>
  );
}

function pct(n: number | null | undefined) {
  return n == null ? "—" : `${fmt(n * 100, 1)}%`;
}

function StructurePanel({ h }: { h: NonNullable<MetricsResponse["structure_hybrid"]> }) {
  const sys: [string, GapMetrics][] = [
    ["Composition only (previous)", h.system_composition_only],
    ["Structure-aware (deployed)", h.system_structure_aware],
  ];
  return (
    <article className="panel wide">
      <h3>Structure-aware band gap model</h3>
      <p className="muted">
        Bulk crystal structure for {h.n_with_structure} training materials ({pct(h.coverage)}),
        stored DFT gap for {h.n_with_dft} ({pct(h.coverage_dft)}) ·{" "}
        {h.n_features.composition} composition + {h.n_features.structure} structure +{" "}
        {h.n_features.dft_proxy} DFT-proxy + {h.n_features.stored_dft} stored-DFT features ·
        default polymorph: {h.recommended_policy.replace("_", " ")}
      </p>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Ablation · identical folds</th>
              <th>MAE (non-metal)</th>
              <th>R²</th>
              <th>Metal acc.</th>
              <th>MAE · with structure</th>
            </tr>
          </thead>
          <tbody>
            {h.ablation.map((r) => (
              <tr key={r.label}>
                <td>{r.label}</td>
                <td>{fmt(r.all.mae_nonmetal)} eV</td>
                <td>{fmt(r.all.r2_nonmetal)}</td>
                <td>{pct(r.all.metal_accuracy)}</td>
                <td>{fmt(r.with_structure.mae_nonmetal)} eV</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>System as deployed</th>
              <th>MAE (non-metal)</th>
              <th>R²</th>
              <th>≤ 0.5 eV</th>
            </tr>
          </thead>
          <tbody>
            {sys.map(([label, m]) => (
              <tr key={label}>
                <td>{label}</td>
                <td>{fmt(m.mae_nonmetal)} eV</td>
                <td>{fmt(m.r2_nonmetal)}</td>
                <td>{pct(m.within_0p5_all)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {h.borlido_holdout && (
        <p className="callout">
          Borlido holdout (never trained on): MAE{" "}
          {fmt(h.borlido_holdout.structure_aware.mae_nonmetal)} eV structure-aware vs{" "}
          {fmt(h.borlido_holdout.composition_only.mae_nonmetal)} eV composition-only.
        </p>
      )}
    </article>
  );
}

function BandStackArt() {
  return (
    <svg className="band-art" viewBox="0 0 720 560" role="img">
      <defs>
        <linearGradient id="sky" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#c5d5de" />
          <stop offset="55%" stopColor="#9eb8c4" />
          <stop offset="100%" stopColor="#6f9399" />
        </linearGradient>
        <linearGradient id="layerA" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#1c6b66" stopOpacity="0.92" />
          <stop offset="100%" stopColor="#0f4542" stopOpacity="0.95" />
        </linearGradient>
        <linearGradient id="layerB" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#2a7f79" stopOpacity="0.85" />
          <stop offset="100%" stopColor="#1a5551" stopOpacity="0.9" />
        </linearGradient>
        <linearGradient id="layerC" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#c46a3a" stopOpacity="0.88" />
          <stop offset="100%" stopColor="#8f4726" stopOpacity="0.92" />
        </linearGradient>
      </defs>
      <rect width="720" height="560" fill="url(#sky)" />
      <motion.g
        initial={{ y: 24, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 1, delay: 0.2, ease }}
      >
        <path d="M40 420 L680 360 L680 500 L40 500 Z" fill="url(#layerA)" />
        <path d="M70 300 L650 250 L650 360 L70 390 Z" fill="url(#layerB)" />
        <path d="M110 190 L610 150 L610 250 L110 285 Z" fill="url(#layerC)" />
        <motion.path
          d="M130 210 C250 180 400 200 590 165"
          fill="none"
          stroke="#f3f6f8"
          strokeWidth="2.5"
          strokeLinecap="round"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 1.4, delay: 0.55, ease }}
        />
        <motion.path
          d="M90 330 C220 300 420 320 630 275"
          fill="none"
          stroke="#f3f6f8"
          strokeWidth="2"
          strokeOpacity="0.75"
          strokeLinecap="round"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 1.4, delay: 0.75, ease }}
        />
        <text x="130" y="175" fill="#f3f6f8" fontFamily="Syne, sans-serif" fontSize="18" fontWeight="700">
          CBM
        </text>
        <text x="90" y="445" fill="#f3f6f8" fontFamily="Syne, sans-serif" fontSize="18" fontWeight="700">
          VBM
        </text>
      </motion.g>
    </svg>
  );
}
