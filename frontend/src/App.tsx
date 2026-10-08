import { useEffect, useRef, useState } from "react";
import { fetchMetrics, predictStack } from "./api";
import type { MetricsResponse, PredictResponse } from "./api";
import Guide from "./pages/Guide";
import Method from "./pages/Method";
import Predict from "./pages/Predict";
import Validation from "./pages/Validation";
import { splitLayers } from "./format";
import "./App.css";

const PAGES = [
  { id: "predict", label: "Predict" },
  { id: "validation", label: "Validation" },
  { id: "method", label: "Method" },
  { id: "guide", label: "Guide" },
] as const;
type PageId = (typeof PAGES)[number]["id"];
const DEFAULT_STACK = ["TiO2", "MAPbI3", "Spiro-OMeTAD"];
const MAX_LAYERS = 20; // same limit as the server (materialstack/api.py)

/** The page named in the URL hash (#/validation …); Predict by default. */
function pageFromHash(): PageId {
  const id = window.location.hash.replace(/^#\/?/, "").split("?")[0];
  return PAGES.find((p) => p.id === id)?.id ?? "predict";
}

/** The stack saved in a shared link (#/predict?stack=TiO2,MAPbI3), else the default stack. */
function stackFromHash(): string[] {
  const query = window.location.hash.split("?")[1] ?? "";
  const stack = new URLSearchParams(query).get("stack");
  const layers = stack ? splitLayers(stack) : [];
  return layers.length ? layers : DEFAULT_STACK;
}

export default function App() {
  const [page, setPage] = useState<PageId>(pageFromHash);
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [input, setInput] = useState(() => stackFromHash().join("\n"));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const started = useRef(false);

  async function run(materials: string[]) {
    if (!materials.length) {
      setError("Enter at least one layer, one per line.");
      return;
    }
    if (materials.length > MAX_LAYERS) {
      setError(`A stack can have at most ${MAX_LAYERS} layers.`);
      return;
    }
    setBusy(true);
    setError(null);
    try {
      setResult(await predictStack(materials));
      // keep the stack in the address bar, so the result can be bookmarked or shared
      if (pageFromHash() === "predict") {
        window.history.replaceState(null, "", `#/predict?stack=${encodeURIComponent(materials.join(","))}`);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    const onHash = () => {
      setPage(pageFromHash());
      window.scrollTo(0, 0);
    };
    window.addEventListener("hashchange", onHash);
    fetchMetrics().then(setMetrics).catch(() => setMetrics(null));
    if (!started.current) {
      started.current = true; // predict the default stack once, so the page is never empty
      run(stackFromHash());
    }
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  return (
    <div className="app">
      <header className="topbar">
        <a className="brand" href="#/predict">
          <span className="brand-mark" aria-hidden="true" />
          <span className="brand-name">MaterialStack</span>
          <span className="brand-tag">Band-alignment screening for device stacks</span>
        </a>
        <nav className="tabs" aria-label="Pages">
          {PAGES.map((p) => (
            <a key={p.id} href={`#/${p.id}`} className={page === p.id ? "tab active" : "tab"} aria-current={page === p.id ? "page" : undefined}>
              {p.label}
            </a>
          ))}
        </nav>
      </header>

      <main className="page">
        {page === "predict" && (
          <Predict input={input} setInput={setInput} run={run} busy={busy} error={error} result={result} />
        )}
        {page === "validation" && <Validation metrics={metrics} />}
        {page === "method" && <Method />}
        {page === "guide" && <Guide metrics={metrics} />}
      </main>
    </div>
  );
}
