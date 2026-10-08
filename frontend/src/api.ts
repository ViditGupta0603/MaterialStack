/** Types and calls for the MaterialStack API (materialstack/api.py). */

export type LayerResult = {
  query: string;
  formula: string | null;
  /** formula as a chemist writes it (alias target, e.g. CH3NH3PbI3) */
  display_formula: string | null;
  gap_ev: number | null;
  gap_kind: "measured" | "ML" | "metal" | "none";
  gap_source: string;
  vbm_ev: number | null;
  cbm_ev: number | null;
  edge_kind: "measured" | "surface" | "estimate" | "none";
  edge_source: string;
  notes: string[];
};

export type JunctionResult = {
  interface: string;
  type: "I" | "II" | "III" | null;
  vbo_ev?: number;
  cbo_ev?: number;
  vbo_sigma_ev?: number;
  cbo_sigma_ev?: number;
  offset_source?: string;
  type_probabilities?: Record<"I" | "II" | "III", number>;
  confidence?: number;
  /** true when the most likely type has < 80% probability: check with DFT */
  uncertain: boolean;
  reason: string;
};

export type PredictResponse = {
  layers: LayerResult[];
  junctions: JunctionResult[];
};

/** One row of results/metrics.csv, written by validate.py. */
export type Metric = {
  section: string;
  metric: string;
  value: number | null;
  unit: string;
  n: number | null;
  baseline: number | null;
  baseline_rule: string;
  note: string;
};

export type MetricsResponse = {
  generated: string;
  metrics: Metric[];
};

/** The server's error message in plain text (FastAPI sends {"detail": ...}). */
async function errorText(res: Response): Promise<string> {
  const text = await res.text();
  try {
    const detail = JSON.parse(text).detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) return detail.map((d) => d.msg).join("; ");
  } catch {
    /* not JSON */
  }
  return text || `server error ${res.status}`;
}

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(await errorText(res));
  return res.json();
}

export function fetchMetrics() {
  return getJson<MetricsResponse>("/api/metrics");
}

export async function predictStack(materials: string[]) {
  const res = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ materials }),
  });
  if (!res.ok) throw new Error(await errorText(res));
  return res.json() as Promise<PredictResponse>;
}
