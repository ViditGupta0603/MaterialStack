export type LayerResult = {
  query: string;
  formula: string | null;
  material_id: number | null;
  gap_ev: number | null;
  cbm_ev: number | null;
  vbm_ev: number | null;
  gap_source: string;
  edge_source: string;
  gap_method: string | null;
  edge_method: string | null;
  p_metal: number | null;
  chi: number | null;
  trusted_gap: boolean;
  trusted_edges: boolean;
  notes: string[];
  polymorph_policy: PolymorphPolicy | null;
  structures_used: StructureUsed[];
};

export type PolymorphPolicy = "ground_state" | "mean";

export type StructureUsed = {
  structure_id: number;
  source: string;
  space_group: number | null;
  e_rel: number | null;
  gap_ev: number;
  p_metal: number;
};

export type GapMetrics = {
  n: number;
  n_nonmetals: number;
  mae_nonmetal: number | null;
  rmse_nonmetal: number | null;
  r2_nonmetal: number | null;
  mae_all: number;
  within_0p5_all: number;
  within_0p5_nonmetal: number | null;
  metal_accuracy: number;
};

export type JunctionResult = {
  interface: string;
  type: string | null;
  cbo_ev?: number;
  vbo_ev?: number;
  margin_ev?: number;
  uncertain: boolean;
  reason: string;
};

export type PredictResponse = {
  max_lookup_rank: number;
  polymorph: PolymorphPolicy;
  layers: LayerResult[];
  junctions: JunctionResult[];
};

export type MetricsResponse = {
  product: string;
  primary_backend: string;
  decision: string;
  eg_train_methods: string[];
  n_eg_train: number;
  seconds: number;
  eg: {
    mae_nonmetal: number;
    rmse_nonmetal: number;
    r2_nonmetal: number;
    mae_all: number;
    metal_accuracy: number;
    metal_f1: number;
    n_train: number;
    n_metals: number;
    n_nonmetals: number;
    n_splits: number;
    top_features: { feature: string; importance: number }[];
  };
  eg_xgb_baseline: { mae_nonmetal: number | null; metal_accuracy: number | null };
  edge: {
    cbm_mae: number;
    vbm_mae: number;
    delta_mae: number;
    n_train: number;
    sources: Record<string, number>;
    top_features: { feature: string; importance: number }[];
  };
  edge_xgb_baseline: { cbm_mae: number | null };
  borlido_holdout?: { n: number; mae_nonmetal: number | null; mae_all: number };
  structure_hybrid?: {
    recommended_policy: PolymorphPolicy;
    fallback_without_structure: "hybrid" | "composition";
    coverage: number;
    coverage_dft: number;
    n_with_structure: number;
    n_with_dft: number;
    n_features: { composition: number; structure: number; dft_proxy: number; stored_dft: number };
    ablation: { label: string; all: GapMetrics; with_structure: GapMetrics }[];
    system_composition_only: GapMetrics;
    system_structure_aware: GapMetrics;
    borlido_holdout?: { composition_only: GapMetrics; structure_aware: GapMetrics; n_known: number };
    top_features: { feature: string; importance: number }[];
  } | null;
  clean_audit?: {
    n_eg_before: number;
    n_eg_after: number;
    n_edge_before: number;
    n_edge_after: number;
    rules: { rule: string; n_dropped: number; detail: string }[];
  };
};

export type StatsResponse = {
  materials: number;
  records: number;
  structures: number;
  interfaces: number;
  aliases: number;
  by_family: { family: string; n_materials: number }[];
};

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export function fetchMetrics() {
  return getJson<MetricsResponse>("/api/metrics");
}

export function fetchStats() {
  return getJson<StatsResponse>("/api/stats");
}

export async function predictStack(
  materials: string[],
  polymorph: PolymorphPolicy | null = null,
  maxLookupRank = 2,
) {
  const res = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ materials, max_lookup_rank: maxLookupRank, polymorph }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<PredictResponse>;
}
