import type { Metric } from "./api";

/** Layers from typed text or a shared link: one per line or comma-separated, but a comma inside brackets
 *  belongs to the name (Zn(O,S) is one layer). */
export function splitLayers(text: string): string[] {
  return text
    .split(/\n|,(?![^()]*\))/)
    .map((s) => s.trim())
    .filter(Boolean);
}

export function fmt(n: number | null | undefined, digits = 2) {
  if (n == null || Number.isNaN(n)) return "—";
  return n.toFixed(digits);
}

export function pct(x: number | null | undefined) {
  return x == null ? "—" : `${Math.round(x * 100)}%`;
}

/** A metric from results/metrics.json by its exact name. */
export function findMetric(m: Metric[] | undefined, name: string): Metric | undefined {
  return m?.find((x) => x.metric === name);
}

/** A metric value with its unit, e.g. "0.40 eV" or "82%". */
export function showMetric(x: Metric | undefined) {
  if (!x || x.value == null) return "—";
  return x.unit === "%" ? `${Math.round(x.value)}%` : `${fmt(x.value)}${x.unit ? ` ${x.unit}` : ""}`;
}
