/** Number formatting for previews. Data speaks mono (§03); these only shape the digits. */

export const fmtInt = (n: number) => Math.round(n).toLocaleString("en-US");

/**
 * A correlation, to two decimals. Residuals are uncorrelated with energy by construction, and
 * floating point returns values like -1.9e-17: anything that rounds to zero prints as 0.00,
 * never as -0.00.
 */
export function fmtR(r: number | null | undefined): string {
  if (r === null || r === undefined || Number.isNaN(r)) return "—";
  if (Math.abs(r) < 0.005) return "0.00";
  return r.toFixed(2);
}

/** A value in a working-table cell: at most four significant digits, grouped thousands. */
export function fmtCell(v: unknown): string {
  if (v === null || v === undefined) return "·";
  if (typeof v !== "number") return String(v);
  if (Number.isInteger(v)) return v.toLocaleString("en-US");
  const abs = Math.abs(v);
  if (abs === 0) return "0";
  if (abs >= 1000) return Math.round(v).toLocaleString("en-US");
  if (abs >= 100) return v.toFixed(0);
  if (abs >= 10) return v.toFixed(1);
  if (abs >= 1) return v.toFixed(2);
  return v.toPrecision(2);
}

/** Axis ticks: short, no trailing zeros. */
export function fmtTick(v: number): string {
  const abs = Math.abs(v);
  if (abs === 0) return "0";
  if (abs >= 1000) return `${+(v / 1000).toFixed(1)}k`;
  if (abs >= 1) return String(+v.toFixed(1));
  return String(+v.toPrecision(2));
}

export function pct(part: number, whole: number): string {
  if (whole === 0) return "0%";
  const p = (100 * part) / whole;
  return `${p < 10 ? p.toFixed(1) : Math.round(p)}%`;
}

export function words(text: string): number {
  return text.split(/\s+/).filter(Boolean).length;
}
