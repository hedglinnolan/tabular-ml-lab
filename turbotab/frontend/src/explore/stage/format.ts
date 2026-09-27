/** Number printing rules for the stage. One place, so every view agrees. */

export const fmtInt = (n: number) => Math.round(n).toLocaleString("en-US");

/**
 * A correlation. The residual method leaves r ≈ -1.9e-17 on the fitting rows; that is
 * zero to the printed precision, so it prints as 0.00 (never "-0.00", never 1.9e-17).
 */
export function fmtR(r: number | null): string {
  if (r === null || Number.isNaN(r)) return "—";
  if (Math.abs(r) < 0.005) return "0.00";
  return r.toFixed(2);
}

/** Axis ticks and cell values: compact, tabular, no float noise. */
export function fmtTick(v: number): string {
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a >= 1000) return Math.round(v).toLocaleString("en-US");
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return +v.toFixed(1) + "";
  if (a >= 1) return +v.toFixed(2) + "";
  if (a >= 0.01) return +v.toFixed(3) + "";
  return +v.toPrecision(2) + "";
}

/** A table cell: integers stay integers; everything else gets the digits it needs. */
export function cellFormatter(target: unknown): (v: number) => string {
  if (typeof target === "number" && Number.isInteger(target)) {
    return (v) => Math.round(v).toLocaleString("en-US");
  }
  if (typeof target === "number") {
    const a = Math.abs(target);
    const digits = a >= 100 ? 1 : a >= 1 ? 2 : 4;
    return (v) => v.toFixed(digits);
  }
  return (v) => String(v);
}

/** NHANES stores exact zeros as 5.4e-79 (a SAS stand-in); treat anything that small as 0. */
export const clean = (v: number) => (Math.abs(v) < 1e-12 ? 0 : v);

export const words = (s: string) => s.split(/\s+/).filter(Boolean).length;
