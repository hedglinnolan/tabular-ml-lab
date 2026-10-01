/** Number printing rules for the stage. One place, so every view (and every saved figure) agrees. */

export const fmtInt = (n: number) => Math.round(n).toLocaleString("en-US");

/**
 * A correlation. The residual method leaves r ≈ −1.9e−17 on the fitting rows; that is zero to the
 * printed precision, so it prints as 0.00 (never "−0.00", never 1.9e−17).
 */
export function fmtR(r: number | null | undefined): string {
  if (r === null || r === undefined || Number.isNaN(r)) return "—";
  if (Math.abs(r) < 0.005) return "0.00";
  return r.toFixed(2).replace("-", "−");
}

/** Axis ticks and cell values: compact, tabular, no float noise, a real minus sign. */
export function fmtTick(v: number): string {
  const a = Math.abs(v);
  let s: string;
  if (a === 0) return "0";
  if (a >= 1000) s = Math.round(v).toLocaleString("en-US");
  else if (a >= 100) s = v.toFixed(0);
  else if (a >= 10) s = String(+v.toFixed(1));
  else if (a >= 1) s = String(+v.toFixed(2));
  else if (a >= 0.01) s = String(+v.toFixed(3));
  else s = String(+v.toPrecision(2));
  return s.replace("-", "−");
}

/** A metric or an estimate: three significant digits, a real minus sign. */
export function fmtNum(v: number | null | undefined, digits = 3): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  if (v === 0) return "0";
  const a = Math.abs(v);
  const s = a >= 1000 ? Math.round(v).toLocaleString("en-US") : String(+v.toPrecision(digits));
  return s.replace("-", "−");
}

/** A signed change: +0.866, −1.71. */
export function fmtSigned(v: number | null | undefined, digits = 3): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  const s = fmtNum(Math.abs(v), digits);
  return v > 0 ? `+${s}` : v < 0 ? `−${s}` : s;
}

/** A table cell: integers stay integers; everything else gets the digits it needs. */
export function cellFormatter(target: unknown): (v: number) => string {
  if (typeof target === "number" && Number.isInteger(target)) {
    return (v) => Math.round(v).toLocaleString("en-US");
  }
  if (typeof target === "number") {
    const a = Math.abs(target);
    const digits = a >= 100 ? 1 : a >= 1 ? 2 : 4;
    return (v) => v.toFixed(digits).replace("-", "−");
  }
  return (v) => String(v);
}

/** NHANES stores exact zeros as 5.4e−79 (a SAS stand-in); treat anything that small as 0. */
export const clean = (v: number) => (Math.abs(v) < 1e-12 ? 0 : v);

export const words = (s: string) => s.split(/\s+/).filter(Boolean).length;

/** Strip the backticks that mark data in the app's prose (for plain-text places: aria, files). */
export const plain = (s: string) => s.replace(/`/g, "");
