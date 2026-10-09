/**
 * Exhibit cells in the paper's conventions: three significant digits, a real minus sign, tabular
 * numbers (the CSS), "to" between interval limits so a negative limit never reads as a range.
 */
import { fmtInt, fmtNum } from "../stage/format";
import type { Cell } from "./types";

export const DASH = "—";

export const fmtInterval = (lo: number | null, hi: number | null): string =>
  lo === null || hi === null ? DASH : `${fmtNum(lo)} to ${fmtNum(hi)}`;

/** An estimate and its interval: "−0.0199 (−0.0327 to −0.00718)"; without one, the estimate. */
export function fmtEstimate(est: number | null, lo: number | null, hi: number | null): string {
  if (est === null) return DASH;
  return lo === null || hi === null ? fmtNum(est) : `${fmtNum(est)} (${fmtInterval(lo, hi)})`;
}

/** A p-value as journals print it: below 0.001 as "<0.001", otherwise two significant digits. */
export function fmtP(p: number | null): string {
  if (p === null || !Number.isFinite(p)) return DASH;
  if (p < 0.001) return "<0.001";
  if (p >= 0.995) return "1.0";
  return String(+p.toPrecision(2));
}

/** A percent with one decimal: 51.2. */
export const fmtPct = (pct: number | null): string => (pct === null ? DASH : pct.toFixed(1));

export function fmtCell(c: Cell | null | undefined): string {
  if (!c) return "";
  switch (c.kind) {
    case "estimate":
      return fmtEstimate(c.est, c.lo, c.hi);
    case "mean_sd":
      return c.mean === null ? DASH : `${fmtNum(c.mean)} (${fmtNum(c.sd)})`;
    case "median_iqr":
      return c.median === null ? DASH : `${fmtNum(c.median)} (${fmtNum(c.q1)}–${fmtNum(c.q3)})`;
    case "count_pct":
      return `${fmtInt(c.n)} (${fmtPct(c.pct)})`;
    case "count":
      return fmtInt(c.n);
    case "p":
      return fmtP(c.p);
    case "text":
      return c.text;
  }
}

/** Whether a cell is a number column's (right-aligned on its digits) or text (left). */
export const isNumeric = (c: Cell | null | undefined): boolean => !!c && c.kind !== "text" && c.kind !== "estimate";
