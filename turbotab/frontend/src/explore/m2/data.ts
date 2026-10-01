/**
 * The real-data fixture for /lab/m2, typed, plus the few number rules the prototype's views share.
 * It stands in for the preview responses M2's backend will serve; nothing here is typed by hand.
 */
import raw from "./fixture.json";
import type { Cell, Fixture } from "./types";

export const FX = raw as unknown as Fixture;

export const fmtInt = (n: number) => Math.round(n).toLocaleString("en-US");

/** A signed number with a real minus sign. */
export function fmtNum(v: number | null | undefined, digits = 3): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  if (v === 0) return "0";
  const a = Math.abs(v);
  const s = a >= 1000 ? Math.round(v).toLocaleString("en-US") : String(+v.toPrecision(digits));
  return s.replace("-", "−");
}

export const pct = (f: number) => `${Math.round(f * 100)}%`;

/** A metric (R², AUC) at a fixed three decimals, so a column of them aligns and compares. */
export function fmtMetric(v: number | null | undefined): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  return v.toFixed(3).replace("-", "−");
}

/** How many decimals a value needs (capped), so a column prints one way within a state. */
function decimalsOf(v: number, cap = 2): number {
  for (let d = 0; d < cap; d++) if (Math.abs(v * 10 ** d - Math.round(v * 10 ** d)) < 1e-6) return d;
  return cap;
}

/** One format per column per state: `5` and `5.39` never sit side by side as if different kinds. */
export function columnFormatter(values: Cell[], cap = 2): (v: Cell) => string {
  const nums = values.filter((v): v is number => typeof v === "number");
  const d = nums.reduce((m, v) => Math.max(m, decimalsOf(v, cap)), 0);
  return (v) => {
    if (v === null || v === undefined || v === "") return "blank";
    if (typeof v === "number")
      return v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d }).replace("-", "−");
    return String(v);
  };
}

/** A column name's shared prefix for a wide table ("ft_"), so headers read as the part that differs. */
export function sharedPrefix(cols: string[]): string {
  if (cols.length < 3) return "";
  let p = cols[0]!;
  for (const c of cols) while (!c.startsWith(p)) p = p.slice(0, -1);
  const cut = p.lastIndexOf("_");
  return cut >= 1 ? p.slice(0, cut + 1) : "";
}

export function words(s: string): number {
  return s.split(/\s+/).filter(Boolean).length;
}
