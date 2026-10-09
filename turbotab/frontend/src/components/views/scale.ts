/**
 * The forest's one scale. Drawn to scale on one axis: the domain is the data's extent with the
 * reference always inside it, padded so no marker meets the edge; ticks name only values inside
 * that extent (never a round number out in the padding), and the reference is always a tick.
 * A ratio sits on a log axis, so 0.5 and 2 lie the same distance from 1.
 */
import { scaleLinear, scaleLog } from "d3-scale";

export interface Interval {
  est: number | null;
  lo: number | null;
  hi: number | null;
}

export interface ForestScale {
  x: (v: number) => number;
  /** The padded domain the range maps. */
  domain: [number, number];
  /** What the data and the reference reach. */
  extent: [number, number];
  ticks: number[];
}

export interface ScaleSpec {
  axis: "linear" | "log";
  reference: number;
  /** The plot's width in px; the range is [inset, width − inset]. */
  width: number;
  /** Room at either end for a marker and a tick label's half width. */
  inset?: number;
}

/** Ticks closer than this to the reference (in px) are dropped so its label stands alone. */
const MIN_GAP = 30;
/** About one tick label per this many px. */
const PER_TICK = 84;
/** A share of the extent added at each end. */
const PAD = 0.06;

export function valuesOf(rows: Interval[]): number[] {
  return rows.flatMap((r) => [r.est, r.lo, r.hi]).filter((v): v is number => v !== null && Number.isFinite(v));
}

/** null when nothing can be drawn: no values, or a value at or below zero on a log axis. */
export function forestScale(rows: Interval[], spec: ScaleSpec): ForestScale | null {
  const values = valuesOf(rows);
  if (!values.length) return null;
  const { axis, reference, width } = spec;
  const inset = spec.inset ?? 14;
  const all = [...values, reference];
  if (axis === "log" && all.some((v) => v <= 0)) return null;
  const lo = Math.min(...all);
  const hi = Math.max(...all);
  const range: [number, number] = [inset, Math.max(inset + 1, width - inset)];
  const count = Math.max(2, Math.floor((range[1] - range[0]) / PER_TICK));

  if (axis === "linear") {
    const span = hi - lo;
    const pad = span > 0 ? span * PAD : Math.abs(lo) * 0.1 || 1;
    const domain: [number, number] = [lo - pad, hi + pad];
    const x = scaleLinear().domain(domain).range(range);
    const eps = (span || 1) * 1e-9;
    const raw = scaleLinear().domain([lo, hi]).ticks(count).filter((t) => t >= lo - eps && t <= hi + eps);
    return { x, domain, extent: [lo, hi], ticks: settle(raw, reference, x) };
  }

  const L0 = Math.log10(lo);
  const L1 = Math.log10(hi);
  const span = L1 - L0;
  const pad = span > 0 ? span * PAD : 0.05;
  const domain: [number, number] = [10 ** (L0 - pad), 10 ** (L1 + pad)];
  const x = scaleLog().domain(domain).range(range);
  return { x, domain, extent: [lo, hi], ticks: settle(logTicks(lo, hi, count), reference, x) };
}

/** Ratio ticks: round mantissas of each decade inside the extent, fewer when they crowd; a
 *  linear subdivision when the extent is too narrow to hold two of them. The densest set that fits
 *  the width wins: 1 to 9, then 1 2 3 5, then 1 2 5, then 1 3, then powers of ten. */
export function logTicks(lo: number, hi: number, count: number): number[] {
  const eps = 1e-9;
  const inside = (t: number) => t >= lo * (1 - eps) && t <= hi * (1 + eps);
  const decades: number[] = [];
  for (let d = Math.floor(Math.log10(lo)); d <= Math.ceil(Math.log10(hi)); d++) decades.push(d);
  const of = (mantissas: number[]) => decades.flatMap((d) => mantissas.map((m) => +(m * 10 ** d).toPrecision(6))).filter(inside);
  for (const m of [[1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 5], [1, 2, 5], [1, 3], [1]]) {
    const t = of(m);
    if (t.length <= count + 1 && t.length >= 2) return t;
  }
  const tens = of([1]);
  if (tens.length >= 2) return tens.filter((_, i) => i % Math.ceil(tens.length / (count + 1)) === 0);
  return scaleLinear().domain([lo, hi]).ticks(count).filter(inside);
}

/** Add the reference and drop any tick crowding it. */
function settle(raw: number[], reference: number, x: (v: number) => number): number[] {
  const kept = raw.filter((t) => t === reference || Math.abs(x(t) - x(reference)) >= MIN_GAP);
  return [...new Set([...kept, reference])].sort((a, b) => a - b);
}
