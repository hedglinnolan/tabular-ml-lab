/**
 * The forest's one scale. Drawn to scale on one axis: the domain is the data's extent with the
 * reference always inside it, padded so no marker meets the edge; ticks name only values inside
 * that extent (never a round number out in the padding), and the reference is always a tick.
 * A ratio sits on a log axis, so 0.5 and 2 lie the same distance from 1.
 */
import { scaleLinear, scaleLog, type ScaleLinear } from "d3-scale";

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

/**
 * The scale math every exhibit view shares, pure so it is tested with known pixels.
 *
 * Rules (FOUNDATION §5 rule 9; the dataviz rules of the views wave):
 * - one linear scale per axis, drawn to scale;
 * - a domain is the extent the drawn data reaches (references included), never padded to round
 *   numbers, so every tick names a value the data reaches;
 * - a domain with no width (one point, or every value equal) widens symmetrically around it so
 *   it can be drawn, and its one value is its only tick (the widening is never labelled);
 * - the pixel range is inset by `PAD` from the plot's edges, so a mark at the domain's edge sits
 *   inside the plot, never on an axis line.
 */

export type Domain = readonly [number, number];

/** The finite extent of `values`; null when none is finite. */
export function extent(values: Iterable<number | null | undefined>): Domain | null {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of values) {
    if (v === null || v === undefined || !Number.isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return lo <= hi ? [lo, hi] : null;
}

/** A domain with width: a single value widens by 10% of its size (or by 1 at zero). */
export function widen(d: Domain): Domain {
  if (d[1] > d[0]) return d;
  const half = d[0] === 0 ? 1 : Math.abs(d[0]) * 0.1;
  return [d[0] - half, d[0] + half];
}

/** The union of domains (nulls skipped). */
export function union(...ds: (Domain | null | undefined)[]): Domain | null {
  return extent(ds.flatMap((d) => (d ? [d[0], d[1]] : [])));
}

/** The inset, in pixels, between a plot's edge and the domain's extreme: a marker (radius 5, ring
 *  1) and a little air, so extreme marks never sit on an axis line. */
export const PAD = 8;

/** A pixel range moved `pad` inward at both ends (either orientation). */
export function insetRange(range: Domain, pad = PAD): Domain {
  const dir = range[1] >= range[0] ? 1 : -1;
  if (Math.abs(range[1] - range[0]) <= 2 * pad) return range;
  return [range[0] + dir * pad, range[1] - dir * pad];
}

/** A linear scale over a domain with width, onto a pixel range inset by `pad`. */
export function linear(domain: Domain, range: Domain, pad = 0): ScaleLinear<number, number> {
  return scaleLinear().domain(widen(domain)).range(insetRange(range, pad));
}

/**
 * Ticks inside the domain: round values the axis reaches, never beyond it. When the domain is too
 * narrow for a round value inside, its two ends are the ticks; a domain with no width (one value)
 * has that value as its only tick, never the widening around it.
 */
export function ticksIn(domain: Domain, count = 5): number[] {
  const [lo, hi] = domain;
  if (!(hi > lo)) return [lo];
  const eps = (hi - lo) * 1e-9;
  const t = scaleLinear().domain([lo, hi]).ticks(count).filter((v) => v >= lo - eps && v <= hi + eps);
  return t.length >= 2 ? t : [lo, hi];
}

/** Where a tick label anchors so its text stays inside [0, width]. */
export function anchorAt(x: number, width: number, half = 18): "start" | "middle" | "end" {
  if (x - half < 0) return "start";
  if (x + half > width) return "end";
  return "middle";
}

/** The plot box inside a view: its outer size and the margins the axes need. */
export interface Box {
  width: number;
  height: number;
  top: number;
  right: number;
  bottom: number;
  left: number;
}

export const inner = (b: Box) => ({ x0: b.left, x1: b.width - b.right, y0: b.top, y1: b.height - b.bottom });

/** The index of the value in sorted `xs` nearest to `x` (a crosshair's snap). */
export function nearest(xs: readonly number[], x: number): number {
  if (!xs.length) return -1;
  let lo = 0;
  let hi = xs.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid]! < x) lo = mid;
    else hi = mid;
  }
  return Math.abs(xs[lo]! - x) <= Math.abs(xs[hi]! - x) ? lo : hi;
}

/** Runs of consecutive defined points, so a line or band breaks where a value is missing. */
export function runs<T>(points: readonly T[], defined: (p: T) => boolean): T[][] {
  const out: T[][] = [];
  let cur: T[] = [];
  for (const p of points) {
    if (defined(p)) cur.push(p);
    else if (cur.length) {
      out.push(cur);
      cur = [];
    }
  }
  if (cur.length) out.push(cur);
  return out;
}

/** An SVG path through points (already in pixels). */
export function pathOf(pts: readonly (readonly [number, number])[]): string {
  return pts.map(([x, y], i) => `${i ? "L" : "M"}${round(x)},${round(y)}`).join("");
}

/** A closed band between an upper and a lower edge (pixels, same x order). */
export function bandOf(upper: readonly (readonly [number, number])[], lower: readonly (readonly [number, number])[]): string {
  if (!upper.length) return "";
  return `${pathOf(upper)}${[...lower].reverse().map(([x, y]) => `L${round(x)},${round(y)}`).join("")}Z`;
}

const round = (v: number) => Math.round(v * 100) / 100;
