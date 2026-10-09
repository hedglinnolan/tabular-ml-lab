/**
 * The scale math every exhibit view shares, pure so it is tested with known pixels.
 *
 * Rules (FOUNDATION §5 rule 9; the dataviz rules of the views wave):
 * - one linear scale per axis, drawn to scale;
 * - a domain is the extent the drawn data reaches (references included), never padded to round
 *   numbers, so every tick names a value the data reaches;
 * - a domain with no width (one point, or every value equal) widens symmetrically around it.
 */
import { scaleLinear, type ScaleLinear } from "d3-scale";

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

/** A linear scale over a domain with width. */
export function linear(domain: Domain, range: Domain): ScaleLinear<number, number> {
  return scaleLinear().domain(widen(domain)).range(range);
}

/**
 * Ticks inside the domain: round values the axis reaches, never beyond it. When the domain is too
 * narrow for a round value inside, its two ends are the ticks.
 */
export function ticksIn(domain: Domain, count = 5): number[] {
  const [lo, hi] = widen(domain);
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
