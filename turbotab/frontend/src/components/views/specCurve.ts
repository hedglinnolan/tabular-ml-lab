/**
 * The specification curve's data contract and geometry (Simonsohn, Simmons & Nelson 2020): the
 * estimate in every declared specification, sorted, with a matrix below that marks which option
 * of each choice the specification took. Both panels share one x: the specification's rank.
 *
 * Sensitivity, never a way to choose: the primary is the reported estimate whatever the others say
 * (BLUEPRINT §11.4). Every specification must estimate the same quantity on the same scale; the
 * view refuses a mix (`scaleKey`).
 *
 * The gate: a specification curve is estimates, so it opens only after the lock (FOUNDATION §5
 * rule 6); shown while the plan is chosen it would invite choosing by the estimate. `sealed`
 * carries the line saying when it opens, and nothing estimated is drawn before then.
 */
import { extent, inner, linear, PAD, ticksIn, union, type Box, type Domain } from "./scale";

export interface SpecChoice {
  key: string;
  /** the choice in plain words ("Calories handled by") */
  label: string;
  options: { key: string; label: string }[];
}

export interface Spec {
  key: string;
  estimate: number;
  low: number | null;
  high: number | null;
  n: number;
  primary: boolean;
  /** the option each choice took: choice key → option key */
  picks: Record<string, string>;
  /** what the estimate is a difference per, so a mix of scales is refused */
  scaleKey?: string;
}

export interface SpecCurveData {
  /** what the estimate is, in plain words ("Difference in mean glucose per g of sugar") */
  estimateLabel: string;
  choices: SpecChoice[];
  specs: Spec[];
  /** draw the no-difference line */
  zero?: boolean;
  /** the level of every interval, as served (0.95); null when the intervals do not say */
  level: number | null;
  /** the gate (FOUNDATION §5 rule 6): before the lock, one line saying when the curve opens, and
   *  nothing estimated is drawn; null once it is open. Every caller says which. */
  sealed: string | null;
}

/** Sorted by estimate, ties by key, so the order is stable. */
export function sortSpecs(specs: readonly Spec[]): Spec[] {
  return [...specs].sort((a, b) => a.estimate - b.estimate || (a.key < b.key ? -1 : a.key > b.key ? 1 : 0));
}

/** The scales the specifications mix, when they mix more than one. */
export function mixedScales(specs: readonly Spec[]): string[] | null {
  const keys = [...new Set(specs.map((s) => s.scaleKey ?? ""))];
  return keys.length > 1 ? keys : null;
}

export function specYDomain(d: SpecCurveData): Domain | null {
  const e = extent(d.specs.flatMap((s) => [s.estimate, s.low, s.high]));
  return d.zero ? union(e, [0, 0]) : e;
}

export const TOP = 24;
export const CURVE_H = 190;
export const GAP = 22;
export const HEAD_H = 20;
export const ROW_H = 17;

export interface SpecLayout {
  /** the box of the whole figure; its plot spans both panels */
  box: Box;
  /** the curve panel's bottom edge */
  curveBottom: number;
  x0: number;
  colW: number;
  /** the center of the column at rank i */
  cx: (i: number) => number;
  y: ReturnType<typeof linear>;
  yTicks: number[];
  /** each choice's heading row and its options' rows */
  rows: { kind: "choice" | "option"; choice: string; option?: string; label: string; y: number }[];
}

export function specLayout(d: SpecCurveData, width: number, gutter: number): SpecLayout | null {
  const yd = specYDomain(d);
  if (!yd) return null;
  const curveBottom = TOP + CURVE_H;
  let yy = curveBottom + GAP;
  const rows: SpecLayout["rows"] = [];
  for (const c of d.choices) {
    rows.push({ kind: "choice", choice: c.key, label: c.label, y: yy + HEAD_H - 6 });
    yy += HEAD_H;
    for (const o of c.options) {
      rows.push({ kind: "option", choice: c.key, option: o.key, label: o.label, y: yy + ROW_H / 2 });
      yy += ROW_H;
    }
  }
  const height = yy + 8;
  const box: Box = { width, height, top: TOP, right: 10, bottom: height - Math.max(curveBottom, yy), left: gutter };
  const { x0, x1 } = inner(box);
  const n = Math.max(1, d.specs.length);
  const colW = (x1 - x0) / n;
  return {
    box,
    curveBottom,
    x0,
    colW,
    cx: (i) => x0 + (i + 0.5) * colW,
    y: linear(yd, [curveBottom, TOP], PAD),
    yTicks: ticksIn(yd, 5),
    rows,
  };
}

/** The column at a pixel x (clamped), for the hover that spans both panels. */
export function columnAt(l: SpecLayout, px: number, n: number): number {
  return Math.min(n - 1, Math.max(0, Math.floor((px - l.x0) / l.colW)));
}
