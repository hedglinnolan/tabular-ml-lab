/**
 * The decision curve view's data contract and geometry.
 *
 * The engine's `decision_curve` (turbotab/core/models/decision_curve.py) returns one row per
 * threshold with each model's net benefit, treat all and treat none (Vickers & Elkin 2006); the
 * evaluation stage serves it with the declared range `low`–`high` and the `useful` span. Treat all
 * falls steeply as the threshold rises, so the y axis stops a little below the models' lowest net
 * benefit and the reference leaves the plot there (the table keeps every value).
 */
import { extent, inner, linear, ticksIn, type Box, type Domain } from "./scale";
import type { Slot } from "./parts";

export interface DecisionCurveRow {
  threshold: number;
  treat_all: number;
  treat_none: number;
  models: Record<string, number | null>;
}

export interface DecisionCurveData {
  rows: DecisionCurveRow[];
  /** the models drawn, in the comparison palette's order (the reported model first) */
  models: { key: string; label: string; slot: Slot }[];
  /** the declared threshold range, shaded */
  low: number | null;
  high: number | null;
  /** where the reported model beats treating everyone and no one */
  useful?: readonly [number, number] | null;
  prevalence?: number | null;
  n?: number | null;
}

export function modelValues(d: DecisionCurveData): number[] {
  return d.rows.flatMap((r) => d.models.map((m) => r.models[m.key]).filter((v): v is number => v !== null && v !== undefined && Number.isFinite(v)));
}

/** x: the thresholds served. y: the models' net benefit and treat none (0), with treat all down to
 *  at most a tenth of the top below zero, so a plunging reference never flattens the models. */
export function decisionDomains(d: DecisionCurveData): { x: Domain; y: Domain } | null {
  const x = extent(d.rows.map((r) => r.threshold));
  const m = extent([...modelValues(d), 0]);
  if (!x || !m) return null;
  const all = extent(d.rows.map((r) => r.treat_all)) ?? m;
  const top = Math.max(m[1], all[1]);
  const floor = Math.min(m[0], Math.max(all[0], -0.1 * Math.abs(top)));
  return { x, y: [floor, top] };
}

/** The first threshold where treating everyone leaves the plot (below its floor), if any. */
export function treatAllLeaves(d: DecisionCurveData, floor: number): number | null {
  const r = d.rows.find((row) => row.treat_all < floor - 1e-12);
  return r ? r.threshold : null;
}

export interface DecisionScales {
  box: Box;
  x: ReturnType<typeof linear>;
  y: ReturnType<typeof linear>;
  xTicks: number[];
  yTicks: number[];
  floor: number;
}

export function decisionScales(d: DecisionCurveData, width: number, left: number, right = 12, height = 300): DecisionScales | null {
  const dom = decisionDomains(d);
  if (!dom) return null;
  const box: Box = { width, height, top: 24, right, bottom: 38, left };
  const { x0, x1, y0, y1 } = inner(box);
  return {
    box,
    x: linear(dom.x, [x0, x1]),
    y: linear(dom.y, [y1, y0]),
    xTicks: ticksIn(dom.x, Math.max(3, Math.floor((x1 - x0) / 80))),
    yTicks: ticksIn(dom.y, 5),
    floor: dom.y[0],
  };
}

/** The declared range, clamped to the thresholds served; null when it misses them. */
export function shadedRange(d: DecisionCurveData): Domain | null {
  if (d.low === null || d.high === null) return null;
  const x = extent(d.rows.map((r) => r.threshold));
  if (!x) return null;
  const lo = Math.max(x[0], d.low);
  const hi = Math.min(x[1], d.high);
  return hi > lo ? [lo, hi] : null;
}
