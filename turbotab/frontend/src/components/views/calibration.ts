/**
 * The calibration view's data contract and geometry: predicted against observed on one shared
 * scale (both axes take the same domain, in a square plot, so the 45° line is drawn at 45°).
 *
 * The engine serves `Calibration` (turbotab/core/models/performance.py): calibration in the large,
 * the slope, and the lowess curve at up to 40 quantiles. The binned points with intervals are not
 * served yet: `bins` is optional and named as an engine contract item.
 *
 * The gate: the calibration horizon is chosen in the Results Confirm sweep, before the held-out
 * rows open (FOUNDATION §3), so a held-out calibration is never drawn while a choice it would
 * inform is pointed at (`choosing`).
 */
import { extent, inner, linear, PAD, ticksIn, union, type Box, type Domain } from "./scale";
import type { ScoredWhere } from "./decisionCurve";

export interface Interval {
  estimate: number | null;
  ci_low?: number | null;
  ci_high?: number | null;
  level?: number | null;
}

export interface CalibrationBin {
  /** the mean prediction in the group */
  predicted: number;
  /** the observed share (risk) or mean (value) in the group */
  observed: number;
  low: number | null;
  high: number | null;
  n: number;
}

export interface CalibrationData {
  /** risk: predicted probabilities of a yes/no outcome; value: predicted values of a number */
  kind: "risk" | "value";
  /** what is predicted, in plain words ("progression", "glucose") */
  outcome: string;
  n: number;
  observed: number;
  expected: number;
  intercept: Interval;
  slope: Interval;
  /** the smoothed curve (the engine's lowess) */
  curve: { x: number; y: number }[];
  bins?: CalibrationBin[] | null;
  /** how the bins were made, quietly */
  binsMethod?: string | null;
  smoother?: string | null;
  /** the engine's concern, when calibration is flagged: its verdict, said in place of the numbers */
  concern?: string | null;
  /** where the predictions were scored: a held-out calibration refuses `choosing` */
  where: ScoredWhere;
  /** the choice being made that this calibration would inform, while it is pointed at ("the
   *  calibration horizon"); null at rest */
  choosing?: string | null;
  /** the gate (FOUNDATION §5 rule 6): before it, one line saying when calibration opens, and
   *  nothing scored is drawn; null once it is open. Every caller says which. */
  sealed: string | null;
}

/** Why this calibration may not be drawn now, or null: held-out scores while a choice they would
 *  inform is being made. */
export function calibrationRefusal(d: CalibrationData): string | null {
  if (d.choosing && d.where === "held_out") {
    return `${d.choosing[0]!.toUpperCase()}${d.choosing.slice(1)} is being chosen, so calibration is drawn on out-of-fold scores: the held-out rows stay closed until it is fixed.`;
  }
  return null;
}

/** The one domain both axes share: everything drawn, the diagonal spanning it. A risk stays in
 *  [0, 1]. */
export function sharedDomain(d: CalibrationData): Domain | null {
  const pts = d.curve.flatMap((p) => [p.x, p.y]);
  const bins = (d.bins ?? []).flatMap((b) => [b.predicted, b.observed, b.low, b.high]);
  const e = union(extent(pts), extent(bins));
  if (!e) return null;
  return d.kind === "risk" ? [Math.max(0, e[0]), Math.min(1, e[1])] : e;
}

export interface CalibrationScales {
  box: Box;
  /** the side of the square plot */
  side: number;
  x: ReturnType<typeof linear>;
  y: ReturnType<typeof linear>;
  ticks: number[];
}

export function calibrationScales(d: CalibrationData, width: number, left: number, maxSide = 380): CalibrationScales | null {
  const dom = sharedDomain(d);
  if (!dom) return null;
  const side = Math.max(160, Math.min(maxSide, width - left - 16));
  const box: Box = { width: Math.max(width, left + side + 16), height: side + 24 + 38, top: 24, right: Math.max(16, width - left - side), bottom: 38, left };
  const { x0, x1, y0, y1 } = inner(box);
  return {
    box,
    side,
    x: linear(dom, [x0, x1], PAD),
    y: linear(dom, [y1, y0], PAD),
    ticks: ticksIn(dom, 5),
  };
}

export function hasCalibration(d: CalibrationData | null): d is CalibrationData {
  return !!d && (d.curve.length > 0 || (d.bins?.length ?? 0) > 0);
}
