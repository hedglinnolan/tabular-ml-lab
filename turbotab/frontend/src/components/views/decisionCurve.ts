/**
 * The decision curve view's data contract and geometry.
 *
 * The engine's `decision_curve` (turbotab/core/models/decision_curve.py) returns one row per
 * threshold with each model's net benefit, treat all and treat none (Vickers & Elkin 2006); the
 * evaluation stage serves it with the declared range `low`–`high` and the `useful` span. Treat all
 * falls steeply as the threshold rises, so the y axis stops a little below the models' lowest net
 * benefit and the reference leaves the plot there (the table keeps every value).
 *
 * The gate: the threshold range is chosen in the Results Confirm sweep, before the held-out rows
 * open (FOUNDATION §3), so a decision curve on held-out scores is never drawn while that range is
 * pointed at: choosing it there would be choosing by the held-out score.
 */
import { extent, inner, linear, PAD, ticksIn, type Box, type Domain } from "./common/scale";
import { CHAR_W, fmtTick, type Slot } from "./common/parts";

export interface DecisionCurveRow {
  threshold: number;
  treat_all: number;
  treat_none: number;
  models: Record<string, number | null>;
}

/** Where the predictions were scored. */
export type ScoredWhere = "out_of_fold" | "held_out";

export interface DecisionCurveData {
  rows: DecisionCurveRow[];
  /** the models drawn, in the comparison palette's order (the reported model first) */
  models: { key: string; label: string; slot: Slot }[];
  /** the declared threshold range, shaded */
  low: number | null;
  high: number | null;
  /** while an option that changes the threshold range is pointed at: the range it would declare,
   *  drawn in indigo beside the declared range in gray (FOUNDATION §4, §5 rule 5) */
  pointed?: { low: number; high: number } | null;
  /** where the reported model beats treating everyone and no one */
  useful?: readonly [number, number] | null;
  /** where the predictions were scored: a held-out curve refuses a pointed range */
  where: ScoredWhere;
  /** the gate (FOUNDATION §5 rule 6): before it, one line saying when the curve opens, and nothing
   *  scored is drawn; null once it is open. Every caller says which. */
  sealed: string | null;
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
    x: linear(dom.x, [x0, x1], PAD),
    y: linear(dom.y, [y1, y0], PAD),
    xTicks: ticksIn(dom.x, Math.max(3, Math.floor((x1 - x0) / 80))),
    yTicks: ticksIn(dom.y, 5),
    floor: dom.y[0],
  };
}

/** A range clamped to the thresholds served; null when it misses them. */
function clampRange(d: DecisionCurveData, low: number | null, high: number | null): Domain | null {
  if (low === null || high === null) return null;
  const x = extent(d.rows.map((r) => r.threshold));
  if (!x) return null;
  const lo = Math.max(x[0], low);
  const hi = Math.min(x[1], high);
  return hi > lo ? [lo, hi] : null;
}

/** The declared range, clamped to the thresholds served; null when it misses them. */
export function shadedRange(d: DecisionCurveData): Domain | null {
  return clampRange(d, d.low, d.high);
}

/** The pointed option's range, clamped the same way. */
export function pointedRange(d: DecisionCurveData): Domain | null {
  return d.pointed ? clampRange(d, d.pointed.low, d.pointed.high) : null;
}

/** Why this curve may not be drawn now, or null: a held-out curve while its range is chosen. */
export function refusal(d: DecisionCurveData): string | null {
  if (d.pointed && d.where === "held_out") {
    return "The threshold range is being chosen, so its decision curve is drawn on out-of-fold scores: the held-out rows stay closed until it is fixed.";
  }
  return null;
}

/** The sentence saying where the reported model does better, a single threshold said as one. */
export function usefulLine(d: DecisionCurveData): string {
  const name = d.models[0]!.label;
  if (!d.useful) return `${name} does better than treating everyone or no one at no threshold served.`;
  const [a, b] = d.useful;
  return a === b ? `${name} does better than treating everyone or no one at a threshold of ${fmtTick(a)}.` : `${name} does better than treating everyone or no one from ${fmtTick(a)} to ${fmtTick(b)}.`;
}

/** A text box: its left, right, top and bottom in pixels. */
interface TextBox {
  l: number;
  r: number;
  t: number;
  b: number;
}

const inside = (b: TextBox, px: number, py: number) => px >= b.l && px <= b.r && py >= b.t && py <= b.b;

/**
 * Where the "Treat everyone" label sits: near the start of its line, just under it and to its left,
 * where the line has moved far enough from the plot's left edge to hold the text, clear of the
 * plot's edges and every line's points, and above the treat-no-one line. Null when no spot in the
 * line's first half is clear (the legend names it then).
 */
export function treatAllLabelAt(
  d: DecisionCurveData,
  x: (v: number) => number,
  y: (v: number) => number,
  plot: { x0: number; y0: number; y1: number },
  text = "Treat everyone",
): { x: number; y: number; anchor: "start" | "end" } | null {
  const w = text.length * CHAR_W * (11 / 12);
  const pts = d.rows.flatMap((r) => [[x(r.threshold), y(r.treat_all)] as const, ...d.models.flatMap((m) => (r.models[m.key] === null || r.models[m.key] === undefined ? [] : [[x(r.threshold), y(r.models[m.key]!)] as const]))]);
  const zero = y(0);
  const clear = (b: TextBox) => b.l >= plot.x0 && b.t >= plot.y0 && b.b <= plot.y1 && !(zero >= b.t - 2 && zero <= b.b + 2) && !pts.some(([px, py]) => inside(b, px, py));
  // Along the line, the label stays above treat no one, so it is never read as naming that line.
  const aboveZero = (b: TextBox) => b.b <= zero - 2;
  if (d.rows.length === 1) {
    const r = d.rows[0]!;
    const px = x(r.threshold);
    const py = y(r.treat_all);
    return clear({ l: px + 8, r: px + 9 + w, t: py - 7, b: py + 4 }) ? { x: px + 9, y: py + 4, anchor: "start" } : null;
  }
  // Along the line's first half, the first spot left of it that holds the text.
  for (let i = 0; i < Math.ceil(d.rows.length / 2); i++) {
    const r = d.rows[i]!;
    const px = x(r.threshold) - 6;
    const py = y(r.treat_all) + 10;
    const b = { l: px - w, r: px, t: py - 9, b: py + 3 };
    if (clear(b) && aboveZero(b)) return { x: px, y: py, anchor: "end" };
  }
  return null;
}

/** The right margin the direct labels at the line ends need, or null when they cannot fit. */
export function directRoom(d: DecisionCurveData, cap = 150): number | null {
  const room = Math.ceil(14 + Math.max(...d.models.map((m) => m.label.length), "Treat no one".length) * CHAR_W);
  return room <= cap ? room : null;
}
