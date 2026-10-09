/**
 * The curve view's data contract and its pure geometry (tested with known pixels).
 *
 * A curve is an estimate across one input's values: a substitution curve (the change in the
 * outcome as k kcal move from one source to another) or an exposure curve (the outcome's
 * difference from a reference value across the exposure, from a spline). It draws after Fit only
 * (FOUNDATION §5 rule 6): before the gate, `sealed` says when it opens and only the input's
 * observed support is drawn, which reads no outcome.
 */
import { extent, inner, linear, PAD, ticksIn, union, type Box, type Domain } from "./common/scale";
import { CHAR_W, fmtTick, leftFor, type Slot } from "./common/parts";

export interface CurveLine {
  key: string;
  label: string;
  /**
   * now: the curve as it stands (gray).
   * choice: the same curve with the pointed choice (indigo). Only a choice made after the lock
   *   (Results) may be previewed this way: a choice made before Fit (a model's shape, what is
   *   adjusted for) is never shown as an estimate while it is chosen (FOUNDATION §5 rule 6); after
   *   Fit its alternatives are compared as `series`, as sensitivity.
   * series: one of several entities compared (the comparison palette, by `slot`).
   */
  role: "now" | "choice" | "series";
  slot?: Slot;
  x: number[];
  /** null where the estimate stops (off support, refused) */
  y: (number | null)[];
  low?: (number | null)[] | null;
  high?: (number | null)[] | null;
}

export interface CurveData {
  xLabel: string;
  yLabel: string;
  lines: CurveLine[];
  /** the no-change reference: y = 0 for a difference */
  zero?: boolean;
  /** observed values of the input (or their quantiles): the rug */
  rug?: number[] | null;
  /** the share of rows on support at each x (a substitution's per-k support), drawn in its own
   *  labelled strip under the plot, on its own 0–100% scale */
  support?: { x: number[]; share: number[] } | null;
  /** where the curve stops, and why, in plain words; it must hold for every line drawn */
  stop?: { x: number; why: string } | null;
  /** the gate (FOUNDATION §5 rule 6): before it, one line saying when the curve opens, and nothing
   *  estimated is drawn; null once the curve is open. Every caller says which. */
  sealed: string | null;
  /** what the curve averages over, quietly */
  basis?: string | null;
  /** the band's meaning, quietly ("95% bootstrap band, 200 resamples") */
  band?: string | null;
  /** what x's values are, for the tooltip and the table ("k (kcal moved)") */
  xName?: string;
}

/** The main plot's top edge, and the space under the x axis for its labels and title. */
export const BOX_TOP = 24;
export const BOX_BOTTOM = 38;
export const HEIGHT = 300;
/** The support strip: its height, and the gap above it that holds its label. */
export const STRIP_H = 28;
export const STRIP_GAP = 24;
/** The widest right margin direct labels may take. */
export const LABEL_CAP = 170;

/** Lines that draw: every line, except the choice while the flip shows the data now. */
export function visibleLines(data: CurveData, showChoice: boolean): CurveLine[] {
  return data.lines.filter((l) => l.role !== "choice" || showChoice);
}

/** The support strip is drawn when per-x support is served and no rug stands in for it. */
export function hasStrip(data: CurveData): boolean {
  return !!(data.support && data.support.x.length && !data.rug?.length);
}

/** The x extent the drawn data reaches: defined estimates, the rug, the support and the stop. */
export function xDomain(data: CurveData): Domain | null {
  const defined = data.lines.flatMap((l) => l.x.filter((_, i) => l.y[i] !== null && l.y[i] !== undefined));
  const sealed = !!data.sealed;
  return union(
    sealed ? null : extent(defined),
    extent(data.rug ?? []),
    // After the gate the curve and its stop bound x; before it, the support is all there is.
    data.support && sealed ? extent(data.support.x.filter((_, i) => (data.support!.share[i] ?? 0) > 0)) : null,
    data.stop && !sealed ? [data.stop.x, data.stop.x] : null,
  );
}

/** The y extent: every estimate and band of every line (the choice included, so the axis holds
 *  still across the flip), and zero when the curve is a difference. */
export function yDomain(data: CurveData): Domain | null {
  const vals = data.lines.flatMap((l) => [...l.y, ...(l.low ?? []), ...(l.high ?? [])]);
  const d = extent(vals);
  if (!d) return null;
  return data.zero ? union(d, [0, 0]) : d;
}

export function hasEstimate(data: CurveData): boolean {
  return data.lines.some((l) => l.y.some((v) => v !== null && v !== undefined && Number.isFinite(v)));
}

export interface CurveScales {
  box: Box;
  x: ReturnType<typeof linear>;
  y: ReturnType<typeof linear>;
  xTicks: number[];
  yTicks: number[];
  /** the main plot's top and bottom (equal when sealed: no estimate is drawn) */
  plotTop: number;
  plotBottom: number;
  /** the line the x axis and its labels hang from: the strip's base, or the plot's bottom */
  axisY: number;
  /** the support strip, on its own scale (a share, 0 at its base, 1 at its top) */
  strip: { top: number; bottom: number; y: (share: number) => number } | null;
}

export function curveScales(data: CurveData, width: number, left: number, right = 12, height = HEIGHT): CurveScales | null {
  const xd = xDomain(data);
  if (!xd) return null;
  const yd = yDomain(data) ?? [0, 1];
  const sealed = !!data.sealed;
  const withStrip = hasStrip(data);
  const plotTop = sealed ? (withStrip ? STRIP_GAP : 8) : BOX_TOP;
  const plotBottom = sealed ? (withStrip ? STRIP_GAP : 28) : height - BOX_BOTTOM;
  const strip = withStrip ? { top: plotBottom + (sealed ? 0 : STRIP_GAP), bottom: plotBottom + (sealed ? 0 : STRIP_GAP) + STRIP_H } : null;
  const axisY = strip ? strip.bottom : plotBottom;
  const box: Box = { width, height: axisY + BOX_BOTTOM, top: plotTop, right, bottom: BOX_BOTTOM, left };
  const { x0, x1 } = inner(box);
  return {
    box,
    x: linear(xd, [x0, x1], PAD),
    y: linear(yd, [plotBottom, plotTop], PAD),
    xTicks: ticksIn(xd, Math.max(3, Math.floor((x1 - x0) / 90))),
    yTicks: ticksIn(yd, 5),
    plotTop,
    plotBottom,
    axisY,
    strip: strip ? { ...strip, y: (share: number) => strip.bottom - share * STRIP_H } : null,
  };
}

/** Every x any visible line is defined at, sorted: where the crosshair can stop. */
export function stops(lines: CurveLine[]): number[] {
  const xs = new Set<number>();
  for (const l of lines) l.x.forEach((v, i) => (l.y[i] !== null && l.y[i] !== undefined ? xs.add(v) : null));
  return [...xs].sort((a, b) => a - b);
}

/** Where the crosshair stops: each defined estimate, and each strip bar inside the x domain. */
export function readAt(data: CurveData, lines: CurveLine[], domain: Domain): number[] {
  const xs = new Set(stops(lines));
  if (hasStrip(data)) data.support!.x.forEach((v) => (v >= domain[0] && v <= domain[1] ? xs.add(v) : null));
  return [...xs].sort((a, b) => a - b);
}

export interface EndLabel {
  key: string;
  label: string;
  x: number;
  y: number;
}

/** Direct labels at each line's last defined point, or null when two would collide or one would
 *  run past `maxX` (the legend and the tooltip carry identity then). */
export function endLabels(lines: CurveLine[], x: (v: number) => number, y: (v: number) => number, maxX = Infinity, gap = 14): EndLabel[] | null {
  if (lines.length < 2 || lines.length > 4) return null;
  const out = lines.flatMap((l) => {
    for (let i = l.y.length - 1; i >= 0; i--) {
      const v = l.y[i];
      if (v !== null && v !== undefined) return [{ key: l.key, label: l.label, x: x(l.x[i]!), y: y(v) }];
    }
    return [];
  });
  if (out.some((o) => o.x + 7 + o.label.length * CHAR_W > maxX)) return null;
  const ys = out.map((o) => o.y).sort((a, b) => a - b);
  for (let i = 1; i < ys.length; i++) if (ys[i]! - ys[i - 1]! < gap) return null;
  return out;
}

/**
 * The curve's frame: its scales, with a right margin for direct labels only when they draw. Direct
 * labels are tried for two to four lines on a wide view; they draw when they stand apart and fit
 * inside the view, and the margin is given back otherwise.
 */
export function curveFrame(data: CurveData, lines: CurveLine[], width: number): { g: CurveScales; labels: EndLabel[] | null } | null {
  const pre = curveScales(data, width, 40);
  if (!pre) return null;
  // The strip's "100%" sits in the left margin beside the y tick labels.
  const stripLeft = hasStrip(data) ? leftFor([1], () => "100%") : 12;
  const left = data.sealed ? stripLeft : Math.max(stripLeft, leftFor(pre.yTicks, fmtTick));
  if (!data.sealed && lines.length >= 2 && lines.length <= 4 && width >= 480) {
    const room = Math.ceil(12 + Math.max(...lines.map((l) => l.label.length)) * CHAR_W);
    if (room <= LABEL_CAP) {
      const g = curveScales(data, width, left, room)!;
      const labels = endLabels(lines, g.x, g.y, width);
      if (labels) return { g, labels };
    }
  }
  return { g: curveScales(data, width, left, 12)!, labels: null };
}
