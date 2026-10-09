/**
 * The curve view's data contract and its pure geometry (tested with known pixels).
 *
 * A curve is an estimate across one input's values: a substitution curve (the change in the
 * outcome as k kcal move from one source to another) or an exposure curve (the outcome's
 * difference from a reference value across the exposure, from a spline). It draws after Fit only
 * (FOUNDATION §5 rule 6): before the gate, `sealed` says when it opens and only the input's
 * observed support is drawn, which reads no outcome.
 */
import { extent, inner, linear, ticksIn, union, widen, type Box, type Domain } from "./scale";
import type { Slot } from "./parts";

export interface CurveLine {
  key: string;
  label: string;
  /** now: the curve as it stands (gray); choice: with the pointed choice (indigo); series: one of
   *  several entities compared (the comparison palette, by `slot`). */
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
  /** the share of rows on support at each x (a substitution's per-k support) */
  support?: { x: number[]; share: number[] } | null;
  /** where the curve stops, and why, in plain words */
  stop?: { x: number; why: string } | null;
  /** before the gate: one line saying when the curve opens; nothing estimated is drawn */
  sealed?: string | null;
  /** what the curve averages over, quietly */
  basis?: string | null;
  /** the band's meaning, quietly ("95% bootstrap band, 200 resamples") */
  band?: string | null;
  /** what x's values are, for the tooltip and the table ("k (kcal moved)") */
  xName?: string;
}

export const BOX_TOP = 24;
export const BOX_BOTTOM = 38;
export const HEIGHT = 300;

/** Lines that draw: every line, except the choice while the flip shows the data now. */
export function visibleLines(data: CurveData, showChoice: boolean): CurveLine[] {
  return data.lines.filter((l) => l.role !== "choice" || showChoice);
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
}

export function curveScales(data: CurveData, width: number, left: number, right = 12, height = HEIGHT): CurveScales | null {
  const xd = xDomain(data);
  if (!xd) return null;
  const yd = yDomain(data) ?? [0, 1];
  const box: Box = { width, height: data.sealed ? 96 : height, top: data.sealed ? 8 : BOX_TOP, right, bottom: BOX_BOTTOM, left };
  const { x0, x1, y0, y1 } = inner(box);
  return {
    box,
    x: linear(xd, [x0, x1]),
    y: linear(yd, [y1, y0]),
    xTicks: ticksIn(widen(xd), Math.max(3, Math.floor((x1 - x0) / 90))),
    yTicks: ticksIn(widen(yd), 5),
  };
}

/** Every x any visible line is defined at, sorted: where the crosshair can stop. */
export function stops(lines: CurveLine[]): number[] {
  const xs = new Set<number>();
  for (const l of lines) l.x.forEach((v, i) => (l.y[i] !== null && l.y[i] !== undefined ? xs.add(v) : null));
  return [...xs].sort((a, b) => a - b);
}

/** Direct labels at each line's last defined point, or null when two would collide (the legend
 *  and the tooltip carry identity then). */
export function endLabels(lines: CurveLine[], x: (v: number) => number, y: (v: number) => number, gap = 14): { key: string; label: string; x: number; y: number }[] | null {
  if (lines.length < 2 || lines.length > 4) return null;
  const out = lines.flatMap((l) => {
    for (let i = l.y.length - 1; i >= 0; i--) {
      const v = l.y[i];
      if (v !== null && v !== undefined) return [{ key: l.key, label: l.label, x: x(l.x[i]!), y: y(v) }];
    }
    return [];
  });
  const ys = out.map((o) => o.y).sort((a, b) => a - b);
  for (let i = 1; i < ys.length; i++) if (ys[i]! - ys[i - 1]! < gap) return null;
  return out;
}
