/**
 * The overlap view's geometry, pure: one horizontal scale for both groups and one vertical scale
 * (each group's share of its own rows) mirrored about the baseline, so a bar of 10% above is as
 * tall as a bar of 10% below.
 */
import { scaleLinear } from "d3-scale";
import { fmtInt, fmtTick } from "../../stage/format";
import { fmtPct } from "../common/frame";
import { gateRefusal } from "../common/gate";
import type { OverlapInput } from "./types";

export const OVERLAP = { height: 260, left: 44, right: 10, top: 24, bottom: 42, gap: 1 } as const;

export interface OverlapBin {
  i: number;
  lo: number;
  hi: number;
  x0: number;
  x1: number;
  /** wholly outside the kept range */
  trimmed: boolean;
  /** the kept range's edge falls inside this bin */
  split: boolean;
  /** [above, below] */
  groups: { count: number; share: number; h: number }[];
}

export interface OverlapLayout {
  width: number;
  height: number;
  left: number;
  right: number;
  top: number;
  /** the baseline both groups grow from */
  mid: number;
  bottom: number;
  half: number;
  x: (v: number) => number;
  /** a share's bar height, the same above and below */
  h: (share: number) => number;
  xTicks: { value: number; px: number; label: string }[];
  yTicks: { value: number; up: number; down: number; label: string }[];
  bins: OverlapBin[];
  totals: [number, number];
  maxShare: number;
  keep: { lo: number; hi: number; xlo: number; xhi: number; nTrimmed: number | null; anySplit: boolean } | null;
}

export type OverlapResult = { empty: string } | { layout: OverlapLayout };

const EPS = 1e-9;

export function layoutOverlap(input: OverlapInput, width: number): OverlapResult {
  const refused = gateRefusal(input.outcome, [input.column]);
  if (refused) return { empty: refused };
  const { edges, groups } = input;
  const k = edges.length - 1;
  if (k < 1 || groups.some((g) => g.counts.length !== k)) {
    return { empty: "The overlap's bins and counts do not match, so nothing is drawn." };
  }
  if (edges.some((e, i) => i > 0 && !(e > edges[i - 1]!))) {
    return { empty: "The overlap's bin edges are not in increasing order, so nothing is drawn." };
  }
  const totals = groups.map((g) => g.counts.reduce((a, b) => a + b, 0)) as [number, number];
  if (totals[0] === 0 && totals[1] === 0) return { empty: "No rows to draw: neither group has any rows here." };
  const none = totals.findIndex((t) => t === 0);
  if (none >= 0) {
    return {
      empty: `No rows have ${groups[none]!.label}, so there is nothing for ${groups[1 - none]!.label} to overlap with.`,
    };
  }

  const { height, left, right, top, bottom } = OVERLAP;
  const half = (height - top - bottom) / 2;
  const mid = top + half;
  const lo = input.scale === "propensity" ? Math.min(0, edges[0]!) : edges[0]!;
  const hi = input.scale === "propensity" ? Math.max(1, edges[k]!) : edges[k]!;
  const xs = scaleLinear().domain([lo, hi]).range([left, width - right]);
  const shares = groups.map((g, gi) => g.counts.map((c) => c / totals[gi]!));
  const maxShare = Math.max(...shares.flat());
  const ys = scaleLinear().domain([0, maxShare]).range([0, half]);

  const keep = input.keep;
  const bins: OverlapBin[] = [];
  for (let i = 0; i < k; i++) {
    const b0 = edges[i]!;
    const b1 = edges[i + 1]!;
    const trimmed = !!keep && (b1 <= keep.lo + EPS || b0 >= keep.hi - EPS);
    const split = !!keep && ((b0 < keep.lo - EPS && keep.lo < b1 - EPS) || (b0 < keep.hi - EPS && keep.hi < b1 - EPS));
    bins.push({
      i,
      lo: b0,
      hi: b1,
      x0: xs(b0),
      x1: xs(b1),
      trimmed,
      split,
      groups: groups.map((g, gi) => ({ count: g.counts[i]!, share: shares[gi]![i]!, h: ys(shares[gi]![i]!) })),
    });
  }

  let keepOut: OverlapLayout["keep"] = null;
  if (keep) {
    const anySplit = bins.some((b) => b.split);
    const summed = bins.filter((b) => b.trimmed).reduce((a, b) => a + b.groups[0]!.count + b.groups[1]!.count, 0);
    keepOut = {
      lo: keep.lo,
      hi: keep.hi,
      xlo: xs(Math.max(lo, keep.lo)),
      xhi: xs(Math.min(hi, keep.hi)),
      nTrimmed: keep.n_trimmed ?? (anySplit ? null : summed),
      anySplit,
    };
  }

  // Ticks name only values the data reaches: inside the drawn range, and shares no taller than
  // the tallest bar.
  const xTicks = xs.ticks(5).map((t) => ({ value: t, px: xs(t), label: fmtTick(t) }));
  const yTicks = ys
    .ticks(3)
    .filter((t) => t > 0 && t <= maxShare + EPS)
    .slice(-2)
    .map((t) => ({ value: t, up: mid - ys(t), down: mid + ys(t), label: fmtPct(t) }));

  return {
    layout: {
      width,
      height,
      left,
      right: width - right,
      top,
      mid,
      bottom: height - bottom,
      half,
      x: (v) => xs(v),
      h: (s) => ys(s),
      xTicks,
      yTicks,
      bins,
      totals,
      maxShare,
      keep: keepOut,
    },
  };
}

/** Which side of the plot a group's direct label sits on: the end where its bars are lower. */
export function labelSide(bins: OverlapBin[], g: 0 | 1): "start" | "end" {
  const q = Math.max(1, Math.floor(bins.length / 4));
  const peak = (bs: OverlapBin[]) => Math.max(...bs.map((b) => b.groups[g]!.share));
  return peak(bins.slice(0, q)) <= peak(bins.slice(-q)) ? "start" : "end";
}

/**
 * A group's direct label baseline: just inside the top (the first group) or the bottom (the
 * second), moved clear of any gridline its text would sit on.
 */
export function labelY(l: OverlapLayout, g: 0 | 1): number {
  const grid = l.yTicks.map((t) => (g === 0 ? t.up : t.down)).sort((a, b) => (g === 0 ? a - b : b - a));
  let y = g === 0 ? l.top + 12 : l.bottom - 6;
  for (const line of grid) {
    // the text's box runs from 10 px above its baseline to 3 px below; keep 1 px clear of a line
    if (line >= y - 11 && line <= y + 4) y = g === 0 ? line + 13 : line - 5;
  }
  return y;
}

/** One line on the trim, in plain words. */
export function keepLine(input: OverlapInput, l: OverlapLayout): string | null {
  if (!l.keep || !input.keep) return null;
  const range = `${fmtTick(l.keep.lo)} to ${fmtTick(l.keep.hi)}`;
  const n = l.keep.nTrimmed;
  const count = n === null ? "Rows" : `${fmtInt(n)} ${n === 1 ? "row" : "rows"}`;
  const split = l.keep.anySplit ? " A bin the cut falls inside is drawn as kept." : "";
  return `${count} outside ${range} ${input.keep_state === "recorded" ? "were" : "would be"} trimmed.${split}`;
}
