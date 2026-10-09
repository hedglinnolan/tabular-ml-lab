/**
 * The embedding view's geometry, pure. Both axes share one scale (the same pixels per unit), so a
 * distance reads the same in every direction, as PCA's components and UMAP's dimensions require
 * (Nguyen & Holmes 2019, PLoS Comput Biol 15:e1006907, tip 4). No tick carries a number: the axes
 * are derived, and a number on them would imply a unit.
 */
import type { LegendItem, Slot } from "../common/frame";
import { fmtPct } from "../common/frame";
import type { EmbeddingInput } from "./types";

export const EMBED = {
  left: 10,
  right: 10,
  top: 22,
  bottom: 24,
  /** above this many rows the view draws density, not points */
  densityAbove: 2000,
  /** a density cell's side, and a point's diameter, in pixels */
  cell: 8,
  radius: 4,
} as const;

export interface EmbedPoint {
  i: number;
  px: number;
  py: number;
  key: string;
}

export interface EmbedCell {
  col: number;
  row: number;
  cx: number;
  cy: number;
  count: number;
  /** rows per legend key in the cell */
  by: Record<string, number>;
  key: string;
  /** 1 (sparse) to 4 (densest) */
  step: 1 | 2 | 3 | 4;
}

export interface EmbedLayout {
  width: number;
  height: number;
  plot: { x0: number; x1: number; y0: number; y1: number };
  /** pixels per unit, the same on both axes */
  k: number;
  toX: (v: number) => number;
  toY: (v: number) => number;
  mode: "points" | "density";
  points: EmbedPoint[];
  cells: EmbedCell[];
  legend: (LegendItem & { count: number })[];
  axisTitles: [string, string];
  n: number;
}

export type EmbedResult = { empty: string } | { layout: EmbedLayout };

export const OPACITY = [0.3, 0.5, 0.75, 1] as const;

/** A row's legend key: its level (the first five keep their slots), "other", or "none". */
export function keyOf(g: number | null | undefined, levels: number): string {
  if (g === null || g === undefined || g < 0 || g >= levels) return "none";
  return g < 5 ? `g${g}` : "other";
}

export function axisTitle(input: EmbeddingInput, i: 0 | 1): string {
  const a = input.axes[i];
  if (input.method === "pca" && typeof a.share === "number") return `${a.label} · ${fmtPct(a.share)} of the spread`;
  return a.label;
}

export function embedHeight(width: number): number {
  return Math.round(Math.min(380, Math.max(220, width * 0.62)));
}

export function layoutEmbedding(input: EmbeddingInput, width: number, focus: string | null = null): EmbedResult {
  if (input.xs.length !== input.ys.length) {
    return { empty: "The embedding's two axes hold different numbers of rows, so nothing is drawn." };
  }
  const levels = input.grouping?.levels.length ?? 0;
  const rows: { i: number; x: number; y: number; key: string }[] = [];
  for (let i = 0; i < input.xs.length; i++) {
    const x = input.xs[i]!;
    const y = input.ys[i]!;
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    rows.push({ i, x, y, key: input.grouping ? keyOf(input.groups?.[i], levels) : "all" });
  }
  if (!rows.length) return { empty: "No rows to place: the embedding has no rows with both coordinates." };

  const height = embedHeight(width);
  const plot = { x0: EMBED.left, x1: width - EMBED.right, y0: EMBED.top, y1: height - EMBED.bottom };
  const pw = plot.x1 - plot.x0 - 2 * EMBED.radius;
  const ph = plot.y1 - plot.y0 - 2 * EMBED.radius;
  let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
  for (const r of rows) {
    xmin = Math.min(xmin, r.x); xmax = Math.max(xmax, r.x);
    ymin = Math.min(ymin, r.y); ymax = Math.max(ymax, r.y);
  }
  const xr = xmax - xmin;
  const yr = ymax - ymin;
  // One scale for both axes: the tighter of the two fits. A single point, or rows that all
  // coincide, sit at the center.
  const k = xr === 0 && yr === 0 ? 1 : Math.min(xr > 0 ? pw / xr : Infinity, yr > 0 ? ph / yr : Infinity);
  const cxData = (xmin + xmax) / 2;
  const cyData = (ymin + ymax) / 2;
  const cxPx = (plot.x0 + plot.x1) / 2;
  const cyPx = (plot.y0 + plot.y1) / 2;
  const toX = (v: number) => cxPx + (v - cxData) * k;
  const toY = (v: number) => cyPx - (v - cyData) * k;

  const counts = new Map<string, number>();
  for (const r of rows) counts.set(r.key, (counts.get(r.key) ?? 0) + 1);
  const legend: EmbedLayout["legend"] = [];
  if (input.grouping) {
    input.grouping.levels.slice(0, 5).forEach((lv, gi) => {
      legend.push({ key: `g${gi}`, label: lv, slot: (gi + 1) as Slot, shape: "dot", count: counts.get(`g${gi}`) ?? 0 });
    });
    if (counts.get("other")) legend.push({ key: "other", label: "Other levels", slot: null, shape: "dot", count: counts.get("other")! });
    if (counts.get("none")) legend.push({ key: "none", label: "Not recorded", slot: null, shape: "dot", count: counts.get("none")! });
  }

  const mode = rows.length > EMBED.densityAbove ? "density" : "points";
  const points: EmbedPoint[] = mode === "points" ? rows.map((r) => ({ i: r.i, px: toX(r.x), py: toY(r.y), key: r.key })) : [];
  const cells: EmbedCell[] = [];
  if (mode === "density") {
    const grid = new Map<string, EmbedCell>();
    for (const r of rows) {
      if (focus !== null && r.key !== focus) continue;
      const col = Math.floor((toX(r.x) - plot.x0) / EMBED.cell);
      const row = Math.floor((toY(r.y) - plot.y0) / EMBED.cell);
      const id = `${col},${row}`;
      let c = grid.get(id);
      if (!c) {
        c = { col, row, cx: plot.x0 + (col + 0.5) * EMBED.cell, cy: plot.y0 + (row + 0.5) * EMBED.cell, count: 0, by: {}, key: r.key, step: 1 };
        grid.set(id, c);
      }
      c.count++;
      c.by[r.key] = (c.by[r.key] ?? 0) + 1;
    }
    const order = legend.map((l) => l.key);
    const rank = (key: string) => (order.includes(key) ? order.indexOf(key) : order.length);
    const max = Math.max(1, ...[...grid.values()].map((c) => c.count));
    for (const c of grid.values()) {
      c.key = Object.entries(c.by).sort((a, b) => b[1] - a[1] || rank(a[0]) - rank(b[0]))[0]![0];
      c.step = Math.min(4, Math.max(1, Math.ceil(4 * Math.sqrt(c.count / max)))) as EmbedCell["step"];
      cells.push(c);
    }
  }

  return {
    layout: {
      width,
      height,
      plot,
      k,
      toX,
      toY,
      mode,
      points,
      cells,
      legend,
      axisTitles: [axisTitle(input, 0), axisTitle(input, 1)],
      n: rows.length,
    },
  };
}

/** The mark nearest a pointer, within `radius` pixels: hit targets larger than the marks. */
export function nearest(l: EmbedLayout, px: number, py: number, radius = 12): EmbedPoint | EmbedCell | null {
  let best: EmbedPoint | EmbedCell | null = null;
  let bd = radius * radius;
  const marks: (EmbedPoint | EmbedCell)[] = l.mode === "points" ? l.points : l.cells;
  for (const m of marks) {
    const x = "px" in m ? m.px : m.cx;
    const y = "py" in m ? m.py : m.cy;
    const d = (x - px) ** 2 + (y - py) ** 2;
    if (d <= bd) {
      bd = d;
      best = m;
    }
  }
  return best;
}
