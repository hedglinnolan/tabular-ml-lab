/**
 * The reshape storyboard (lifted from /lab/m2's reshape.ts) — pure, so its two claims are tested
 * without a browser: the frames come in the method's own order, and a row is the same row from the
 * file to the settled table.
 *
 * The preview of combining rows (`set_aggregation`, turbotab/core/structure_previews.py) is a
 * table_focus whose story is the method's real steps, with the row map on every frame row (`unit`,
 * `sources`). Four real, labeled states: 0 your data now (file order) · 1 each unit's records,
 * gathered · 2 combined (mean, change) or kept (first, last) · 3 settled, one row per unit. Every
 * number on screen is read from a whole state (`valuesAt`); only positions and opacities glide.
 *
 * Identity is the row (DESIGN_LANGUAGE §05.2): a combined row is its unit's records meeting in one
 * place — a true correspondence — and a record that leaves is struck where it stood.
 */
import type { TableFocusView } from "../../../api/m1-stage-types";

export const RH = 22;
export const GAP = 6;

export type Kind = "combine" | "keep";

export interface ReshapeRow {
  /** The row's id in the table as loaded (its place in the file). */
  row: number;
  unit: string;
  /** The unit's index, in order of first appearance in the file. */
  u: number;
  /** This record's place among its unit's records (the method's order: time, then file). */
  k: number;
  values: Record<string, unknown>;
}

export interface ReshapeModel {
  /** The column naming the unit. */
  idColumn: string;
  /** The other columns shown, in the preview's order. */
  columns: string[];
  /** Every record shown, in file order. */
  rows: ReshapeRow[];
  units: string[];
  /** Records per unit, by unit index. */
  sizes: number[];
  /** What each unit becomes: its values, and the source rows it stands for (the row map). */
  settled: Record<string, { row: number; values: Record<string, unknown>; sources: number[] }>;
  kind: Kind;
  /** Columns that change in the real data, beyond those shown. */
  nAffected: number;
}

/**
 * A reshape, read from a table_focus: its first frame lists each unit's records, its last frame
 * the one row each unit becomes, both carrying the row map. Anything else is not a reshape (null).
 */
export function reshapeOf(view: TableFocusView): ReshapeModel | null {
  if (view.story.length < 2) return null;
  const records = view.story[0]!;
  const last = view.story[view.story.length - 1]!;
  if (!records.rows.length || !last.rows.length || last.rows.length >= records.rows.length) return null;
  if (records.rows.some((r) => !r.unit) || last.rows.some((r) => !r.unit || !r.sources.length)) return null;
  const idColumn =
    records.columns.find((c) => records.rows.every((r) => String(r.values[c]) === r.unit)) ?? records.columns[0]!;
  const ordered = [...records.rows].sort((a, b) => a.row_id - b.row_id);
  const units: string[] = [];
  for (const r of ordered) if (!units.includes(r.unit!)) units.push(r.unit!);
  const kOf = new Map<number, number>();
  const seen = new Map<string, number>();
  for (const r of records.rows) {
    const k = seen.get(r.unit!) ?? 0;
    kOf.set(r.row_id, k);
    seen.set(r.unit!, k + 1);
  }
  const rows: ReshapeRow[] = ordered.map((r) => ({
    row: r.row_id,
    unit: r.unit!,
    u: units.indexOf(r.unit!),
    k: kOf.get(r.row_id)!,
    values: r.values as Record<string, unknown>,
  }));
  const settled: ReshapeModel["settled"] = {};
  for (const r of last.rows) {
    settled[r.unit!] = { row: r.row_id, values: r.values as Record<string, unknown>, sources: [...r.sources].sort((a, b) => a - b) };
  }
  const kind: Kind = last.rows.every((r) => r.sources.length === 1) ? "keep" : "combine";
  return {
    idColumn,
    columns: records.columns.filter((c) => c !== idColumn),
    rows,
    units,
    sizes: units.map((u) => rows.filter((r) => r.unit === u).length),
    settled,
    kind,
    nAffected: view.n_affected_columns,
  };
}

/** Is this record the one its unit keeps (first, last)? null when the method combines. */
export function keeps(m: ReshapeModel, r: ReshapeRow): boolean | null {
  if (m.kind === "combine") return null;
  return m.settled[r.unit]?.row === r.row;
}

export interface RowGeom {
  y: number;
  alpha: number;
  /** 0..1: the row is struck through and hatched (it leaves). */
  struck: number;
}

export interface FrameGeom {
  y: number;
  h: number;
  alpha: number;
}

export interface Layout {
  rows: RowGeom[];
  /** One soft frame per unit (the gathered group). */
  frames: FrameGeom[];
  /** The elisions between runs of shown rows that are far apart in the file. */
  gaps: { y: number; alpha: number; skipped: number }[];
  height: number;
}

export function layout(m: ReshapeModel, state: number): Layout {
  const { rows, sizes } = m;
  const U = m.units.length;
  const offsets: number[] = [];
  let acc = 0;
  for (const n of sizes) {
    offsets.push(acc);
    acc += n * RH + GAP;
  }
  const gathered = acc - GAP;

  // State 0: the file's own order, with an elision where the shown rows skip rows.
  const slot0: number[] = [];
  const gaps: Layout["gaps"] = [];
  let slot = 0;
  rows.forEach((r, i) => {
    const prev = rows[i - 1];
    if (prev && r.row - prev.row > 1) {
      gaps.push({ y: slot * RH, alpha: state === 0 ? 1 : 0, skipped: r.row - prev.row - 1 });
      slot += 1;
    }
    slot0.push(slot);
    slot += 1;
  });
  const height0 = slot * RH;

  const geoms: RowGeom[] = rows.map((r, i) => {
    const keep = keeps(m, r);
    const mid = offsets[r.u]! + ((sizes[r.u]! - 1) * RH) / 2;
    switch (state) {
      case 0:
        return { y: slot0[i]! * RH, alpha: 1, struck: 0 };
      case 1:
        return { y: offsets[r.u]! + r.k * RH, alpha: 1, struck: 0 };
      case 2:
        // The unit's records meet in the middle of their group: one row, the combination.
        if (keep === null) return { y: mid, alpha: r.k === 0 ? 1 : 0, struck: 0 };
        return { y: offsets[r.u]! + r.k * RH, alpha: keep ? 1 : 0.6, struck: keep ? 0 : 1 };
      default:
        if (keep === null) return { y: r.u * RH, alpha: r.k === 0 ? 1 : 0, struck: 0 };
        return { y: r.u * RH, alpha: keep ? 1 : 0, struck: keep ? 0 : 1 };
    }
  });

  const frames: FrameGeom[] = m.units.map((_, u) => {
    const block = sizes[u]! * RH;
    switch (state) {
      case 0: {
        const ys = rows.map((r, i) => (r.u === u ? slot0[i]! * RH : NaN)).filter((y) => !Number.isNaN(y));
        const top = Math.min(...ys);
        return { y: top, h: Math.max(...ys) + RH - top, alpha: 0 };
      }
      case 1:
        return { y: offsets[u]! - 3, h: block + 6, alpha: 1 };
      case 2:
        return m.kind === "combine"
          ? { y: offsets[u]! + ((sizes[u]! - 1) * RH) / 2 - 3, h: RH + 6, alpha: 1 }
          : { y: offsets[u]! - 3, h: block + 6, alpha: 1 };
      default:
        return { y: u * RH, h: RH, alpha: 0 };
    }
  });

  const height = state === 0 ? height0 : state === 3 ? U * RH : gathered;
  return { rows: geoms, frames, gaps, height };
}

const mix = (a: number, b: number, t: number) => a + (b - a) * t;

export function lerpLayout(a: Layout, b: Layout, t: number): Layout {
  if (t <= 0) return a;
  if (t >= 1) return b;
  return {
    rows: a.rows.map((r, i) => {
      const s = b.rows[i] ?? r;
      return { y: mix(r.y, s.y, t), alpha: mix(r.alpha, s.alpha, t), struck: mix(r.struck, s.struck, t) };
    }),
    frames: a.frames.map((f, i) => {
      const g = b.frames[i] ?? f;
      return { y: mix(f.y, g.y, t), h: mix(f.h, g.h, t), alpha: mix(f.alpha, g.alpha, t) };
    }),
    gaps: a.gaps.map((g, i) => ({ ...g, alpha: mix(g.alpha, b.gaps[i]?.alpha ?? 0, t) })),
    height: mix(a.height, b.height, t),
  };
}

/**
 * The values a row shows in a whole state: its own until it is combined, then its unit's
 * combination; a kept or struck record keeps its own values (it is that record).
 */
export function valuesAt(m: ReshapeModel, r: ReshapeRow, state: number): Record<string, unknown> {
  if (state < 2 || keeps(m, r) !== null) return r.values;
  return m.settled[r.unit]?.values ?? r.values;
}

/** Cells a whole state shows changed (tinted): a combined value that differs from the row's own. */
export function changedAt(m: ReshapeModel, r: ReshapeRow, state: number, col: string): boolean {
  if (state < 2 || keeps(m, r) !== null) return false;
  const v = m.settled[r.unit]?.values[col];
  return v !== undefined && v !== r.values[col];
}

/** The source rows a shown row stands for, as the working table's row map says ("0+1"). */
export function sourceRows(m: ReshapeModel, r: ReshapeRow, state: number): string {
  if (state < 2 || keeps(m, r) !== null) return String(r.row);
  return (m.settled[r.unit]?.sources ?? [r.row]).join("+");
}
