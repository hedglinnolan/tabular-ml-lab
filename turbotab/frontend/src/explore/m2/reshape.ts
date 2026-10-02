/**
 * The reshape storyboard's geometry — pure, so the states can be checked without a browser.
 *
 * Four real, labeled states (BLUEPRINT §11.1): 0 your data now · 1 gathered by the unit ·
 * 2 combined (mean, change) or kept (first, last) · 3 settled, one row per unit. The player moves
 * between them; a frame between two states is motion, never data, so every number on screen is
 * read from a whole state (`valuesAt`) and only positions, widths and opacities glide (`lerpLayout`).
 *
 * Identity is the row (DESIGN_LANGUAGE §05.2): a row keeps its element from the file to the
 * settled table. A combined row is its unit's rows meeting in one place — a true correspondence —
 * and a dropped row is struck and folds away where it stood, never a neighbor taking its slot.
 */
import type { Cell, MethodFixture, ReshapeFixture, WindowRow } from "./types";

export const RH = 22;
export const GAP = 6;

export type Kind = "combine" | "keep";

export function kindOf(m: MethodFixture): Kind {
  return Object.values(m.units).some((u) => u.kept_row !== null) ? "keep" : "combine";
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
  /** The elision between two runs of window rows that are far apart in the file. */
  gaps: { y: number; alpha: number; skipped: number }[];
  height: number;
  /** Width of each column by name (record columns fold to 0 when combined). */
  widths: Record<string, number>;
}

/** Window rows in file order, with their unit index and slot. */
export function windowRows(fx: ReshapeFixture): (WindowRow & { u: number })[] {
  const units = fx.window.units;
  return [...fx.window.rows].sort((a, b) => a.row - b.row).map((r) => ({ ...r, u: units.indexOf(r.unit) }));
}

/** Is this row the one its unit keeps (first / last)? null when the method combines. */
export function keeps(m: MethodFixture, r: WindowRow): boolean | null {
  const kept = m.units[r.unit]?.kept_row;
  return kept === null || kept === undefined ? null : kept === r.row;
}

export function layout(
  fx: ReshapeFixture,
  m: MethodFixture,
  state: number,
  baseWidths: Record<string, number>,
): Layout {
  const rows = windowRows(fx);
  const P = fx.per_unit;
  const U = fx.window.units.length;
  const block = P * RH + GAP;
  const widths: Record<string, number> = { ...baseWidths };
  if (state >= 2) for (const c of m.folds) widths[c] = 0;

  // State 0: the file's own order, with an elision where the window skips rows.
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
    switch (state) {
      case 0:
        return { y: slot0[i]! * RH, alpha: 1, struck: 0 };
      case 1:
        return { y: r.u * block + r.k * RH, alpha: 1, struck: 0 };
      case 2:
        if (keep === null) {
          // The unit's rows meet in the middle of their group: one row, the combination.
          return { y: r.u * block + ((P - 1) * RH) / 2, alpha: r.k === 0 ? 1 : 0, struck: 0 };
        }
        return { y: r.u * block + r.k * RH, alpha: keep ? 1 : 0.6, struck: keep ? 0 : 1 };
      default:
        if (keep === null) return { y: r.u * RH, alpha: r.k === 0 ? 1 : 0, struck: 0 };
        return { y: r.u * RH, alpha: keep ? 1 : 0, struck: keep ? 0 : 1 };
    }
  });

  const frames: FrameGeom[] = fx.window.units.map((_, u) => {
    switch (state) {
      case 0: {
        // Around the unit's rows where they sit in the file (only meaningful when adjacent).
        const ys = rows.filter((r) => r.u === u).map((r) => slot0[rows.indexOf(r)]! * RH);
        const top = Math.min(...ys);
        const bottom = Math.max(...ys) + RH;
        return { y: top, h: bottom - top, alpha: 0 };
      }
      case 1:
        return { y: u * block - 3, h: P * RH + 6, alpha: 1 };
      case 2: {
        const combine = rows.some((r) => r.u === u && keeps(m, r) === null);
        return combine
          ? { y: u * block + ((P - 1) * RH) / 2 - 3, h: RH + 6, alpha: 1 }
          : { y: u * block - 3, h: P * RH + 6, alpha: 1 };
      }
      default:
        return { y: u * RH, h: RH, alpha: 0 };
    }
  });

  const height = state === 0 ? height0 : state === 3 ? U * RH : U * block - GAP;
  return { rows: geoms, frames, gaps, height, widths };
}

const mix = (a: number, b: number, t: number) => a + (b - a) * t;

export function lerpLayout(a: Layout, b: Layout, t: number): Layout {
  if (t <= 0) return a;
  if (t >= 1) return b;
  const widths: Record<string, number> = {};
  for (const k of Object.keys(a.widths)) widths[k] = mix(a.widths[k]!, b.widths[k] ?? a.widths[k]!, t);
  return {
    rows: a.rows.map((r, i) => {
      const s = b.rows[i]!;
      return { y: mix(r.y, s.y, t), alpha: mix(r.alpha, s.alpha, t), struck: mix(r.struck, s.struck, t) };
    }),
    frames: a.frames.map((f, i) => {
      const g = b.frames[i]!;
      return { y: mix(f.y, g.y, t), h: mix(f.h, g.h, t), alpha: mix(f.alpha, g.alpha, t) };
    }),
    gaps: a.gaps.map((g, i) => ({ ...g, alpha: mix(g.alpha, b.gaps[i]?.alpha ?? 0, t) })),
    height: mix(a.height, b.height, t),
    widths,
  };
}

/**
 * The values a row shows in a whole state: its own until it is combined, then its unit's
 * combination; a kept or struck row keeps its own record (it is that record).
 */
export function valuesAt(m: MethodFixture, r: WindowRow, state: number): Record<string, Cell> {
  if (state < 2) return r.values;
  if (keeps(m, r) !== null) return r.values;
  return m.units[r.unit]?.values ?? r.values;
}

/** Cells a whole state shows changed (tinted): a combined value that differs from the row's own. */
export function changedAt(m: MethodFixture, r: WindowRow, state: number, col: string): boolean {
  if (state < 2 || keeps(m, r) !== null) return false;
  const v = m.units[r.unit]?.values[col];
  return v !== undefined && v !== r.values[col];
}

/** The rows of the source table a shown row stands for, as the working table's row map says. */
export function sourceRows(fx: ReshapeFixture, m: MethodFixture, r: WindowRow, state: number): string {
  if (state < 2) return String(r.row);
  const keep = keeps(m, r);
  if (keep !== null) return String(r.row);
  return fx.window.rows
    .filter((x) => x.unit === r.unit)
    .map((x) => x.row)
    .sort((a, b) => a - b)
    .join("+");
}
