/**
 * The matrix view's geometry and color steps, pure. A correlation takes a diverging scale with
 * equal steps per arm through a neutral midpoint at 0 (steel for moving opposite, clay for moving
 * together); a share of blanks takes one gray ramp from the same neutral at none (your data as it
 * is, FOUNDATION §4). Labels keep the declared order and stay inside the drawing.
 */
import { fmtPct } from "../common/frame";
import type { MatrixInput } from "./types";

export const MATRIX = {
  maxCols: 40,
  /** below this, labels rotated 45° at 11 px would touch */
  minCell: 16,
  maxCell: 44,
  /** a value is printed only in a cell at least this wide */
  printCell: 30,
  charPx: 6.1,
  maxLabel: 22,
  gap: 2,
} as const;

/** The color mix of each step, from the neutral midpoint (0) to the pole (4). */
export const MIX = [0, 30, 55, 78, 100] as const;

export type Step = 0 | 1 | 2 | 3 | 4;

/** The step of a value: |r| in equal fifths, or the share of blanks in equal quarters above none. */
export function stepOf(kind: MatrixInput["kind"], v: number): Step {
  const a = Math.abs(v);
  if (kind === "correlation") return a < 0.2 ? 0 : a < 0.4 ? 1 : a < 0.6 ? 2 : a < 0.8 ? 3 : 4;
  if (a === 0) return 0;
  return a <= 0.25 ? 1 : a <= 0.5 ? 2 : a <= 0.75 ? 3 : 4;
}

export function fillOf(kind: MatrixInput["kind"], v: number): string {
  const step = stepOf(kind, v);
  if (step === 0) return "var(--canvas-line)";
  const pole = kind === "missingness" ? "var(--canvas-muted)" : v < 0 ? "var(--cat-4)" : "var(--cat-5)";
  return step === 4 ? pole : `color-mix(in oklab, ${pole} ${MIX[step]}%, var(--canvas-line))`;
}

/** A cell's text wears an ink token: the raised ink on the strongest steps, the canvas ink below. */
export function inkOf(kind: MatrixInput["kind"], v: number): string {
  return stepOf(kind, v) >= (kind === "missingness" ? 3 : 4) ? "var(--raised-ink)" : "var(--canvas-ink)";
}

/** A value as printed in a cell: r to two places without the leading zero, a share as a percent. */
export function fmtCell(kind: MatrixInput["kind"], v: number): string {
  if (kind === "missingness") return fmtPct(v);
  if (Math.abs(v) < 0.005) return ".00";
  return v.toFixed(2).replace(/^(-?)0\./, "$1.").replace("-", "−");
}

/**
 * The same matrix with its labels in a declared order (`orderName` says which, in plain words):
 * labels the order names come first, in its order; the rest keep their places after them. For a
 * symmetric matrix the columns follow the rows.
 */
export function inOrder(input: MatrixInput, order: string[], orderName: string): MatrixInput {
  const rank = (labels: string[]) => {
    const pos = new Map(order.map((s, i) => [s, i]));
    return labels
      .map((s, i) => ({ i, k: pos.get(s) ?? order.length + i }))
      .sort((a, b) => a.k - b.k)
      .map((x) => x.i);
  };
  const ri = rank(input.rows);
  const ci = input.symmetric ? ri : rank(input.cols);
  const pick = <T,>(m: T[][] | undefined) => m && ri.map((i) => ci.map((j) => m[i]![j]!));
  return {
    ...input,
    rows: ri.map((i) => input.rows[i]!),
    cols: ci.map((j) => input.cols[j]!),
    values: pick(input.values)!,
    n: pick(input.n),
    order: orderName,
  };
}

export interface MatrixCell {
  r: number;
  c: number;
  x: number;
  y: number;
  v: number | null;
  n: number | null;
  printed: boolean;
}

export interface MatrixLayout {
  width: number;
  height: number;
  left: number;
  top: number;
  cell: number;
  rows: { i: number; label: string; full: string; y: number }[];
  cols: { j: number; label: string; full: string; x: number }[];
  cells: MatrixCell[];
  capped: { shown: number; of: number } | null;
}

export type MatrixResult = { empty: string } | { layout: MatrixLayout };

const cut = (s: string) => (s.length <= MATRIX.maxLabel ? s : `${s.slice(0, MATRIX.maxLabel - 1)}…`);

export function layoutMatrix(input: MatrixInput, width: number): MatrixResult {
  const { values } = input;
  if (!input.rows.length || !input.cols.length) return { empty: "No columns to compare, so there is no matrix to draw." };
  if (values.length !== input.rows.length || values.some((row) => row.length !== input.cols.length)) {
    return { empty: "The matrix's values do not match its labels, so nothing is drawn." };
  }
  if (input.symmetric && input.rows.length < 2) {
    return { empty: `Only ${input.rows[0]} is here, and a column has no pair to correlate with on its own.` };
  }
  if (values.every((row) => row.every((x) => x === null || !Number.isFinite(x)))) {
    return {
      empty:
        input.kind === "correlation"
          ? "No pair of these columns is recorded together often enough to correlate."
          : "No share of blanks could be computed for these columns.",
    };
  }

  // The declared order, capped; a symmetric matrix drops its first row and last column (the lower
  // triangle, without the diagonal of ones).
  const cap = (n: number) => Math.min(n, MATRIX.maxCols);
  const nr = cap(input.rows.length);
  const nc = cap(input.cols.length);
  const rowIdx = Array.from({ length: nr }, (_, i) => i).slice(input.symmetric ? 1 : 0);
  const colIdx = Array.from({ length: nc }, (_, j) => j).slice(0, input.symmetric ? nc - 1 : nc);
  const capped = input.rows.length > nr || input.cols.length > nc ? { shown: Math.max(nr, nc), of: Math.max(input.rows.length, input.cols.length) } : null;

  const rowLabels = rowIdx.map((i) => cut(input.rows[i]!));
  const colLabels = colIdx.map((j) => cut(input.cols[j]!));
  const left = Math.ceil(Math.max(...rowLabels.map((s) => s.length)) * MATRIX.charPx) + 10;
  const rise = (s: string) => s.length * MATRIX.charPx * Math.SQRT1_2;
  const top = Math.ceil(Math.max(...colLabels.map(rise))) + 12;
  // Rotated column labels run up and to the right: leave room for the one reaching furthest.
  const overhang = (cell: number) => Math.max(0, ...colLabels.map((s, k) => rise(s) - (colLabels.length - 1 - k) * cell - cell / 2)) + 4;
  let cell = (width - left - overhang(MATRIX.maxCell)) / colLabels.length;
  cell = (width - left - overhang(cell)) / colLabels.length;
  cell = Math.max(MATRIX.minCell, Math.min(MATRIX.maxCell, Math.floor(cell)));
  const totalW = Math.max(width, Math.ceil(left + cell * colLabels.length + overhang(cell)));
  const height = Math.ceil(top + cell * rowLabels.length + 4);

  const labelAt = input.label_at ?? 0.5;
  const cells: MatrixCell[] = [];
  rowIdx.forEach((i, ri) => {
    colIdx.forEach((j, cj) => {
      if (input.symmetric && j >= i) return;
      const raw = values[i]![j]!;
      const val = raw !== null && Number.isFinite(raw) ? raw : null;
      cells.push({
        r: i,
        c: j,
        x: left + cj * cell,
        y: top + ri * cell,
        v: val,
        n: input.n?.[i]?.[j] ?? null,
        printed: val !== null && Math.abs(val) >= labelAt && cell >= MATRIX.printCell,
      });
    });
  });

  return {
    layout: {
      width: totalW,
      height,
      left,
      top,
      cell,
      rows: rowIdx.map((i, ri) => ({ i, label: rowLabels[ri]!, full: input.rows[i]!, y: top + ri * cell + cell / 2 })),
      cols: colIdx.map((j, cj) => ({ j, label: colLabels[cj]!, full: input.cols[j]!, x: left + cj * cell + cell / 2 })),
      cells,
      capped,
    },
  };
}
