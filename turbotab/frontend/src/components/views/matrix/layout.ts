/**
 * The matrix view's geometry and color steps, pure. Both kinds take one gray ramp, your data as it
 * is (FOUNDATION §4): from a quiet neutral through --data-context to --data-fit, the darkest data
 * gray. A correlation steps by its size either way, in equal fifths of |r| (its sign is printed in
 * the cell and said on hover); a share of blanks steps from none in equal quarters. No categorical
 * slot is spent on a scale: those name entities. Labels keep the declared order and stay inside the
 * drawing.
 */
import { fmtPct } from "../common/frame";
import { gateRefusal } from "../common/gate";
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

export type Step = 0 | 1 | 2 | 3 | 4;

/** The fill of each step, every one from the data grays; step 0 is the neutral. */
export const RAMP: readonly string[] = [
  "color-mix(in oklab, var(--data-context) 25%, var(--canvas-line))",
  "color-mix(in oklab, var(--data-context) 70%, var(--canvas-line))",
  "color-mix(in oklab, var(--data-fit) 12%, var(--data-context))",
  "color-mix(in oklab, var(--data-fit) 58%, var(--data-context))",
  "var(--data-fit)",
];

/** A cell that could not be computed: no fill, an outline that holds 3:1 on the canvas in both themes. */
export const NOT_COMPUTED = "var(--canvas-muted)";

/** What each step covers, in the key's words: every bound named, so any cell's color reads back to a range. */
export const STEP_LABELS: Record<MatrixInput["kind"], readonly string[]> = {
  correlation: ["under .2", ".2–.4", ".4–.6", ".6–.8", ".8–1"],
  missingness: ["none", "up to 25%", "25–50%", "50–75%", "over 75%"],
};

/** The step of a value: |r| in equal fifths, or the share of blanks in equal quarters above none. */
export function stepOf(kind: MatrixInput["kind"], v: number): Step {
  const a = Math.abs(v);
  if (kind === "correlation") return a < 0.2 ? 0 : a < 0.4 ? 1 : a < 0.6 ? 2 : a < 0.8 ? 3 : 4;
  if (a === 0) return 0;
  return a <= 0.25 ? 1 : a <= 0.5 ? 2 : a <= 0.75 ? 3 : 4;
}

export function fillOf(kind: MatrixInput["kind"], v: number): string {
  return RAMP[stepOf(kind, v)]!;
}

/** The ink of a printed value: the canvas ink on the three lighter steps, the raised ink on the two darker (4.5:1 both themes). */
export function inkOfStep(step: Step): string {
  return step >= 3 ? "var(--raised-ink)" : "var(--canvas-ink)";
}

export function inkOf(kind: MatrixInput["kind"], v: number): string {
  return inkOfStep(stepOf(kind, v));
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
  const refused = gateRefusal(input.outcome, input.kind === "correlation" ? [...input.rows, ...input.cols] : [...input.cols, ...(input.groups_by ?? [])]);
  if (refused) return { empty: refused };
  if (input.symmetric && input.rows.length < 2) {
    return { empty: `Only ${input.rows[0]} is here, and a column has no pair to correlate with on its own.` };
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

  // Only the drawn cells count: a symmetric matrix's diagonal of ones is never drawn.
  if (cells.every((c) => c.v === null)) {
    return {
      empty:
        input.kind === "correlation"
          ? "No pair of these columns is recorded together often enough to correlate."
          : "No share of blanks could be computed for these columns.",
    };
  }

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
