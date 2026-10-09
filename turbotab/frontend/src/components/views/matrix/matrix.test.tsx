import { fireEvent, render, screen } from "@testing-library/react";
import { MATRIX_CORRELATION, MATRIX_MISSINGNESS, MATRIX_ONE_PAIR } from "../fixtures";
import { MATRIX, fillOf, fmtCell, inOrder, layoutMatrix, stepOf, type MatrixLayout } from "./layout";
import { MatrixView } from "./MatrixView";
import type { MatrixInput } from "./types";

const lay = (input: MatrixInput, w = 400): MatrixLayout => {
  const r = layoutMatrix(input, w);
  if ("empty" in r) throw new Error(r.empty);
  return r.layout;
};

describe("matrix scale", () => {
  it("places one pair's cell at known pixels with its labels inside the bounds", () => {
    const l = lay(MATRIX_ONE_PAIR);
    // the row label "fat_g" (5 × 6.1 px) + 10; the rotated column label rises 11 × 6.1 × √½ + 12
    expect(l.left).toBe(41);
    expect(l.top).toBe(60);
    expect(l.cell).toBe(MATRIX.maxCell);
    expect(l.cells).toEqual([{ r: 1, c: 0, x: 41, y: 60, v: 0.9, n: 600, printed: true }]);
    expect(l.width).toBe(400);
  });

  it("steps a correlation in equal fifths each way from a neutral 0", () => {
    expect([0, 0.19, 0.2, 0.39, 0.4, 0.6, 0.8, 1].map((v) => stepOf("correlation", v))).toEqual([0, 0, 1, 1, 2, 3, 4, 4]);
    expect(stepOf("correlation", -0.85)).toBe(4);
    expect(fillOf("correlation", 0.1)).toBe("var(--canvas-line)");
    expect(fillOf("correlation", -0.1)).toBe("var(--canvas-line)");
    expect(fillOf("correlation", 0.9)).toBe("var(--cat-5)");
    expect(fillOf("correlation", -0.9)).toBe("var(--cat-4)");
    expect(fillOf("correlation", 0.5)).toBe("color-mix(in oklab, var(--cat-5) 55%, var(--canvas-line))");
  });

  it("steps a share of blanks from neutral at none to gray at all", () => {
    expect([0, 0.01, 0.25, 0.26, 0.5, 0.75, 1].map((v) => stepOf("missingness", v))).toEqual([0, 1, 1, 2, 2, 3, 4]);
    expect(fillOf("missingness", 1)).toBe("var(--canvas-muted)");
    expect(fmtCell("missingness", 0.638889)).toBe("64%");
    expect(fmtCell("correlation", -0.0696)).toBe("−.07");
    expect(fmtCell("correlation", 0.891876)).toBe(".89");
  });

  it("draws a correlation's lower triangle only, in the declared order", () => {
    const l = lay(MATRIX_CORRELATION, 700);
    const n = MATRIX_CORRELATION.rows.length;
    expect(l.cells).toHaveLength((n * (n - 1)) / 2);
    expect(l.cells.every((c) => c.c < c.r)).toBe(true);
    expect(l.rows.map((r) => r.full)).toEqual(MATRIX_CORRELATION.rows.slice(1));
    expect(l.cols.map((c) => c.full)).toEqual(MATRIX_CORRELATION.cols.slice(0, -1));
    // values are printed selectively: only from |r| = 0.5 up
    for (const c of l.cells) expect(c.printed).toBe(c.v !== null && Math.abs(c.v) >= 0.5 && l.cell >= MATRIX.printCell);
    expect(l.cells.some((c) => c.printed)).toBe(true);
    expect(l.cells.some((c) => !c.printed)).toBe(true);
  });

  it("reorders by a declared order, values with their labels", () => {
    const m = inOrder(MATRIX_ONE_PAIR, ["fat_g"], "by clustering");
    expect(m.rows).toEqual(["fat_g", "energy_kcal"]);
    expect(m.values[0]).toEqual([1, 0.9]);
    expect(m.order).toBe("by clustering");
  });

  it("draws the first 40 columns of a wider matrix and says so", () => {
    const labels = Array.from({ length: 45 }, (_, i) => `c${i}`);
    const values = labels.map((_, i) => labels.map((__, j) => (i === j ? 1 : 0.1)));
    const l = lay({ kind: "correlation", rows: labels, cols: labels, order: "as in your file", symmetric: true, values }, 500);
    expect(l.capped).toEqual({ shown: 40, of: 45 });
    expect(l.cell).toBe(MATRIX.minCell);
    expect(l.width).toBeGreaterThan(500); // it scrolls in its own container
  });

  it("keeps the missingness matrix whole: every group and column", () => {
    const l = lay(MATRIX_MISSINGNESS, 700);
    expect(l.cells).toHaveLength(MATRIX_MISSINGNESS.rows.length * MATRIX_MISSINGNESS.cols.length);
  });
});

describe("matrix states", () => {
  it("says why in one line when one column has no pair", () => {
    render(<MatrixView input={{ ...MATRIX_ONE_PAIR, rows: ["fat_g"], cols: ["fat_g"], values: [[1]], n: undefined }} />);
    expect(screen.getByRole("note")).toHaveTextContent("Only fat_g is here");
    expect(screen.queryByRole("img")).toBeNull();
  });

  it("says why when no value could be computed", () => {
    const r = layoutMatrix({ ...MATRIX_ONE_PAIR, values: [[null, null], [null, null]] }, 400);
    expect(r).toEqual({ empty: "No pair of these columns is recorded together often enough to correlate." });
  });

  it("offers its table alternative", () => {
    render(<MatrixView input={MATRIX_CORRELATION} />);
    fireEvent.click(screen.getByRole("button", { name: "Show as a table" }));
    const table = screen.getByRole("table");
    expect(table).toHaveTextContent("0.89");
    expect(screen.getAllByRole("row")).toHaveLength(1 + MATRIX_CORRELATION.rows.length);
  });
});
