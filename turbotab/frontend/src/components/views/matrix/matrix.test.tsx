import { fireEvent, render, screen, act } from "@testing-library/react";
import { readFileSync } from "node:fs";
import { contrast, resolveColor, themeTokens } from "../common/contrast";
import { MATRIX_CORRELATION, MATRIX_MISSINGNESS, MATRIX_ONE_PAIR } from "../fixtures";
import { MATRIX, NOT_COMPUTED, RAMP, STEP_LABELS, fillOf, fmtCell, inOrder, inkOfStep, layoutMatrix, stepOf, type MatrixLayout, type Step } from "./layout";
import { MatrixView } from "./MatrixView";
import type { MatrixInput } from "./types";

/** Opens the view's table alternative, the one disclosure under the drawing. */
function openTable() {
  const d = screen.getByTestId("table-alternative") as HTMLDetailsElement;
  act(() => {
    d.open = true;
    d.dispatchEvent(new Event("toggle"));
  });
}

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

  it("steps a correlation by its size either way, in equal fifths, on the data grays", () => {
    expect([0, 0.19, 0.2, 0.39, 0.4, 0.6, 0.8, 1].map((v) => stepOf("correlation", v))).toEqual([0, 0, 1, 1, 2, 3, 4, 4]);
    expect(stepOf("correlation", -0.85)).toBe(4);
    expect(fillOf("correlation", 0.1)).toBe(RAMP[0]);
    expect(fillOf("correlation", 0.9)).toBe("var(--data-fit)");
    expect(fillOf("correlation", -0.9)).toBe("var(--data-fit)");
    expect(fillOf("correlation", 0.5)).toBe(RAMP[2]);
    // no categorical slot is spent on a scale, and no text token fills a cell
    for (const f of RAMP) expect(f).not.toMatch(/--cat-|--canvas-muted|--canvas-ink/);
  });

  it("steps a share of blanks from none in quarters", () => {
    expect([0, 0.01, 0.25, 0.26, 0.5, 0.75, 1].map((v) => stepOf("missingness", v))).toEqual([0, 1, 1, 2, 2, 3, 4]);
    expect(fillOf("missingness", 1)).toBe("var(--data-fit)");
    expect(fmtCell("missingness", 0.638889)).toBe("64%");
    expect(fmtCell("correlation", -0.0696)).toBe("−.07");
    expect(fmtCell("correlation", 0.891876)).toBe(".89");
  });

  it("names every step's bounds in its key, matching the steps", () => {
    // the first value inside each named range falls on that step
    expect([0.1, 0.2, 0.4, 0.6, 0.8].map((v) => stepOf("correlation", v))).toEqual([0, 1, 2, 3, 4]);
    expect([0, 0.25, 0.5, 0.75, 0.76].map((v) => stepOf("missingness", v))).toEqual([0, 1, 2, 3, 4]);
    render(<MatrixView input={MATRIX_MISSINGNESS} />);
    const key = document.querySelector("[aria-label^='Scale']")!;
    for (const label of STEP_LABELS.missingness) expect(key).toHaveTextContent(label);
  });

  it.each(["light", "dark"] as const)("prints every value at 4.5:1 and marks 'not computed' at 3:1 in %s", (theme) => {
    const tokens = themeTokens(readFileSync(`${process.cwd()}/src/explore/calm-kit/tokens.css`, "utf8"), theme);
    const canvas = resolveColor("var(--canvas)", tokens);
    const lum = RAMP.map((f) => contrast(resolveColor(f, tokens), resolveColor("var(--canvas)", tokens)));
    RAMP.forEach((f, k) => {
      expect(contrast(resolveColor(f, tokens), resolveColor(inkOfStep(k as Step), tokens)), `step ${k}`).toBeGreaterThanOrEqual(4.5);
    });
    // the steps move away from the canvas, the neutral still seen
    for (let k = 1; k < lum.length; k++) expect(lum[k]!).toBeGreaterThan(lum[k - 1]!);
    expect(lum[0]!).toBeGreaterThanOrEqual(1.3);
    expect(contrast(resolveColor(NOT_COMPUTED, tokens), canvas)).toBeGreaterThanOrEqual(3);
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
    const l = lay({ kind: "correlation", rows: labels, cols: labels, order: "as in your file", outcome: null, symmetric: true, values }, 500);
    expect(l.capped).toEqual({ shown: 40, of: 45 });
    expect(l.cell).toBe(MATRIX.minCell);
    expect(l.width).toBeGreaterThan(500); // it scrolls in its own container
  });

  it("keeps the correlation fixture's columns in the file's own order, as its caption says", () => {
    const header = readFileSync(`${process.cwd()}/../sample_data/dietary_recalls.csv`, "utf8").split("\n")[0]!.split(",");
    expect(MATRIX_CORRELATION.order).toBe("as in your file");
    expect(header.filter((c) => MATRIX_CORRELATION.rows.includes(c))).toEqual(MATRIX_CORRELATION.rows);
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
  });

  it("says why when no value could be computed", () => {
    const r = layoutMatrix({ ...MATRIX_ONE_PAIR, values: [[null, null], [null, null]] }, 400);
    expect(r).toEqual({ empty: "No pair of these columns is recorded together often enough to correlate." });
  });

  it("says why when only the diagonal was computed: the drawn cells decide", () => {
    const r = layoutMatrix({ ...MATRIX_ONE_PAIR, values: [[1, null], [null, 1]] }, 400);
    expect(r).toEqual({ empty: "No pair of these columns is recorded together often enough to correlate." });
  });

  it("refuses the outcome before its gate, as a column or a grouping, in one line", () => {
    const corr = { ...MATRIX_ONE_PAIR, rows: ["energy_kcal", "hba1c"], cols: ["energy_kcal", "hba1c"] };
    expect(layoutMatrix(corr, 400)).toEqual({ empty: expect.stringMatching(/^hba1c is the outcome, .* not drawn yet\.$/) });
    expect("layout" in layoutMatrix({ ...corr, outcome: { name: "hba1c", gate_open: true } }, 400)).toBe(true);
    const blanks = { ...MATRIX_MISSINGNESS, cols: [...MATRIX_MISSINGNESS.cols.slice(0, -1), "responder"] };
    expect("empty" in layoutMatrix(blanks, 400)).toBe(true);
    expect("empty" in layoutMatrix({ ...MATRIX_MISSINGNESS, groups_by: ["responder"] }, 400)).toBe(true);
    render(<MatrixView input={corr} />);
    expect(screen.getByRole("note")).toHaveTextContent("hba1c is the outcome");
  });

  it("says how a selection of columns was chosen", () => {
    render(<MatrixView input={MATRIX_MISSINGNESS} />);
    expect(screen.getByText(/^The share of blanks, in the 12 of 392 columns blank most often\. Order: /)).toBeInTheDocument();
  });

  it("offers its table alternative: one line per pair, with r as drawn and its rows", () => {
    render(<MatrixView input={MATRIX_CORRELATION} />);
    openTable();
    const n = MATRIX_CORRELATION.rows.length;
    const rows = screen.getAllByRole("row");
    expect(rows).toHaveLength(1 + (n * (n - 1)) / 2);
    expect(rows[0]).toHaveTextContent(/^Pair\s*r\s*Rows$/);
    const pair = rows.find((r) => r.textContent?.startsWith("protein_g and energy_kcal"))!;
    expect(pair).toHaveTextContent(/\.89/);
    expect(pair).not.toHaveTextContent("0.89");
    expect(pair).toHaveTextContent(/600$/);
    for (const r of rows.slice(1)) expect(r.querySelectorAll("td")[1]!.textContent).not.toBe("");
  });

  it("reads each cell by the keyboard", () => {
    render(<MatrixView input={MATRIX_ONE_PAIR} />);
    const svg = screen.getByRole("img");
    expect(svg).toHaveAttribute("tabindex", "0");
    fireEvent.keyDown(svg, { key: "ArrowRight" });
    expect(screen.getByRole("status")).toHaveTextContent("fat_g and energy_kcal · r .90 · 600 rows");
    fireEvent.keyDown(svg, { key: "Escape" });
    expect(screen.queryByRole("status")).toBeNull();
  });
});
