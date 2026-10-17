import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { substitutionChoice, substitutionCompare, substitutionSealed } from "./lab/curves.fixtures";
import { curveFromExplain, curveFromSubstitution, pastPalette } from "./curveAdapters";
import { CurveView } from "./CurveView";
import type { CurveLine } from "./curve";
import prediction from "../../mocks/fixtures/m3-nhanes-prediction.json";
import type { InputCurves, SubstitutionArtifact } from "../../api/m3-types";

describe("the substitution artifact as a curve (captured NHANES Predict)", () => {
  it("draws each k's own rows in gray and the same rows at every k in indigo, with the stop said", () => {
    expect(substitutionChoice.lines.map((l) => l.role)).toEqual(["now", "choice"]);
    expect(substitutionChoice.lines[1]!.y.slice(0, 3)).toEqual([0, -1.84247, -1.52805]);
    expect(substitutionChoice.stop).toEqual({ x: 300, why: "The curve stops at 300 kcal, where 39% of rows stay within the range observed." });
    expect(substitutionChoice.support?.share[2]).toBeCloseTo(0.7187, 3);
  });

  it("compares the models in the palette's order", () => {
    expect(substitutionCompare.lines.map((l) => [l.label, l.slot])).toEqual([
      ["Linear model", 1],
      ["Boosted trees", 2],
    ]);
  });

  it("draws the support and the stop beside compared models only when every model shares them", () => {
    expect(substitutionCompare.stop?.x).toBe(300); // both models stop at 300 on the same support
    const art = (prediction as unknown as { artifacts: { substitution: { base: SubstitutionArtifact } } }).artifacts.substitution.base;
    const differ = { ...art, models: art.models.map((m, i) => (i === 1 ? { ...m, stopped_at: 400 } : m)) };
    const c = curveFromSubstitution(differ, { outcome: "glucose" });
    expect(c.stop).toBeNull();
    expect(c.support).toBeNull();
  });

  it("before Fit carries no estimate", () => {
    expect(substitutionSealed.lines).toEqual([]);
    expect(substitutionSealed.sealed).toMatch(/after Fit/);
  });
});

// ── the explain artifact's curves (W1-G) ─────────────────────────────────────

describe("an explain artifact's input curves as a curve, keeping the support mask", () => {
  // By construction, as turbotab/core/models/explain.py serves them: each family's values at the
  // grid's edges, null at an edge that touches no supported segment; segment k spans grid[k] to
  // grid[k + 1] and holds counts[k] rows.
  const ic = (patch: Partial<InputCurves> = {}): InputCurves => ({
    input: "waist",
    sources: ["waist"],
    role: "predictor",
    method: "ale",
    grid: [60, 80, 90, 130, 140],
    counts: [40, 40, 2, 30],
    supported: [true, true, false, true],
    unit: "cm",
    curves: [
      { family: "linear", label: "Linear model", drawn: true, values: [-3, -1, 0.5, 4, 6], reason: null },
      { family: "ridge", label: "Ridge", drawn: false, values: [], reason: "Ridge draws no curve." },
    ],
    ...patch,
  });
  const regression = { outcome: "glucose", task: "regression" };
  const defined = (l: CurveLine) => l.x.flatMap((x, i) => (l.y[i] === null || l.y[i] === undefined ? [] : [[x, l.y[i]]]));

  it("draws each drawn family's served values at the grid, and no line for a family not drawn", () => {
    const c = curveFromExplain(ic(), regression);
    expect(c.lines.map((l) => [l.key, l.label, l.role, l.slot])).toEqual([["linear", "Linear model", "series", 1]]);
    expect(defined(c.lines[0]!)).toEqual([[60, -3], [80, -1], [90, 0.5], [130, 4], [140, 6]]);
    expect(c.sealed).toBeNull();
    expect(c.zero).toBe(true);
    expect(c.xLabel).toBe("waist (cm)");
    expect(c.yLabel).toBe("Change in predicted glucose");
  });

  it("breaks the line across a segment with too few rows, though both its edges carry a value", () => {
    const { container } = render(<CurveView data={curveFromExplain(ic(), regression)} />);
    // Two runs, 60 to 90 and 130 to 140: nothing is drawn across 90 to 130, which holds 2 rows.
    expect(container.querySelectorAll('[data-line="linear"] [data-mark="line"]')).toHaveLength(2);
    // The table alternative lists each served value once, at its own grid value.
    const rows = [...screen.getByTestId("table-alternative").querySelectorAll("tbody tr")].map((r) => [...r.children].map((c) => c.textContent));
    expect(rows).toEqual([
      ["60", "−3"],
      ["80", "−1"],
      ["90", "0.5"],
      ["130", "4"],
      ["140", "6"],
    ]);
  });

  it("draws one unbroken line where every segment holds enough rows", () => {
    const c = curveFromExplain(ic({ supported: [true, true, true, true], counts: [40, 40, 9, 30] }), regression);
    const { container } = render(<CurveView data={c} />);
    expect(container.querySelectorAll('[data-line="linear"] [data-mark="line"]')).toHaveLength(1);
    expect(c.lines[0]!.x).toEqual([60, 80, 90, 130, 140]);
  });

  it("keeps an edge the engine masked, and gives a family its color by its place among all", () => {
    const c = curveFromExplain(
      ic({
        supported: [false, true, true, true],
        curves: [
          { family: "linear", label: "Linear model", drawn: false, values: [], reason: "Linear draws no curve." },
          { family: "ridge", label: "Ridge", drawn: true, values: [null, 1, 2, 3, 4], reason: null },
        ],
      }),
      regression,
    );
    expect(c.lines.map((l) => [l.key, l.slot])).toEqual([["ridge", 2]]);
    expect(defined(c.lines[0]!)).toEqual([[80, 1], [90, 2], [130, 3], [140, 4]]);
    expect(c.lines[0]!.y[0]).toBeNull();
  });

  it("draws the families in the palette's places and names each drawn family past them", () => {
    const fam = (family: string, label: string, drawn = true) => ({ family, label, drawn, values: drawn ? [-3, -1, 0.5, 4, 6] : [], reason: drawn ? null : `${label} draws no curve.` });
    const six = ic({
      curves: [
        fam("linear", "Linear model"),
        fam("ridge", "Ridge"),
        fam("elastic_net", "Elastic net"),
        fam("huber", "Huber regression"),
        fam("random_forest", "Random forest"),
        fam("boosted_trees", "Boosted trees"),
      ],
    });
    const c = curveFromExplain(six, regression);
    // Five colors (--cat-1 to --cat-5): the sixth family has no place in them.
    expect(c.lines.map((l) => [l.key, l.slot])).toEqual([["linear", 1], ["ridge", 2], ["elastic_net", 3], ["huber", 4], ["random_forest", 5]]);
    expect(pastPalette(six)).toBe(
      "Not drawn here: Boosted trees. Each curve takes the color of its family's place among the families explained, and there are 5 colors.",
    );
    // Seven, with one not drawn by the engine inside the palette's places: its place still counts,
    // and the engine's own reason, not this one, says why it has no line.
    const seven = ic({ curves: [fam("linear", "Linear model"), fam("ridge", "Ridge", false), ...six.curves.slice(2), fam("svm", "Support vector machine")] });
    expect(curveFromExplain(seven, regression).lines.map((l) => l.key)).toEqual(["linear", "elastic_net", "huber", "random_forest"]);
    expect(pastPalette(seven)).toBe(
      "Not drawn here: Boosted trees and Support vector machine. Each curve takes the color of its family's place among the families explained, and there are 5 colors.",
    );
    // A family past the palette the engine did not draw is the engine's to explain: none named here.
    expect(pastPalette(ic({ curves: [...six.curves.slice(0, 5), fam("boosted_trees", "Boosted trees", false)] }))).toBeNull();
    expect(pastPalette(ic())).toBeNull();
  });

  it("says the model's own scale where the prediction is not the outcome's value", () => {
    expect(curveFromExplain(ic(), { outcome: "progression", task: "binary" }).yLabel).toBe("Change on the model's own scale");
  });
});
