import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { substitutionChoice, substitutionCompare, substitutionSealed } from "./lab/fixtures";
import { curveFromSubstitution } from "./adapters";
import prediction from "../../mocks/fixtures/m3-nhanes-prediction.json";
import type { SubstitutionArtifact } from "../../api/m3-types";
import { LabViews, scopedTokens } from "./lab/LabViews";
import { entries } from "./lab/curves.lab";

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

describe("/lab/views", () => {
  it("renders every entry in the light and the dark theme", () => {
    const { container } = render(<LabViews entries={entries} />);
    expect(container.querySelectorAll("[data-lab-entry]")).toHaveLength(entries.length);
    expect(container.querySelectorAll('[data-views-theme="light"]')).toHaveLength(entries.length);
    expect(container.querySelectorAll('[data-views-theme="dark"]')).toHaveLength(entries.length);
  });

  it("scopes both token blocks to an element", () => {
    const css = ':root { --canvas: #fff; } @media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --canvas: #111; } } :root[data-theme="dark"] { --canvas: #000; }';
    expect(scopedTokens(css)).toBe('[data-views-theme="light"]{ --canvas: #fff; ;color-scheme:light}[data-views-theme="dark"]{ --canvas: #000; }');
  });
});
