/**
 * The coefficients' note, and the saved forest figure's caption, claim only what the fit drew (the
 * ruling of 2026-10-23 on the shrunk effect). Under inference a family whose fit gives no
 * coefficient table (ridge: its fit serves `inference: null`, no word of how intervals were made)
 * draws no interval and reports no effect, so neither claims one: no "95%", no lines, no change per
 * unit read as an effect. The note says why in plain words, from what the family declares it gives
 * under inference (GET /models' `inference_table`): a shrunk table with no intervals is shrunk
 * toward zero by its penalty; with no such declaration known, it says only that there are no
 * standard errors. A family with a table keeps the fit's own caption of its intervals; under
 * prediction nothing changes.
 */
import { cleanup, render, screen } from "@testing-library/react";
import { beforeAll, describe, expect, it } from "vitest";
import type { Coefficient, FittedModel } from "../../../api/m1-stage-types";
import type { FamilyInfo } from "../../../api/m3-types";
import { Coefficients, declaredTables, forestCaption, type DeclaredTable } from "./Coefficients";

class NoResize {
  observe() {}
  unobserve() {}
  disconnect() {}
}

const row = (estimate: number, low: number | null, high: number | null): Coefficient =>
  ({ feature: "walks_daily", estimate, ci_low: low, ci_high: high, p: low === null ? null : 0.0003 }) as Coefficient;

// A caption of the intervals as a fit with a table serves one (``inference.caption``): quoted whole.
const HC3 = "95% confidence intervals from HC3 (heteroskedasticity-robust) standard errors, t(116).";
const ridge = { family: "ridge", label: "Ridge", inference: null } as unknown as FittedModel;
const net = { family: "elastic_net", label: "Elastic net", inference: null } as unknown as FittedModel;
const linear = { family: "linear", label: "Linear model", inference: { caption: HC3, refused: null } } as unknown as FittedModel;
// The tables these families declare under inference, as GET /models serves them (ridge.py,
// elastic_net.py and linear.py's ``inference_decl``).
const DECLARED = declaredTables([
  { key: "ridge", inference_table: "shrunk_no_intervals" },
  { key: "elastic_net", inference_table: "shrunk_no_intervals" },
  { key: "linear", inference_table: "intervals" },
] as unknown as FamilyInfo[]);
const PREDICTION = "Squares: estimates of the fitted prediction model, not interpreted as effects.";

function note(model: FittedModel, coefficients: Coefficient[], purpose: string, tables: Record<string, DeclaredTable> = DECLARED): string {
  cleanup(); // one note at a time, so a test may read two
  render(<Coefficients models={[{ model, coefficients }]} purpose={purpose} target="sbp_mmhg" tables={tables} />);
  return screen.getByTestId("coefficients").querySelector("p")!.textContent ?? "";
}

describe("the coefficients' note", () => {
  beforeAll(() => {
    (globalThis as unknown as { ResizeObserver: unknown }).ResizeObserver ??= NoResize;
  });

  it("claims no interval and no effect for a family with no coefficient table under inference", () => {
    const said = note(ridge, [row(-5.4, null, null)], "inference");
    expect(said).not.toMatch(/95%|confidence interval/);
    expect(said).not.toMatch(/^Change in predicted/);
    expect(said).not.toMatch(/coefficient table/);
    expect(said).toBe(
      "Ridge's coefficients are shrunk toward zero by its penalty and have no standard errors, so they describe the fitted model; they are not interpreted as effects, and no intervals are drawn.",
    );
  });

  it("says the penalty shrinks them only where the family declares a shrunk table", () => {
    expect(note(net, [row(-5.25, null, null)], "inference")).toMatch(/^Elastic net's coefficients are shrunk toward zero by its penalty/);
    // Before GET /models answers, nothing is known of the declaration: no shrinkage is claimed.
    expect(note(ridge, [row(-5.4, null, null)], "inference", {})).toBe(
      "Ridge's coefficients have no standard errors, so they describe the fitted model; they are not interpreted as effects, and no intervals are drawn.",
    );
  });

  it("keeps the fit's own caption for a family with a table, and the prediction note", () => {
    expect(note(linear, [row(-5.81, -8.88, -2.74)], "inference")).toBe(
      `Change in predicted sbp_mmhg per unit of each input, holding the others; ${HC3}`,
    );
  });

  it("says under prediction what it said before, for either family", () => {
    const before =
      "Under prediction the coefficients describe the fitted model; they are not interpreted as effects, so no intervals are drawn.";
    expect(note(ridge, [row(-5.4, null, null)], "prediction")).toBe(before);
  });
});

describe("the saved forest figure's caption", () => {
  it("claims no interval and no effect for a family with no coefficient table under inference", () => {
    const said = forestCaption(ridge, "inference", DECLARED.ridge);
    expect(said).not.toMatch(/95%/);
    expect(said).not.toMatch(/lines:/);
    expect(said).toBe(
      "Squares: coefficients as fitted. Ridge's coefficients are shrunk toward zero by its penalty and have no standard errors, so they describe the fitted model; they are not interpreted as effects, and no intervals are drawn.",
    );
  });

  it("keeps a fit's lines and its own caption of them where it has a table, or says why it drew none", () => {
    expect(forestCaption(linear, "inference", DECLARED.linear)).toBe(`Squares: estimates; lines: ${HC3}`);
    const none = "No intervals: fewer than 2 clusters.";
    const refused = { ...linear, inference: { caption: none, refused: "fewer than 2 clusters" } } as unknown as FittedModel;
    expect(forestCaption(refused, "inference", DECLARED.linear)).toBe(`Squares: estimates. ${none}`);
  });

  it("says under prediction what it said before, for either family", () => {
    expect(forestCaption(ridge, "prediction", DECLARED.ridge)).toBe(PREDICTION);
    expect(forestCaption(linear, "prediction", DECLARED.linear)).toBe(PREDICTION);
  });
});
