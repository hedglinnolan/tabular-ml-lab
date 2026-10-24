/**
 * The coefficients' note claims only what the fit drew (the ruling of 2026-10-23 on the shrunk
 * effect). Under inference a family whose fit gives no coefficient table (ridge: its fit serves
 * `inference: null`, no word of how intervals were made) draws no interval and reports no effect,
 * so its note claims neither: no "95%", no change per unit read as an effect. A family with a
 * table keeps its note, the fit's own caption of its intervals; under prediction nothing changes.
 */
import { render, screen } from "@testing-library/react";
import { beforeAll, describe, expect, it } from "vitest";
import type { Coefficient, FittedModel } from "../../../api/m1-stage-types";
import { Coefficients } from "./Coefficients";

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
const linear = { family: "linear", label: "Linear model", inference: { caption: HC3 } } as unknown as FittedModel;

function note(model: FittedModel, coefficients: Coefficient[], purpose: string): string {
  render(<Coefficients models={[{ model, coefficients }]} purpose={purpose} target="sbp_mmhg" />);
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
    expect(said).toMatch(/^Ridge has no coefficient table/);
    expect(said).toMatch(/not interpreted as effects, and no intervals are drawn\.$/);
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
