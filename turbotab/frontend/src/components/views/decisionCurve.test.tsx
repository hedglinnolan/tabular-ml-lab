import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { DecisionCurveView } from "./DecisionCurveView";
import { decisionDomains, decisionScales, shadedRange, treatAllLeaves, type DecisionCurveData } from "./decisionCurve";
import { decisionFromEngine } from "./adapters";
import { decision } from "./lab/fixtures";

const d: DecisionCurveData = {
  rows: [
    { threshold: 0.1, treat_all: 0.2, treat_none: 0, models: { m: 0.22, b: 0.2 } },
    { threshold: 0.2, treat_all: 0.1, treat_none: 0, models: { m: 0.15, b: 0.12 } },
    { threshold: 0.3, treat_all: -0.5, treat_none: 0, models: { m: 0.05, b: -0.01 } },
  ],
  models: [
    { key: "m", label: "Boosted trees", slot: 1 },
    { key: "b", label: "Benchmark", slot: 2 },
  ],
  low: 0.05,
  high: 0.25,
  useful: [0.1, 0.3],
};

describe("the decision curve's scales", () => {
  it("stops a plunging treat-everyone a tenth of the top below zero, never past the models", () => {
    const dom = decisionDomains(d)!;
    expect(dom.x).toEqual([0.1, 0.3]);
    expect(dom.y[0]).toBeCloseTo(-0.022, 12);
    expect(dom.y[1]).toBe(0.22);
    expect(treatAllLeaves(d, -0.022)).toBe(0.3);
    const g = decisionScales(d, 600, 40, 12)!;
    expect(g.x(0.1)).toBe(40);
    expect(g.x(0.3)).toBe(588);
    expect(g.y(0.22)).toBe(24);
    expect(g.y(dom.y[0])).toBe(262);
    for (const t of g.yTicks) expect(t >= -0.022 && t <= 0.22).toBe(true);
  });

  it("clamps the declared range to the thresholds served", () => {
    expect(shadedRange(d)).toEqual([0.1, 0.25]);
    expect(shadedRange({ ...d, low: 0.4, high: 0.5 })).toBeNull();
  });

  it("orders the engine's models with the reported one first, so it takes sage", () => {
    const m = decisionFromEngine({ family: "b", low: 0.05, high: 0.5, rows: d.rows, useful: null }, { m: "Boosted trees", b: "Benchmark" }).models;
    expect(m.map((x) => [x.key, x.slot])).toEqual([["b", 1], ["m", 2]]);
  });
});

describe("the decision curve view", () => {
  it("draws both references, the shaded range, each model, and a table of every threshold", () => {
    const { container } = render(<DecisionCurveView data={d} />);
    expect(container.querySelector('[data-ref="treat-all"]')).toBeTruthy();
    expect(container.querySelector('[data-ref="treat-none"]')).toBeTruthy();
    expect(container.querySelector('[data-mark="range"]')).toBeTruthy();
    expect(container.querySelectorAll("[data-line]")).toHaveLength(2);
    expect(screen.getByTestId("view-table").querySelectorAll("tbody tr")).toHaveLength(3);
    expect(container.textContent).toContain("Treating everyone falls below the chart past 0.3");
  });

  it("draws one threshold as markers", () => {
    const { container } = render(<DecisionCurveView data={{ ...d, rows: [d.rows[0]!] }} />);
    expect(container.querySelectorAll("[data-line] circle")).toHaveLength(2);
  });

  it("says why in one line when there is no curve", () => {
    const { container } = render(<DecisionCurveView data={null} />);
    expect(container.querySelector("svg")).toBeNull();
    expect(screen.getByRole("status")).toHaveTextContent(/yes\/no outcome/);
  });

  it("reads the engine's rows (illustrative fixture)", () => {
    expect(decision.models[0]!.label).toBe("Boosted trees");
    expect(decision.useful).toEqual([0.05, 0.5]);
    expect(decision.rows[0]!.threshold).toBe(0.01);
  });
});
