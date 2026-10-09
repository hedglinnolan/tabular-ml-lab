import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { DecisionCurveView } from "./DecisionCurveView";
import { decisionDomains, decisionScales, pointedRange, refusal, shadedRange, treatAllLeaves, usefulLine, type DecisionCurveData } from "./decisionCurve";
import { decisionFromEngine } from "./curveAdapters";
import { decision, decisionOnePoint, decisionPointed, decisionPointedHeldOut } from "./lab/curves.fixtures";

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
  where: "out_of_fold",
  sealed: null,
};

describe("the decision curve's scales", () => {
  it("stops a plunging treat-everyone a tenth of the top below zero, never past the models", () => {
    const dom = decisionDomains(d)!;
    expect(dom.x).toEqual([0.1, 0.3]);
    expect(dom.y[0]).toBeCloseTo(-0.022, 12);
    expect(dom.y[1]).toBe(0.22);
    expect(treatAllLeaves(d, -0.022)).toBe(0.3);
    const g = decisionScales(d, 600, 40, 12)!;
    // the plot runs 40 … 588 and 24 … 262; the domain is inset 8 px from each edge
    expect(g.x(0.1)).toBe(48);
    expect(g.x(0.3)).toBe(580);
    expect(g.y(0.22)).toBe(32);
    expect(g.y(dom.y[0])).toBe(254);
    for (const t of g.yTicks) expect(t >= -0.022 && t <= 0.22).toBe(true);
  });

  it("clamps the declared and the pointed range to the thresholds served", () => {
    expect(shadedRange(d)).toEqual([0.1, 0.25]);
    expect(shadedRange({ ...d, low: 0.4, high: 0.5 })).toBeNull();
    expect(pointedRange({ ...d, pointed: { low: 0.15, high: 0.9 } })).toEqual([0.15, 0.3]);
  });

  it("names only the one threshold a single row reaches", () => {
    const g = decisionScales(decisionOnePoint, 600, 40)!;
    expect(g.xTicks).toEqual([0.21]);
  });

  it("orders the engine's models with the reported one first, so it takes sage", () => {
    const m = decisionFromEngine({ family: "b", low: 0.05, high: 0.5, rows: d.rows, useful: null }, { where: "out_of_fold", labels: { m: "Boosted trees", b: "Benchmark" } }).models;
    expect(m.map((x) => [x.key, x.slot])).toEqual([["b", 1], ["m", 2]]);
  });

  it("says a single threshold as one, never as a span", () => {
    expect(usefulLine(decisionOnePoint)).toBe("Boosted trees does better than treating everyone or no one at a threshold of 0.21.");
    expect(usefulLine(d)).toBe("Boosted trees does better than treating everyone or no one from 0.1 to 0.3.");
  });
});

describe("the decision curve's gate", () => {
  it("refuses a held-out curve while the threshold range is pointed at", () => {
    expect(refusal(decisionPointed)).toBeNull();
    expect(refusal(decisionPointedHeldOut)).toMatch(/held-out rows stay closed/);
    const { container } = render(<DecisionCurveView data={decisionPointedHeldOut} />);
    expect(container.querySelector("svg")).toBeNull();
    expect(container.textContent).not.toMatch(/does better/);
  });

  it("before the gate says one line and draws nothing scored", () => {
    const { container } = render(<DecisionCurveView data={{ ...d, sealed: "The decision curve opens after Fit." }} />);
    expect(container.querySelector("svg")).toBeNull();
    expect(screen.getByRole("status")).toHaveTextContent("opens after Fit");
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

  it("reserves no right margin when the end labels collide", () => {
    // the benchmark ends at −0.01, 9 px from treat no one: no direct labels, so no margin
    const { container } = render(<DecisionCurveView data={d} />);
    expect(container.querySelector('[data-label="treat-none"]')).toBeNull();
    expect(container.querySelector('[data-ref="treat-none"]')).toHaveAttribute("x2", String(600 - 12));
  });

  it("names treat everyone directly on its line, so it never rests on stroke width", () => {
    const { container } = render(<DecisionCurveView data={decision} />);
    expect(container.querySelector('[data-label="treat-all"]')).toHaveTextContent("Treat everyone");
  });

  it("draws the pointed option's range in indigo beside the declared range in gray", () => {
    const { container } = render(<DecisionCurveView data={decisionPointed} />);
    const declared = container.querySelector('[data-mark="range"] rect')!;
    const pointed = container.querySelector('[data-mark="pointed-range"] rect')!;
    expect(declared.getAttribute("style")).toContain("canvas-line");
    expect(pointed.getAttribute("style")).toContain("data-affected");
    expect(Number(pointed.getAttribute("width"))).toBeLessThan(Number(declared.getAttribute("width")));
    expect(screen.getByLabelText("Legend")).toHaveTextContent("With this choice");
  });

  it("draws one threshold as whole markers, treat everyone included", () => {
    const { container } = render(<DecisionCurveView data={decisionOnePoint} />);
    expect(container.querySelectorAll("[data-line] circle")).toHaveLength(2);
    expect(container.querySelector('circle[data-ref="treat-all"]')).toBeTruthy();
    // the models are drawn outside the clip, so a marker at the domain's top is whole
    for (const c of container.querySelectorAll("[data-line] circle")) expect(c.closest("[clip-path]")).toBeNull();
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
    expect(decisionOnePoint.useful).toEqual([0.21, 0.21]);
  });
});
