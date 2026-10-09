import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { SpecCurveView } from "./SpecCurveView";
import { CURVE_H, GAP, HEAD_H, ROW_H, TOP, columnAt, mixedScales, sortSpecs, specLayout, type SpecCurveData } from "./specCurve";
import { specCurve } from "./lab/fixtures";

const d: SpecCurveData = {
  estimateLabel: "Difference per g",
  zero: true,
  choices: [
    { key: "adj", label: "Adjusted for", options: [{ key: "none", label: "Nothing" }, { key: "full", label: "Everything" }] },
  ],
  specs: [
    { key: "a", estimate: -0.02, low: -0.03, high: -0.01, n: 100, primary: true, picks: { adj: "full" } },
    { key: "b", estimate: -0.04, low: -0.05, high: -0.03, n: 100, primary: false, picks: { adj: "none" } },
  ],
};

describe("the specification curve's layout", () => {
  it("sorts by estimate and shares one x between the curve and the choices", () => {
    expect(sortSpecs(d.specs).map((s) => s.key)).toEqual(["b", "a"]);
    const l = specLayout(d, 600, 100)!;
    // two columns across 100 … 590
    expect(l.colW).toBe(245);
    expect(l.cx(0)).toBe(222.5);
    expect(l.cx(1)).toBe(467.5);
    expect(columnAt(l, 101, 2)).toBe(0);
    expect(columnAt(l, 589, 2)).toBe(1);
    // the estimate's axis: the intervals and zero
    expect(l.y(0)).toBe(TOP);
    expect(l.y(-0.05)).toBe(TOP + CURVE_H);
    for (const t of l.yTicks) expect(t >= -0.05 && t <= 0).toBe(true);
    // a heading row, then each option
    const start = TOP + CURVE_H + GAP;
    expect(l.rows.map((r) => [r.kind, r.y])).toEqual([
      ["choice", start + HEAD_H - 6],
      ["option", start + HEAD_H + ROW_H / 2],
      ["option", start + HEAD_H + ROW_H * 1.5],
    ]);
  });

  it("refuses estimates on different scales", () => {
    expect(mixedScales(d.specs)).toBeNull();
    expect(mixedScales([{ ...d.specs[0]!, scaleKey: "per g" }, { ...d.specs[1]!, scaleKey: "per kcal" }])).toEqual(["per g", "per kcal"]);
  });
});

describe("the specification curve view", () => {
  it("marks the primary, draws each pick on the shared x, and has a table", () => {
    const { container } = render(<SpecCurveView data={d} />);
    expect(container.querySelectorAll("[data-spec]")).toHaveLength(2);
    expect(container.querySelector('[data-primary="true"]')).toHaveAttribute("data-spec", "a");
    expect(container.textContent).toContain("Primary");
    expect(container.querySelectorAll('[data-option="adj:full"] circle')).toHaveLength(1);
    const rows = screen.getByTestId("view-table").querySelectorAll("tbody tr");
    expect(rows).toHaveLength(2);
    expect(rows[1]).toHaveTextContent("2 (primary)");
    expect(container.textContent).toContain("2 of 2 intervals exclude zero.");
  });

  it("draws a single specification", () => {
    const { container } = render(<SpecCurveView data={{ ...d, specs: [d.specs[0]!] }} />);
    expect(container.querySelectorAll("[data-spec]")).toHaveLength(1);
  });

  it("says why in one line when there is nothing, or the scales mix", () => {
    const { container, rerender } = render(<SpecCurveView data={null} />);
    expect(container.querySelector("svg")).toBeNull();
    expect(screen.getByRole("status")).toHaveTextContent(/no alternative/);
    rerender(<SpecCurveView data={{ ...d, specs: [{ ...d.specs[0]!, scaleKey: "per g" }, { ...d.specs[1]!, scaleKey: "per kcal" }] }} />);
    expect(container.querySelector("svg")).toBeNull();
    expect(screen.getByRole("status")).toHaveTextContent(/different scales/);
  });

  it("reads the calm scenario's 27 fitted plans, one primary", () => {
    expect(specCurve.specs).toHaveLength(27);
    expect(specCurve.specs.filter((s) => s.primary).map((s) => s.estimate)).toEqual([-0.0199336]);
  });
});
