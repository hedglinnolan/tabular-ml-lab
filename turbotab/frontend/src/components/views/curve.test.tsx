import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { CurveView } from "./CurveView";
import { curveScales, endLabels, xDomain, yDomain, type CurveData } from "./curve";
import { extent, nearest, ticksIn, widen } from "./scale";

const base: CurveData = {
  xLabel: "kcal moved",
  yLabel: "Change in predicted glucose",
  zero: true,
  lines: [
    { key: "now", label: "Each k's own rows", role: "now", x: [0, 100, 200, 300], y: [0, -1, -2, null], low: [0, -1.5, -3, null], high: [0, -0.5, -1, null] },
    { key: "choice", label: "The same rows", role: "choice", x: [0, 100, 200, 300], y: [0, -1.2, -1.6, null] },
  ],
};

describe("the shared scale math", () => {
  it("ticks name values inside the domain, and a flat domain widens around its value", () => {
    expect(ticksIn([0.013, 0.87])).toEqual([0.2, 0.4, 0.6, 0.8]);
    for (const t of ticksIn([-3.1, 0])) expect(t >= -3.1 && t <= 0).toBe(true);
    expect(widen([5, 5])).toEqual([4.5, 5.5]);
    expect(widen([0, 0])).toEqual([-1, 1]);
    expect(extent([null, 3, Number.NaN, -1])).toEqual([-1, 3]);
    expect(nearest([0, 100, 200], 149)).toBe(1);
    expect(nearest([0, 100, 200], 151)).toBe(2);
  });
});

describe("the curve's scales", () => {
  it("spans the defined estimates and the band, with zero, in known pixels", () => {
    expect(xDomain(base)).toEqual([0, 200]); // 300 has no estimate
    expect(yDomain(base)).toEqual([-3, 0]);
    const g = curveScales(base, 600, 40, 12)!;
    expect(g.x(0)).toBe(40);
    expect(g.x(200)).toBe(588);
    expect(g.x(100)).toBe(314);
    expect(g.y(0)).toBe(24); // the top of the plot
    expect(g.y(-3)).toBe(262); // 300 − 38
    for (const t of g.xTicks) expect(t >= 0 && t <= 200).toBe(true);
    for (const t of g.yTicks) expect(t >= -3 && t <= 0).toBe(true);
  });

  it("reaches the stop and the rug, but not values never estimated", () => {
    expect(xDomain({ ...base, stop: { x: 300, why: "" } })).toEqual([0, 300]);
    expect(xDomain({ ...base, rug: [-20, 50] })).toEqual([-20, 200]);
  });

  it("drops direct labels when two line ends would collide", () => {
    const x = (v: number) => v;
    expect(endLabels(base.lines, x, (v) => v * 100)).toHaveLength(2); // ends 40 px apart
    expect(endLabels(base.lines, x, (v) => v * 10)).toBeNull(); // 4 px apart
  });
});

describe("the curve view", () => {
  it("draws gray now and indigo with this choice, and the flip hides the choice", () => {
    const { container, rerender } = render(<CurveView data={base} />);
    expect(container.querySelector('[data-role="now"] [data-mark="line"]')).toBeTruthy();
    expect(container.querySelector('[data-role="choice"] [data-mark="line"]')).toBeTruthy();
    expect(container.querySelector('[data-role="now"] [data-mark="band"]')).toBeTruthy();
    rerender(<CurveView data={base} showChoice={false} />);
    expect(container.querySelector('[data-role="choice"]')).toBeNull();
  });

  it("has a table alternative with every x", () => {
    render(<CurveView data={base} />);
    const table = screen.getByTestId("view-table");
    expect(table.querySelectorAll("tbody tr")).toHaveLength(4);
    expect(table.textContent).toContain("−1 (−1.5 to −0.5)");
  });

  it("draws one point as a marker, not a line", () => {
    const { container } = render(<CurveView data={{ ...base, lines: [{ key: "a", label: "A", role: "now", x: [100], y: [-1.86] }] }} />);
    expect(container.querySelectorAll('[data-mark="point"]')).toHaveLength(1);
    expect(container.querySelector('[data-mark="line"]')).toBeNull();
  });

  it("says why in one line when there is nothing to draw", () => {
    const { container } = render(<CurveView data={null} />);
    expect(container.querySelector("svg")).toBeNull();
    expect(screen.getByRole("status")).toHaveTextContent(/no curve/);
  });

  it("before the gate draws only the outcome-free support, and says when the curve opens", () => {
    const sealed: CurveData = { ...base, lines: [], sealed: "The curve opens after Fit.", support: { x: [0, 100, 200], share: [1, 0.9, 0.5] } };
    const { container } = render(<CurveView data={sealed} />);
    expect(container.querySelector('[data-mark="line"]')).toBeNull();
    expect(container.querySelector('[data-axis="y"]')).toBeNull();
    expect(container.querySelectorAll('[data-mark="rug"] line')).toHaveLength(3);
    expect(screen.getByRole("status")).toHaveTextContent("The curve opens after Fit.");
  });
});
