import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { CurveView } from "./CurveView";
import { BOX_BOTTOM, STRIP_GAP, STRIP_H, curveFrame, curveScales, endLabels, xDomain, yDomain, type CurveData } from "./curve";
import { CHAR_W } from "./common/parts";
import { extent, nearest } from "./common/scale";
import { curveOnePoint, exposureCurve, substitutionSealed } from "./lab/curves.fixtures";

const base: CurveData = {
  xLabel: "kcal moved",
  yLabel: "Change in predicted glucose",
  zero: true,
  sealed: null,
  lines: [
    { key: "now", label: "Each k's own rows", role: "now", x: [0, 100, 200, 300], y: [0, -1, -2, null], low: [0, -1.5, -3, null], high: [0, -0.5, -1, null] },
    { key: "choice", label: "The same rows", role: "choice", x: [0, 100, 200, 300], y: [0, -1.2, -1.6, null] },
  ],
};

describe("the curve's scales", () => {
  it("spans the defined estimates and the band, with zero, in known pixels inset off the axes", () => {
    expect(extent([null, 3, Number.NaN, -1])).toEqual([-1, 3]);
    expect(nearest([0, 100, 200], 149)).toBe(1);
    expect(xDomain(base)).toEqual([0, 200]); // 300 has no estimate
    expect(yDomain(base)).toEqual([-3, 0]);
    const g = curveScales(base, 600, 40, 12)!;
    // the plot runs 40 … 588 and 24 … 262; the domain is inset 8 px from each edge
    expect(g.x(0)).toBe(48);
    expect(g.x(200)).toBe(580);
    expect(g.x(100)).toBe(314);
    expect(g.y(0)).toBe(32);
    expect(g.y(-3)).toBe(254);
    for (const t of g.xTicks) expect(t >= 0 && t <= 200).toBe(true);
    for (const t of g.yTicks) expect(t >= -3 && t <= 0).toBe(true);
  });

  it("reaches the stop and the rug, but not values never estimated", () => {
    expect(xDomain({ ...base, stop: { x: 300, why: "" } })).toEqual([0, 300]);
    expect(xDomain({ ...base, rug: [-20, 50] })).toEqual([-20, 200]);
  });

  it("names only the one value a single point reaches, and keeps the point off the axes", () => {
    const g = curveScales(curveOnePoint, 600, 40)!;
    expect(g.xTicks).toEqual([100]);
    expect(g.yTicks.every((t) => t === -1.86 || t === 0 || (t > -1.86 && t < 0))).toBe(true);
    expect(g.y(-1.86)).toBe(g.plotBottom - 8); // above the axis line, not on it
    expect(g.x(100)).toBe((40 + 588) / 2);
  });

  it("draws the support in its own strip under the plot, never over the curve", () => {
    const g = curveScales({ ...base, support: { x: [0, 100, 200], share: [1, 0.9, 0.5] } }, 600, 40)!;
    expect(g.strip!.top).toBe(g.plotBottom + STRIP_GAP);
    expect(g.strip!.bottom).toBe(g.strip!.top + STRIP_H);
    expect(g.strip!.y(1)).toBe(g.strip!.top);
    expect(g.strip!.y(0.5)).toBe(g.strip!.bottom - STRIP_H / 2);
    expect(g.axisY).toBe(g.strip!.bottom);
    expect(g.box.height).toBe(g.axisY + BOX_BOTTOM);
  });
});

describe("the direct labels", () => {
  it("drop when two line ends would collide, or one would run past the view", () => {
    const x = (v: number) => v;
    expect(endLabels(base.lines, x, (v) => v * 100)).toHaveLength(2); // ends 40 px apart
    expect(endLabels(base.lines, x, (v) => v * 10)).toBeNull(); // 4 px apart
    expect(endLabels(base.lines, x, (v) => v * 100, 200 + 7 + 10 * CHAR_W)).toBeNull(); // "Each k's own rows" is 17 characters
  });

  it("reserve their margin only when they draw, and stay inside the view", () => {
    // a 432 px pane: no labels, and the right margin is given back
    const narrow = curveFrame(exposureCurve, exposureCurve.lines, 432)!;
    expect(narrow.labels).toBeNull();
    expect(narrow.g.box.right).toBe(12);
    const wide = curveFrame(exposureCurve, exposureCurve.lines, 1000)!;
    expect(wide.labels).toHaveLength(2);
    for (const l of wide.labels!) expect(l.x + 7 + l.label.length * CHAR_W).toBeLessThanOrEqual(1000);
    // labels that collide take no margin
    const near = [base.lines[0]!, { ...base.lines[1]!, y: [0, -1, -2.01, null] }];
    const close = curveFrame({ ...base, lines: near }, near, 1000)!;
    expect(close.labels).toBeNull();
    expect(close.g.box.right).toBe(12);
    // a label too long for the cap is never drawn
    const long = { ...base, lines: base.lines.map((l) => ({ ...l, label: `${l.label} with a much longer name here` })) };
    expect(curveFrame(long, long.lines, 1000)!.labels).toBeNull();
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
    const table = screen.getByTestId("table-alternative");
    expect(table.querySelectorAll("tbody tr")).toHaveLength(4);
    expect(table.textContent).toContain("−1 (−1.5 to −0.5)");
  });

  it("draws one point as a marker, not a line, with only its own x tick", () => {
    const { container } = render(<CurveView data={curveOnePoint} />);
    expect(container.querySelectorAll('[data-mark="point"]')).toHaveLength(1);
    expect(container.querySelector('[data-mark="line"]')).toBeNull();
    expect([...container.querySelectorAll('[data-axis="x"] [data-tick]')].map((t) => t.getAttribute("data-tick"))).toEqual(["100"]);
  });

  it("says why in one line when there is nothing to draw", () => {
    const { container } = render(<CurveView data={null} />);
    expect(container.querySelector("svg")).toBeNull();
    expect(screen.getByRole("note")).toHaveTextContent(/no curve/);
  });

  it("before the gate draws only the outcome-free support, labelled, with a tooltip and a table", () => {
    const sealed: CurveData = { ...base, lines: [], sealed: "The curve opens after Fit.", support: { x: [0, 100, 200], share: [1, 0.9, 0.5] } };
    const { container } = render(<CurveView data={sealed} />);
    expect(container.querySelector('[data-mark="line"]')).toBeNull();
    expect(container.querySelector('[data-axis="y"]')).toBeNull();
    expect(container.querySelectorAll('[data-mark="strip"] rect')).toHaveLength(3);
    expect(container.querySelector('[data-mark="strip"]')!.textContent).toContain("Share of rows within the observed range");
    expect(container.querySelector('[role="slider"]')).toBeTruthy(); // the crosshair reads each bar
    const table = screen.getByTestId("table-alternative");
    expect(table.querySelectorAll("tbody tr")).toHaveLength(3);
    expect(table.textContent).toContain("90%");
    expect(screen.getByRole("note")).toHaveTextContent("The curve opens after Fit.");
  });

  it("reads the captured sealed substitution as a strip of eleven bars", () => {
    const { container } = render(<CurveView data={substitutionSealed} />);
    expect(container.querySelectorAll('[data-mark="strip"] rect')).toHaveLength(11);
    expect(screen.getByTestId("table-alternative").querySelectorAll("tbody tr")).toHaveLength(11);
  });
});
