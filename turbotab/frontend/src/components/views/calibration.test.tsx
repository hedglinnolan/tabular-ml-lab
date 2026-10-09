import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { CalibrationView, calibrationLine, curvePoints } from "./CalibrationView";
import { calibrationScales, sharedDomain, type CalibrationData } from "./calibration";
import { calibrationBinned, calibrationClinical } from "./lab/fixtures";

const d: CalibrationData = {
  kind: "risk",
  outcome: "progression",
  n: 400,
  observed: 0.3,
  expected: 0.32,
  intercept: { estimate: -0.2, ci_low: -0.4, ci_high: 0 },
  slope: { estimate: 0.8, ci_low: 0.6, ci_high: 1 },
  curve: [
    { x: 0.1, y: 0.12 },
    { x: 0.1, y: 0.12 },
    { x: 0.5, y: 0.45 },
  ],
  bins: [{ predicted: 0.2, observed: 0.25, low: 0.05, high: 0.6, n: 40 }],
};

describe("calibration's one shared scale", () => {
  it("gives both axes the same domain, so the 45° line is drawn at 45°", () => {
    expect(sharedDomain(d)).toEqual([0.05, 0.6]);
    const g = calibrationScales(d, 600, 40)!;
    const [x0, x1] = g.x.range() as [number, number];
    const [y1, y0] = g.y.range() as [number, number];
    expect(x1 - x0).toBe(y1 - y0); // a square plot
    expect(g.x(0.05)).toBe(40);
    expect(g.y(0.05)).toBe(24 + g.side);
    expect(g.x(0.6) - g.x(0.05)).toBe(g.y(0.05) - g.y(0.6));
    for (const t of g.ticks) expect(t >= 0.05 && t <= 0.6).toBe(true);
  });

  it("keeps a risk inside [0, 1]", () => {
    expect(sharedDomain({ ...d, bins: [{ predicted: 0.9, observed: 0.95, low: 0.8, high: 1.04, n: 3 }] })).toEqual([0.1, 1]);
  });
});

describe("the calibration view", () => {
  it("draws the diagonal, the smoothed curve, the groups, and says slope and intercept quietly", () => {
    const { container } = render(<CalibrationView data={d} />);
    expect(container.querySelector('[data-ref="diagonal"]')).toBeTruthy();
    expect(container.querySelector('path[data-mark="curve"]')).toBeTruthy();
    expect(container.querySelectorAll('[data-mark="bin"]')).toHaveLength(1);
    expect(screen.getByTestId("calibration-line")).toHaveTextContent("Slope 0.8 (0.6 to 1), 1 when predictions match · intercept −0.2 (−0.4 to 0), 0 when they match · 400 rows.");
    expect(screen.getByTestId("view-table").querySelectorAll("tbody tr")).toHaveLength(3); // one group, two distinct curve points
  });

  it("drops the repeated points lowess leaves at tied predictions", () => {
    expect(curvePoints(d)).toHaveLength(2);
    expect(curvePoints(calibrationClinical).length).toBeLessThan(calibrationClinical.curve.length);
  });

  it("draws a single group as one marker", () => {
    const { container } = render(<CalibrationView data={{ ...d, curve: [], bins: d.bins }} />);
    expect(container.querySelectorAll("circle[data-mark], [data-mark='bin'] circle")).toHaveLength(1);
  });

  it("says why in one line when nothing was scored", () => {
    const { container } = render(<CalibrationView data={{ ...d, curve: [], bins: [] }} />);
    expect(container.querySelector("svg")).toBeNull();
    expect(screen.getByRole("status")).toHaveTextContent(/not assessed/);
  });

  it("reads the engine's captured and illustrative calibrations", () => {
    expect(calibrationLine(calibrationClinical)).toMatch(/^Slope −0\.106 \(−1\.26 to 1\.05\)/);
    expect(calibrationBinned.bins).toHaveLength(10);
  });
});
