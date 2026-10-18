import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { CalibrationView, calibrationLine, curvePoints } from "./CalibrationView";
import { calibrationRefusal, calibrationScales, sharedDomain, STRIP, supportBars, type CalibrationData } from "./calibration";
import { inner } from "./common/scale";
import { calibrationBinned, calibrationChoosing, calibrationClinical, calibrationOnePoint } from "./lab/curves.fixtures";

const d: CalibrationData = {
  kind: "risk",
  outcome: "progression",
  n: 400,
  observed: 0.3,
  expected: 0.32,
  intercept: { estimate: -0.2, ci_low: -0.4, ci_high: 0, level: 0.95 },
  slope: { estimate: 0.8, ci_low: 0.6, ci_high: 1, level: 0.95 },
  curve: [
    { x: 0.1, y: 0.12 },
    { x: 0.1, y: 0.12 },
    { x: 0.5, y: 0.45 },
  ],
  bins: [{ predicted: 0.2, observed: 0.25, low: 0.05, high: 0.6, n: 40 }],
  where: "out_of_fold",
  sealed: null,
};

describe("calibration's one shared scale", () => {
  it("gives both axes the same domain, so the 45° line is drawn at 45°, inset off the axes", () => {
    expect(sharedDomain(d)).toEqual([0.05, 0.6]);
    const g = calibrationScales(d, 600, 40)!;
    const [x0, x1] = g.x.range() as [number, number];
    const [y1, y0] = g.y.range() as [number, number];
    expect(x1 - x0).toBe(y1 - y0); // a square plot
    expect(g.x(0.05)).toBe(48);
    expect(g.y(0.05)).toBe(24 + g.side - 8);
    expect(g.x(0.6) - g.x(0.05)).toBe(g.y(0.05) - g.y(0.6));
    for (const t of g.ticks) expect(t >= 0.05 && t <= 0.6).toBe(true);
  });

  it("keeps a risk inside [0, 1]", () => {
    expect(sharedDomain({ ...d, bins: [{ predicted: 0.9, observed: 0.95, low: 0.8, high: 1.04, n: 3 }] })).toEqual([0.1, 1]);
  });
});

// W1-R finding 10: the engine's `support`, the rows behind each equal-width stretch of the
// predictions' range (turbotab/core/models/performance.py, as np.histogram counts them).
const supported: CalibrationData = {
  ...d,
  bins: null,
  curve: [
    { x: 0.1, y: 0.12 },
    { x: 0.5, y: 0.45 },
  ],
  support: [
    { lo: 0.1, hi: 0.2, n: 300 },
    { lo: 0.2, hi: 0.3, n: 75 },
    { lo: 0.3, hi: 0.4, n: 0 },
    { lo: 0.4, hi: 0.5, n: 25 },
  ],
};

describe("the rows behind the curve", () => {
  it("draws each stretch's rows as a bar under the plot, as wide as its stretch and as tall as its share of the fullest", () => {
    const g = calibrationScales(supported, 600, 40)!;
    const bars = supportBars(supported, g);
    expect(bars.map((b) => b.bin.n)).toEqual([300, 75, 25]); // a stretch with no rows draws nothing
    for (const b of bars) {
      expect(b.x).toBeCloseTo(g.x(b.bin.lo), 9);
      expect(b.x + b.width).toBeCloseTo(g.x(b.bin.hi), 9);
    }
    expect(bars[1]!.height / bars[0]!.height).toBeCloseTo(75 / 300, 9);
    expect(bars[2]!.height / bars[0]!.height).toBeCloseTo(25 / 300, 9);
    // on the x axis, in a band of its own below everything the plot draws
    const { y1 } = inner(g.box);
    const [lo] = sharedDomain(supported)!;
    for (const b of bars) {
      expect(b.y + b.height).toBe(y1);
      expect(b.y).toBeGreaterThanOrEqual(y1 - STRIP);
    }
    expect(g.y(lo)).toBeLessThan(y1 - STRIP);
  });

  it("keeps the plot square with the strip, and draws none where none is served", () => {
    const g = calibrationScales(supported, 600, 40)!;
    const [x0, x1] = g.x.range() as [number, number];
    const [y1, y0] = g.y.range() as [number, number];
    expect(x1 - x0).toBe(y1 - y0);
    const plain = { ...supported, support: undefined };
    expect(supportBars(plain, calibrationScales(plain, 600, 40)!)).toEqual([]);
    expect(calibrationScales(plain, 600, 40)!.box.height).toBe(g.box.height - STRIP);
    const { container } = render(<CalibrationView data={plain} />);
    expect(container.querySelector('[data-mark="support"]')).toBeNull();
  });

  it("draws the strip in the view, keys it, and lists each stretch's rows in the table", () => {
    const { container } = render(<CalibrationView data={supported} />);
    expect(container.querySelectorAll('[data-mark="support"] rect')).toHaveLength(3);
    expect(container.textContent).toContain("Rows at each prediction");
    const rows = [...screen.getByTestId("table-alternative").querySelectorAll("tbody tr")].map((r) => r.textContent);
    expect(rows.filter((r) => r?.startsWith("Rows predicted"))).toEqual([
      "Rows predicted0.1 to 0.2——300",
      "Rows predicted0.2 to 0.3——75",
      "Rows predicted0.3 to 0.4——0",
      "Rows predicted0.4 to 0.5——25",
    ]);
    expect(screen.getByTestId("table-alternative").querySelector("caption")?.textContent).toContain(
      "rows counted in 4 equal-width stretches of the predictions",
    );
  });
});

describe("the calibration view", () => {
  it("draws the diagonal, the smoothed curve, the groups, and says what they show in one plain paragraph", () => {
    const { container } = render(<CalibrationView data={d} />);
    expect(container.querySelector('[data-ref="diagonal"]')).toBeTruthy();
    expect(container.querySelector('path[data-mark="curve"]')).toBeTruthy();
    expect(container.querySelectorAll('[data-mark="bin"]')).toHaveLength(1);
    expect(screen.getByTestId("calibration-line")).toHaveTextContent(
      "Predicted 32% on average against 30% observed (calibration intercept −0.2, 95% interval −0.4 to 0; 0 when they agree). Outcomes moved 0.8 times as far as the predictions did (calibration slope, 95% interval 0.6 to 1; 1 when they agree, below 1 when predictions are too extreme). 400 rows, scored out of fold.",
    );
    expect(screen.getByTestId("table-alternative").querySelectorAll("tbody tr")).toHaveLength(3); // one group, two distinct curve points
  });

  it("says a flagged calibration once: the engine's verdict, never the same numbers twice", () => {
    const { container } = render(<CalibrationView data={calibrationBinned} />);
    const quiet = [...container.querySelectorAll("figure > p")];
    expect(quiet).toHaveLength(1);
    expect(quiet[0]!.textContent).toMatch(/^Predictions are too extreme out of fold: calibration slope 0\.65 \(95% interval/);
    expect(quiet[0]!.textContent).not.toContain("0.653");
    expect(quiet[0]!.textContent).toContain("1,200 rows, scored out of fold.");
  });

  it("says the interval's level as served, and leaves it out when none is served", () => {
    expect(calibrationLine(calibrationClinical)).toMatch(/Outcomes moved −?\d+(\.\d+)? times as far as the predictions did \(calibration slope, 95% interval −?[\d.]+ to −?[\d.]+;/);
    expect(calibrationLine({ ...d, slope: { estimate: 0.8, ci_low: 0.6, ci_high: 1 } })).toContain("(calibration slope, interval 0.6 to 1;");
  });

  it("says a slope that reads 1 as outcomes moving as far as the predictions did, never \"1 times as far\" (minor 21)", () => {
    // The walk's held-out slope: 1.0 (0.936 to 1.07), shown at the view's precision as 1.
    const one = { ...d, slope: { estimate: 1.0004, ci_low: 0.936, ci_high: 1.07, level: 0.95 } };
    const line = calibrationLine(one);
    expect(line).toContain(
      "Outcomes moved as far as the predictions did (calibration slope 1, 95% interval 0.936 to 1.07; 1 when they agree, below 1 when predictions are too extreme).",
    );
    expect(line).not.toContain("times as far");
    // Any other slope keeps its "times as far", the number said once.
    expect(calibrationLine({ ...d, slope: { estimate: 1.07, ci_low: 0.9, ci_high: 1.2, level: 0.95 } })).toContain(
      "Outcomes moved 1.07 times as far as the predictions did (calibration slope, 95% interval 0.9 to 1.2;",
    );
  });

  it("drops the repeated points lowess leaves at tied predictions", () => {
    expect(curvePoints(d)).toHaveLength(2);
    expect(curvePoints(calibrationClinical).length).toBeLessThan(calibrationClinical.curve.length);
  });

  it("draws a single group as one marker, with that group's own rows", () => {
    const { container } = render(<CalibrationView data={calibrationOnePoint} />);
    expect(container.querySelectorAll("[data-mark='bin'] circle")).toHaveLength(1);
    expect(screen.getByTestId("calibration-line")).toHaveTextContent("Predicted 14% on average against 24.2% observed. 120 rows, scored out of fold.");
  });

  it("says why in one line when nothing was scored", () => {
    const { container } = render(<CalibrationView data={{ ...d, curve: [], bins: [] }} />);
    expect(container.querySelector("svg")).toBeNull();
    expect(screen.getByRole("note")).toHaveTextContent(/not assessed/);
  });

  it("refuses held-out scores while a choice they would inform is made, and is sealed before Fit", () => {
    expect(calibrationRefusal({ ...d, choosing: "the calibration horizon" })).toBeNull(); // out of fold
    expect(calibrationRefusal(calibrationChoosing)).toMatch(/^The calibration horizon is being chosen/);
    const { container, rerender } = render(<CalibrationView data={calibrationChoosing} />);
    expect(container.querySelector("svg")).toBeNull();
    rerender(<CalibrationView data={{ ...d, sealed: "Calibration opens after Fit." }} />);
    expect(container.querySelector("svg")).toBeNull();
    expect(screen.getByRole("note")).toHaveTextContent("Calibration opens after Fit.");
  });

  it("reads the engine's captured and illustrative calibrations", () => {
    expect(calibrationLine(calibrationClinical)).toMatch(/^Predicted \d+(\.\d)?% on average against \d+(\.\d)?% observed/);
    expect(calibrationBinned.bins).toHaveLength(10);
  });
});
