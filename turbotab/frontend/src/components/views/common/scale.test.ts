import { forestScale, logTicks } from "./scale";

describe("the forest's one scale", () => {
  it("maps a known difference to known pixels, with ticks the data reaches and the reference", () => {
    // extent [−10, 0] padded by 6% each side → domain [−10.6, 0.6]; range [16, 240]: 20 px a unit.
    const s = forestScale([{ est: -5, lo: -10, hi: -1 }], { axis: "linear", reference: 0, width: 256, inset: 16 })!;
    expect(s.domain[0]).toBeCloseTo(-10.6);
    expect(s.domain[1]).toBeCloseTo(0.6);
    expect(s.x(-10)).toBeCloseTo(28);
    expect(s.x(-5)).toBeCloseTo(128);
    expect(s.x(0)).toBeCloseTo(228);
    expect(s.ticks).toEqual([-10, -5, 0]);
    for (const t of s.ticks) expect(t >= s.extent[0] && t <= s.extent[1]).toBe(true);
  });

  it("puts the reference inside the axis even when every interval sits away from it", () => {
    const s = forestScale([{ est: 3, lo: 2, hi: 4 }], { axis: "linear", reference: 0, width: 400 })!;
    expect(s.extent).toEqual([0, 4]);
    expect(s.ticks).toContain(0);
    expect(s.x(0)).toBeGreaterThanOrEqual(14);
    expect(s.x(4)).toBeLessThanOrEqual(400 - 14);
  });

  it("draws a ratio on a log axis: doubling and halving sit the same distance from 1", () => {
    const s = forestScale([{ est: 2, lo: 0.5, hi: 4 }], { axis: "log", reference: 1, width: 400 })!;
    expect(s.x(2) - s.x(1)).toBeCloseTo(s.x(1) - s.x(0.5));
    expect(s.x(4) - s.x(2)).toBeCloseTo(s.x(2) - s.x(1));
    expect(s.ticks).toContain(1);
    for (const t of s.ticks) expect(t >= 0.5 && t <= 4).toBe(true);
  });

  it("names round ratios inside the extent, fewer when the width is narrow", () => {
    expect(logTicks(1, 4, 4)).toEqual([1, 2, 3, 4]);
    expect(logTicks(0.5, 4, 4)).toEqual([0.5, 1, 2, 3]);
    expect(logTicks(0.1, 100, 2)).toEqual([0.1, 10]);
    // Too narrow for two round ratios: a linear subdivision inside it.
    const narrow = logTicks(0.9, 1.1, 3);
    expect(narrow.length).toBeGreaterThanOrEqual(2);
    for (const t of narrow) expect(t >= 0.9 && t <= 1.1).toBe(true);
  });

  it("drops a tick that would crowd the reference's label", () => {
    // 0 sits 25 px from the reference 0.25 here (100 px a unit), inside the extent: the reference keeps its label alone.
    const s = forestScale([{ est: 1, lo: 0, hi: 2 }], { axis: "linear", reference: 0.25, width: 256, inset: 16 })!;
    expect(s.extent).toEqual([0, 2]);
    expect(s.ticks).toEqual([0.25, 1, 2]);
  });

  it("centers one point that sits on the reference, and refuses what cannot be drawn", () => {
    const one = forestScale([{ est: 0, lo: null, hi: null }], { axis: "linear", reference: 0, width: 200 })!;
    expect(one.x(0)).toBeCloseTo(100);
    expect(one.ticks).toEqual([0]);
    expect(forestScale([], { axis: "linear", reference: 0, width: 200 })).toBeNull();
    expect(forestScale([{ est: null, lo: null, hi: null }], { axis: "linear", reference: 0, width: 200 })).toBeNull();
    expect(forestScale([{ est: 0, lo: -1, hi: 1 }], { axis: "log", reference: 1, width: 200 })).toBeNull();
  });
});
