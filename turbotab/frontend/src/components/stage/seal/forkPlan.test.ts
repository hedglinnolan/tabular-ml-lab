/**
 * The seal's picture (Tier A: seal integrity): every held-out cell lands in the held-out lane and
 * every training cell in the training lane; a grouped seal never draws a unit on both sides; an
 * abandoned one draws a hole and a line for every unit it split; the storyboard ends sealed.
 */
import type { SealCells } from "../../../api/m2-stage-types";
import { forkPlan, phasesOf } from "./forkPlan";

function cells(over: Partial<SealCells>): SealCells {
  return {
    state: "grouped",
    label: "grouped by `pid`",
    exploratory: false,
    column: "pid",
    chronological: false,
    time_column: null,
    boundary: null,
    time_start: null,
    time_end: null,
    n_rows: 12,
    n_holdout: 4,
    n_units: 6,
    n_holdout_units: 2,
    straddle: 0,
    evidence: null,
    hold: [0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
    unit: [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
    unit_time: null,
    ...over,
  };
}

const lanes = (c: SealCells) => {
  const p = forkPlan(c, 760);
  const split = p.states[p.phases.indexOf("split")]!;
  return { p, split };
};

describe("the seal fork", () => {
  it("draws every held-out row in the held lane and every training row in the training lane", () => {
    const variants = [
      cells({}),
      cells({ state: "undetermined", exploratory: true, unit: null, column: null, straddle: null, n_units: null, n_holdout_units: null }),
      cells({
        state: "abandoned",
        exploratory: true,
        hold: [0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0],
        straddle: 2,
        n_holdout_units: 3,
      }),
      cells({ chronological: true, unit_time: [0, 0.2, 0.4, 0.5, 1, 0.1], time_column: "visit_date", time_start: "2021-01-01", time_end: "2023-12-01", boundary: "2023-06-01" }),
    ];
    for (const c of variants) {
      const { p, split } = lanes(c);
      c.hold.forEach((h, i) => {
        if (h) expect(split.x[i]!).toBeGreaterThanOrEqual(p.held[0]);
        else expect(split.x[i]!).toBeLessThan(p.train[1]);
      });
      expect(p.phases.at(-1)).toBe("sealed");
      expect(p.states.at(-1)!.sealed).toBe(1);
    }
  });

  it("never splits a unit across a grouped seal", () => {
    const { p } = lanes(cells({}));
    expect(p.straddles).toHaveLength(0);
    expect(p.holes.size).toBe(0);
  });

  it("draws a hole and a line for every unit an abandoned grouping split", () => {
    const c = cells({ state: "abandoned", exploratory: true, hold: [0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0], straddle: 2 });
    const { p } = lanes(c);
    // Units 0 and 4 have rows on both sides; unit 1 is wholly held out (no hole, no line).
    expect(p.straddles.map(([held]) => c.unit![held])).toEqual([0, 4]);
    expect([...p.holes.keys()]).toEqual([1, 8]);
  });

  it("orders a chronological seal's units by their last time, the boundary before the held-out ones", () => {
    const c = cells({
      chronological: true,
      unit_time: [0, 0.2, 0.4, 0.5, 1, 0.1],
      hold: [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
      time_column: "visit_date",
      time_start: "2021-01-01",
      time_end: "2023-12-01",
      boundary: "2023-03-01",
    });
    const p = forkPlan(c, 760);
    expect(p.phases).toEqual(["flat", "stacked", "ordered", "split", "sealed"]);
    const ordered = p.states[2]!;
    const xOf = (u: number) => ordered.x[c.unit!.indexOf(u)]!;
    expect(xOf(0)).toBeLessThan(xOf(1));
    expect(xOf(3)).toBeLessThan(xOf(4));
    expect(p.boundaryX).not.toBeNull();
    expect(p.boundaryX!).toBeLessThan(xOf(3));
    expect(p.axis.map((a) => a.label)).toEqual(["2021-01-01", "2023-12-01", "2023-03-01"]);
  });

  it("names each basis in its storyboard, never calling an undetermined one sealed clean", () => {
    expect(phasesOf(cells({})).labels.at(-1)).toBe("Sealed: grouped by pid");
    const und = cells({ state: "undetermined", unit: null, straddle: null });
    expect(phasesOf(und).labels.at(-1)).toBe("Sealed by row: basis undetermined");
    for (const c of [cells({}), und]) for (const l of phasesOf(c).labels) expect(l.split(/\s+/).length).toBeLessThanOrEqual(8);
  });
});
