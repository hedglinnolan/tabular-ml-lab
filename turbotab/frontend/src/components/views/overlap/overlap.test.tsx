import { fireEvent, render, screen } from "@testing-library/react";
import { ENGINE_N_TRIMMED, OVERLAP_ENGINE, OVERLAP_ONE_EACH, OVERLAP_ONE_GROUP, OVERLAP_UNTRIMMED } from "../fixtures";
import { keepLine, layoutOverlap, type OverlapLayout } from "./layout";
import { OverlapView } from "./OverlapView";

const lay = (input: Parameters<typeof layoutOverlap>[0], w = 400): OverlapLayout => {
  const r = layoutOverlap(input, w);
  if ("empty" in r) throw new Error(r.empty);
  return r.layout;
};

describe("overlap scale", () => {
  it("puts known values at known pixels, one scale above and below", () => {
    const l = lay(OVERLAP_ONE_EACH);
    // left 44, right 400 − 10: a chance of 0 at 44, of 1 at 390
    expect(l.x(0)).toBe(44);
    expect(l.x(1)).toBe(390);
    expect(l.x(0.5)).toBe(217);
    // one row in each group: each is all of its group, as tall as the half (260 − 24 − 42) / 2
    expect(l.half).toBe(97);
    expect(l.mid).toBe(121);
    expect(l.bins[2]!.groups[0]!.h).toBe(97);
    expect(l.bins[1]!.groups[1]!.h).toBe(97);
    expect(l.h(0.5)).toBeCloseTo(48.5);
    // the same share is the same height above and below
    const t = l.yTicks[0]!;
    expect(l.mid - t.up).toBeCloseTo(t.down - l.mid);
  });

  it("names only values the data reaches on its ticks", () => {
    const l = lay(OVERLAP_ENGINE, 600);
    expect(l.xTicks.map((t) => t.label)).toEqual(["0", "0.2", "0.4", "0.6", "0.8", "1"]);
    for (const t of l.yTicks) expect(t.value).toBeLessThanOrEqual(l.maxShare);
    for (const t of l.xTicks) expect(t.px).toBeGreaterThanOrEqual(l.left);
    for (const t of l.xTicks) expect(t.px).toBeLessThanOrEqual(l.right);
  });

  it("trims the engine's bins to exactly the rows the engine trimmed", () => {
    const l = lay(OVERLAP_ENGINE, 600);
    expect(l.totals).toEqual([666, 834]);
    const summed = l.bins.filter((b) => b.trimmed).reduce((a, b) => a + b.groups[0]!.count + b.groups[1]!.count, 0);
    expect(summed).toBe(ENGINE_N_TRIMMED);
    expect(l.bins.filter((b) => b.trimmed).map((b) => b.lo)).toEqual([0, 0.05, 0.9, 0.95]);
    expect(keepLine(OVERLAP_ENGINE, l)).toBe("506 rows outside 0.1 to 0.9 would be trimmed.");
  });

  it("says when the cut falls inside a bin and does not guess its count", () => {
    const input = { ...OVERLAP_ENGINE, keep: { lo: 0.12, hi: 0.88, n_trimmed: null } };
    const l = lay(input, 600);
    expect(l.bins.some((b) => b.split)).toBe(true);
    expect(l.keep!.nTrimmed).toBeNull();
    expect(keepLine(input, l)).toMatch(/^Rows outside .* inside is drawn as kept\.$/);
  });
});

describe("overlap states", () => {
  it("says why in one line, without a frame, when a group has no rows", () => {
    render(<OverlapView input={OVERLAP_ONE_GROUP} />);
    expect(screen.getByRole("note")).toHaveTextContent("No rows have not exposed, so there is nothing for exposed to overlap with.");
    expect(screen.queryByRole("img")).toBeNull();
  });

  it("says why when the bins and counts disagree", () => {
    const r = layoutOverlap({ ...OVERLAP_ONE_EACH, edges: [0, 1] }, 400);
    expect(r).toEqual({ empty: "The overlap's bins and counts do not match, so nothing is drawn." });
  });

  it("draws one row in each group", () => {
    render(<OverlapView input={OVERLAP_ONE_EACH} />);
    expect(screen.getByRole("img")).toBeInTheDocument();
    expect(screen.getAllByText("exposed").length).toBeGreaterThan(0);
  });

  it("offers its table alternative", () => {
    render(<OverlapView input={OVERLAP_UNTRIMMED} title="Overlap" />);
    fireEvent.click(screen.getByRole("button", { name: "Show as a table" }));
    const table = screen.getByRole("table");
    expect(table).toHaveTextContent("heavy_user = 1");
    expect(screen.getAllByRole("row")).toHaveLength(1 + 20 + 1);
    expect(screen.queryByRole("img")).toBeNull();
  });
});
