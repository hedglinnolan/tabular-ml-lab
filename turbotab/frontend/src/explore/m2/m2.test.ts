/**
 * The prototype's two claims that must hold in the picture, not only in the data (BLUEPRINT §8:
 * row identity and seal integrity): a reshape's storyboard never shows a row its state does not
 * have, and a grouped seal never draws a unit on both sides. Plus the coach's budget.
 */
import { FX } from "./data";
import { layout } from "./reshape";
import { plan } from "./seal";
import { variantAt } from "./Scenes";

const widths = Object.fromEntries(
  [...FX.reshape.columns.record, ...FX.reshape.columns.shown].map((c) => [c, 60]),
);

describe("the reshape storyboard", () => {
  it("settles to one visible row per unit for every method", () => {
    for (const m of Object.values(FX.reshape.methods)) {
      const l = layout(FX.reshape, m, 3, widths);
      const visible = l.rows.filter((r) => r.alpha > 0.5).length;
      expect(visible).toBe(FX.reshape.window.units.length);
    }
  });

  it("starts in file order and only folds record columns when it combines", () => {
    const mean = FX.reshape.methods.mean!;
    const first = FX.reshape.methods.first!;
    const ys = layout(FX.reshape, mean, 0, widths).rows.map((r) => r.y);
    expect([...ys].sort((a, b) => a - b)).toEqual(ys);
    expect(layout(FX.reshape, mean, 3, widths).widths.recall_date).toBe(0);
    expect(layout(FX.reshape, first, 3, widths).widths.recall_date).toBe(60);
  });

  it("keeps the coach within budget: at most two notes of at most twelve words", () => {
    for (const fx of [FX.reshape, FX.wide])
      for (const m of Object.values(fx.methods)) {
        expect(m.coach.length).toBeLessThanOrEqual(2);
        for (const n of m.coach) expect(n.text.split(/\s+/).length).toBeLessThanOrEqual(12);
      }
  });
});

describe("the seal fork", () => {
  it("draws every held-out row in the held lane and every training row in the training lane", () => {
    for (const v of Object.values(FX.seal.variants)) {
      const at = variantAt(v, 0.2);
      const p = plan(at, 760);
      const split = p.states[p.phases.indexOf("split")]!;
      at.hold.forEach((h, i) => {
        if (h) expect(split.x[i]!).toBeGreaterThanOrEqual(p.held[0]);
        else expect(split.x[i]!).toBeLessThan(p.train[1]);
      });
    }
  });

  it("never splits a unit across a grouped seal, at any holdout size", () => {
    for (const key of ["grouped", "chronological"] as const) {
      const v = FX.seal.variants[key];
      for (const f of [0.1, 0.2, 0.3]) {
        const at = variantAt(v, f);
        const side = new Map<number, number>();
        at.row_unit.forEach((u, i) => {
          const s = at.hold[i]!;
          expect(side.get(u) ?? s).toBe(s);
          side.set(u, s);
        });
        expect(at.straddle).toBe(0);
      }
    }
  });

  it("states an undetermined basis as unknown, never as zero", () => {
    const v = variantAt(FX.seal.variants.undetermined, 0.2);
    expect(v.straddle).toBeNull();
    expect(v.exploratory).toBe(true);
  });
});
