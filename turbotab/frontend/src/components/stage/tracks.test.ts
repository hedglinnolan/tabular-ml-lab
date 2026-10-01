/**
 * The morph rule and the honest-text rules, on the real NHANES previews the server built
 * (src/mocks/m1-stage-fixture.json): within a unit values glide, across a change of unit they
 * crossfade; a long tail is clipped and counted; titles say each thing once.
 */
import type {
  DistributionView,
  PreviewResult,
  RelationshipView,
} from "../../api/m1-stage-types";
import fixture from "../../mocks/m1-stage-fixture.json";
import {
  clipTail,
  extentOfHist,
  extentOfPoints,
  localPos,
  localStep,
  noRepeats,
  readoutOf,
  sameUnit,
  storyboardOf,
  trackOf,
  transitionBetween,
  unitMarker,
  unitRuns,
} from "./tracks";

interface Captured {
  decision: { method?: string; nutrients?: string[]; kind: string };
  body: PreviewResult;
}
const previews = (fixture as unknown as { previews: Record<string, Captured[]> }).previews;
const energy = (method: string) =>
  previews.energy_adjustment!.find((c) => c.decision.method === method && (c.decision.nutrients?.length ?? 0) === 6)!
    .body;
const view = <K extends string>(p: PreviewResult, kind: K) => p.views.find((v) => v.kind === kind)!;

describe("unit-change detection", () => {
  it("morphs a residual (grams stay grams) and crossfades density and partition", () => {
    const rel = (m: string) => view(energy(m), "relationship") as RelationshipView;
    const r = rel("residual");
    expect(sameUnit(extentOfPoints(r.y_label_before, r.points_before), extentOfPoints(r.y_label_after, r.points_after))).toBe(true);
    const d = rel("density");
    expect(transitionBetween(extentOfPoints(d.y_label_before, d.points_before), extentOfPoints(d.y_label_after, d.points_after))).toBe(
      "crossfade",
    );
    const part = previews.energy_adjustment!.find((c) => c.decision.method === "partition" && c.decision.nutrients!.length === 3)!.body;
    const pd = view(part, "distribution") as DistributionView;
    expect(transitionBetween(extentOfHist(pd.before_label, pd.before), extentOfHist(pd.after_label, pd.after))).toBe("crossfade");
  });

  it("reads units from names and from a change of scale", () => {
    expect(unitMarker("fat_total_per_kcal")).toBe("per-kcal");
    expect(unitMarker("kcal_from_fat_total")).toBe("kcal");
    expect(unitMarker("log2(gene_001 + 1)")).toBe("log");
    expect(unitMarker("fat_total_adj")).toBe("");
    // Same name, a rescaled range: a unit change all the same.
    expect(sameUnit({ label: "x", lo: 0, hi: 3000 }, { label: "x", lo: 0, hi: 30 })).toBe(false);
    expect(sameUnit({ label: "x", lo: 0, hi: 260 }, { label: "x_adj", lo: -33, hi: 182 })).toBe(true);
  });

  it("an option switch across units crossfades, and a different bin count never morphs", () => {
    const g = { label: "fat_total", lo: 0, hi: 260 };
    const perKcal = { label: "fat_total_per_kcal", lo: 0.006, hi: 0.071 };
    expect(transitionBetween(g, perKcal)).toBe("crossfade");
    expect(transitionBetween(g, { ...g, label: "fat_total_adj" })).toBe("morph");
    expect(transitionBetween(g, g, false)).toBe("crossfade");
  });

  it("splits a storyboard into runs of one unit: one shared axis per run", () => {
    expect(
      unitRuns([
        { label: "fat_total", lo: 0, hi: 260 },
        { label: "fat_total", lo: 0, hi: 260 },
        { label: "fat_total_adj", lo: -100, hi: 110 },
        { label: "kcal_from_fat_total", lo: 40, hi: 2400 },
      ]),
    ).toEqual([0, 0, 0, 1]);
  });
});

describe("tracks", () => {
  it("makes before, the storyboard and after into real states, labeled", () => {
    const rel = view(energy("residual"), "relationship") as RelationshipView;
    const story: RelationshipView = {
      ...rel,
      story: [{ label: "Fit each nutrient on energy", points: rel.points_before, r: rel.r_before, fit_line: { slope: 0.03, intercept: 4 }, y_label: null }],
    };
    const t = trackOf(story);
    expect(t.states.map((s) => s.label)).toEqual(["Your data now", "Fit each nutrient on energy", "With this choice"]);
    expect(t.still).toBe(false);
    const sb = storyboardOf([t, trackOf(view(energy("residual"), "lineage"))]);
    expect(sb.last).toBe(2);
    // The lineage (no story) follows the storyboard proportionally and lands with it.
    expect(localPos(2, 2, 1)).toBe(1);
    expect(localStep(1, 2, 1, true)).toBe(1);
    expect(localStep(1, 2, 1, false)).toBe(0);
  });

  it("knows a finding's evidence (the same state twice) is still", () => {
    const rel = view(energy("residual"), "relationship") as RelationshipView;
    expect(trackOf({ ...rel, points_after: rel.points_before, y_label_after: rel.y_label_before }).still).toBe(true);
  });

  it("pins the headline numbers as real values: r and n, before → after", () => {
    expect(readoutOf(energy("residual").views)).toEqual([
      { key: "r", name: "r", before: "0.84", after: "0.00" },
      { key: "cols", name: "columns", before: "20", after: "19" },
    ]);
    const ex = previews.exclusions![0]!.body;
    expect(readoutOf(ex.views)[0]).toEqual({ key: "n", name: "n", before: "21,849", after: "20,430" });
  });
});

describe("the clipped tail", () => {
  it("draws kcal to its 99.5th percentile and counts the rows past it", () => {
    const ex = previews.exclusions![1]!.body;
    const d = view(ex, "distribution") as DistributionView;
    const clip = clipTail(d.before);
    expect(clip.hi).toBeLessThan(7000);
    expect(clip.max).toBeGreaterThan(15000);
    const total = d.before.counts.reduce((a, c) => a + c, 0);
    expect(clip.over).toBeGreaterThan(0);
    expect(clip.over / total).toBeLessThanOrEqual(0.005);
  });

  it("keeps a cut that falls in the tail on the axis", () => {
    const h = { edges: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], counts: [500, 300, 150, 40, 5, 2, 1, 1, 0, 1], n_missing: 0 };
    expect(clipTail(h).hi).toBeLessThan(10);
    expect(clipTail(h, [8.5]).hi).toBeGreaterThanOrEqual(8.5);
  });

  it("leaves a histogram alone when there is no long tail", () => {
    const h = { edges: [0, 1, 2, 3, 4], counts: [10, 20, 20, 10], n_missing: 0 };
    expect(clipTail(h)).toMatchObject({ bins: 4, over: 0 });
  });
});

describe("titles", () => {
  it("say each thing once", () => {
    expect(noRepeats("kcal outside sex-specific kcal ranges")).toBe("kcal outside sex-specific ranges");
    expect(noRepeats("`kcal` within its range for each `gender`")).toBe("`kcal` within its range for each `gender`");
    expect(noRepeats("`fat_total` against `kcal`, before and after")).toBe("`fat_total` against `kcal`, before and after");
  });
});
