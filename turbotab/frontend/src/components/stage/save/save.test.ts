/**
 * A saved plot is a real, labeled state with its provenance — never an interpolated frame.
 */
import type { PreviewResult, RelationshipView } from "../../../api/m1-stage-types";
import fixture from "../../../mocks/m1-stage-fixture.json";
import { heading, initial, reduce } from "../player";
import { trackOf } from "../tracks";
import { fileName, saveIndices } from "./export";
import { assertRealState, figureForTrack } from "./journal";

const previews = (fixture as unknown as { previews: Record<string, { decision: { method?: string }; body: PreviewResult }[]> })
  .previews;
const residual = previews.energy_adjustment!.find((c) => c.decision.method === "residual")!.body;
const rel = residual.views.find((v) => v.kind === "relationship") as RelationshipView;
const story: RelationshipView = {
  ...rel,
  story: [
    { label: "Fit each nutrient on energy", points: rel.points_before, r: rel.r_before, fit_line: { slope: 0.03, intercept: 4 } },
    { label: "Keep what energy does not explain", points: rel.points_after, r: 0, fit_line: { slope: 0, intercept: 0 } },
  ],
};
const track = trackOf(story);
const meta = { title: "fat_total against kcal", caption: rel.caption, provenance: "Preview, not recorded: Willett residual model." };

describe("saving", () => {
  it("never captures a mid-animation frame: every save index is a whole step", () => {
    let s = reduce(initial(3), { type: "flip" });
    while (s.pos !== s.target) {
      for (const which of ["now", "step", "with", "pair"] as const) {
        const idx = saveIndices(which, heading(s), 3, 3);
        expect(idx.every(Number.isInteger)).toBe(true);
      }
      s = reduce(s, { type: "tick", ms: 37 });
    }
  });

  it("saves the step the player is heading to, mapped onto a shorter storyboard", () => {
    expect(saveIndices("step", 2, 3, 3)).toEqual([2]);
    expect(saveIndices("step", 2, 3, 1)).toEqual([1]);
    expect(saveIndices("pair", 2, 3, 1)).toEqual([0, 1]);
  });

  it("refuses a fractional state outright", () => {
    expect(() => assertRealState(1.5, 3)).toThrow(/real state/);
    expect(() => figureForTrack(track, [1.4], meta)).toThrow();
    expect(() => figureForTrack(track, [4], meta)).toThrow();
  });

  it("writes a journal figure: serif, literal colors, labeled panels, provenance in the caption", () => {
    const svg = figureForTrack(track, [0, 3], meta);
    expect(svg.startsWith("<svg xmlns=")).toBe(true);
    expect(svg).toContain("Charter");
    expect(svg).not.toContain("var(--");
    expect(svg).toContain("a  Your data now");
    expect(svg).toContain("b  With this choice");
    expect(svg).toContain("Preview, not recorded: Willett residual model.");
    expect(svg).toMatch(/r<\/tspan>[^<]*= 0\.84/);
  });

  it("names files after the plot and the state", () => {
    expect(fileName("`fat_total` against `kcal`", "pair", "png")).toBe("turbotab-fat-total-against-kcal-pair.png");
  });
});
