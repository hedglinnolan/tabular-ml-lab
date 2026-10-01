/**
 * Lineage identity is what makes a morph honest: a node may only glide into a node that is the
 * same column. Checked on real lineages: the NHANES energy-adjustment previews the server built
 * (src/mocks/m1-stage-fixture.json) and the 495-column genomics fixture of the stage prototype.
 */
import type { LineageView, PreviewResult } from "../../../api/m1-stage-types";
import stageFixture from "../../../mocks/m1-stage-fixture.json";
import exploreFixture from "../../../explore/stage/fixtures.json";
import { placeLineage } from "./lineageLayout";

interface Captured {
  decision: { method?: string; nutrients?: string[] };
  status: number;
  body: PreviewResult;
}
const energy = (stageFixture as unknown as { previews: { energy_adjustment: Captured[] } }).previews
  .energy_adjustment;

const lineageOf = (method: string, nutrients?: number) => {
  const p = energy.find(
    (x) =>
      x.decision.method === method && (nutrients === undefined || x.decision.nutrients?.length === nutrients),
  )!;
  return (p.body.views.find((v) => v.kind === "lineage") as LineageView).after;
};
const placedNode = (method: string, id: string, nutrients?: number) =>
  placeLineage(lineageOf(method, nutrients)).nodes.find((n) => n.node.id === id)!;

describe("placeLineage", () => {
  it("gives a rewritten nutrient the same identity under every method", () => {
    const keys = [
      placedNode("residual", "adj:protein_adj").key,
      placedNode("density", "adj:protein_per_kcal").key,
      placedNode("none", "adj:protein").key,
    ];
    expect(new Set(keys).size).toBe(1);
    expect(placedNode("residual", "adj:protein_adj").changed).toBe(true);
    expect(placedNode("none", "adj:protein").changed).toBe(false);
  });

  it("keeps kcal's identity on the partition's kcal_from_other", () => {
    const node = placedNode("partition", "adj:kcal_from_other", 3);
    expect(node.identity).toBe("kcal");
    expect(placedNode("partition", "adj:kcal_from_protein", 3).identity).toBe("protein");
  });

  it("says where kcal went when it leaves the model", () => {
    const placed = placeLineage(lineageOf("residual"));
    expect(placed.notes).toContainEqual(expect.objectContaining({ identity: "kcal", text: "folded into 6" }));
    expect(placed.counts.matrix).toBe(19);
    expect(placeLineage(lineageOf("density_multivariate")).counts.matrix).toBe(20);
  });

  it("draws 495 genomics columns as one group row and fans one-hot out", () => {
    const b = (exploreFixture as unknown as { scenario_b: { preview: PreviewResult } }).scenario_b;
    const g = (b.preview.views.find((v) => v.kind === "lineage") as LineageView).after;
    const placed = placeLineage(g);
    expect(placed.counts.raw).toBe(499);
    expect(placed.rows).toBeLessThan(10);
    const batch = placed.nodes.filter((n) => n.lane === "matrix" && n.identity === "batch");
    expect(batch.map((n) => n.row)).toEqual([batch[0]!.row, batch[0]!.row + 1, batch[0]!.row + 2]);
    expect(placed.notes).toContainEqual(expect.objectContaining({ identity: "sample_id", text: "not a predictor" }));
  });
});
