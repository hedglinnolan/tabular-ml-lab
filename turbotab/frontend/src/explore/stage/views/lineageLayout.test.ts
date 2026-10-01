/**
 * Lineage identity is what makes a morph honest: a node may only glide into a node that
 * is the same column. Checked on the fixture's real lineages.
 */
import { A, B, viewOf } from "../fixture";
import { placeLineage } from "./lineageLayout";

const lineageOf = (method: string) => {
  const o = A.energy_adjustment.options.find((x) => x.method === method)!;
  return viewOf(o.preview!.views, "lineage").after;
};
const identityOf = (method: string, id: string) =>
  placeLineage(lineageOf(method)).nodes.find((n) => n.node.id === id)!;

describe("placeLineage", () => {
  it("gives a rewritten nutrient the same identity under every method", () => {
    const keys = [
      identityOf("residual", "adj:protein_adj").key,
      identityOf("density", "adj:protein_per_kcal").key,
      identityOf("none", "adj:protein").key,
    ];
    expect(new Set(keys).size).toBe(1);
    expect(identityOf("residual", "adj:protein_adj").changed).toBe(true);
    expect(identityOf("none", "adj:protein").changed).toBe(false);
  });

  it("keeps kcal's identity on the partition's kcal_from_other", () => {
    const partition = viewOf(
      A.energy_adjustment.partition_on_macronutrient_totals.preview!.views,
      "lineage",
    ).after;
    const node = placeLineage(partition).nodes.find((n) => n.node.id === "adj:kcal_from_other")!;
    expect(node.identity).toBe("kcal");
    const protein = placeLineage(partition).nodes.find((n) => n.node.id === "adj:kcal_from_protein")!;
    expect(protein.identity).toBe("protein");
  });

  it("says where kcal went when it leaves the model", () => {
    const placed = placeLineage(lineageOf("residual"));
    expect(placed.notes).toContainEqual(expect.objectContaining({ identity: "kcal", text: "folded into 7" }));
    expect(placed.counts.matrix).toBe(10);
    expect(placeLineage(lineageOf("density_multivariate")).counts.matrix).toBe(11);
  });

  it("draws 495 genomics columns as one group row and fans one-hot out", () => {
    const g = viewOf(B.preview.views, "lineage").after;
    const placed = placeLineage(g);
    expect(placed.counts.raw).toBe(499);
    expect(placed.rows).toBeLessThan(10);
    const batch = placed.nodes.filter((n) => n.lane === "matrix" && n.identity === "batch");
    expect(batch.map((n) => n.row)).toEqual([batch[0]!.row, batch[0]!.row + 1, batch[0]!.row + 2]);
    expect(placed.notes).toContainEqual(expect.objectContaining({ identity: "sample_id", text: "not a predictor" }));
  });
});
