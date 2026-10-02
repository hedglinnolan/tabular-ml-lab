/** The live lineage before the design exists keeps its size on a wide table (BLUEPRINT §11 rule 3). */
import { MAX_LINEAGE_COLUMNS, predictorsLineage } from "./LiveScenes";

describe("predictorsLineage", () => {
  it("draws each predictor on a narrow table", () => {
    const l = predictorsLineage(["a", "b"], { a: "exposure", b: "covariate" });
    expect(l.collapsed).toBe(false);
    expect(l.nodes.filter((n) => n.lane === "raw").map((n) => n.column)).toEqual(["a", "b"]);
  });

  it("collapses a wide table into one node per role, with its count, in every lane", () => {
    const genes = Array.from({ length: 495 }, (_, i) => `gene_${i}`);
    const roles = Object.fromEntries([...genes.map((g) => [g, "exposure"]), ["age", "covariate"], ["sex", "covariate"]]);
    const l = predictorsLineage([...genes, "age", "sex"], roles);
    expect(l.collapsed).toBe(true);
    const raw = l.nodes.filter((n) => n.lane === "raw");
    expect(raw.length).toBeLessThanOrEqual(MAX_LINEAGE_COLUMNS);
    expect(raw.map((n) => [n.label, n.count])).toEqual([
      ["495 exposure columns", 495],
      ["2 covariate columns", 2],
    ]);
    expect(l.links).toHaveLength(4); // raw → adjusted → matrix, per group
  });
});
