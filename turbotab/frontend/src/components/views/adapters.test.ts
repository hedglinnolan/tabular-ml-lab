import { forestFromEffects, table1FromArtifact, table1FromProfile, table2FromEffects } from "./adapters";
import { FX, TABLE1_COLUMNS } from "./lab/fixtures";
import type { TableRow } from "./types";

const keys = (rows: TableRow[]) => rows.flatMap((r) => (r.kind === "row" ? [r.key] : []));

describe("from the engine's artifacts to the views' inputs", () => {
  it("passes the effects' numbers through: Table 2 and its forest share rows, keys and labels", () => {
    const t2 = table2FromEffects(FX.nhanes.effects);
    const f = forestFromEffects(FX.nhanes.effects);
    expect(keys(t2.rows)).toEqual(f.rows.map((r) => r.key));
    const primary = f.rows.find((r) => r.primary)!;
    expect([primary.key, primary.est, primary.lo, primary.hi]).toEqual(["model_2", -0.0199336, -0.0326921, -0.00717513]);
    expect(f).toMatchObject({ axis: "linear", reference: 0, sides: ["Lower mean glucose", "Higher mean glucose"] });
    expect(f.measure).toBe("Difference in mean glucose per unit of sugar");
    // Model 3's note rides in its footnote: possible mediators, not a total effect.
    expect(t2.footnotes.find((n) => n.mark === "b")!.text).toMatch(/possible mediators: not a total effect\.$/);
  });

  it("puts a ratio on its ratio scale, the odds ratio and its limits, on a log axis around 1", () => {
    const f = forestFromEffects(FX.timeVarying.effects);
    expect(f).toMatchObject({ axis: "log", reference: 1, sides: ["Lower odds of cvd", "Higher odds of cvd"] });
    expect(f.rows.map((r) => [r.est, r.lo, r.hi])).toEqual([
      [2.11186, 1.28641, 3.46699],
      [2.10927, 1.29223, 3.44291],
    ]);
    expect(table2FromEffects(FX.timeVarying.effects).columns[0]!.label).toBe("Conditional odds ratio (95% CI)");
  });

  it("builds Table 1's overall column from the profile: mean (SD), and each level's n (%)", () => {
    const t1 = table1FromArtifact(
      table1FromProfile(FX.nhanes.profile, { columns: TABLE1_COLUMNS, unit: "participants", rows: "all 21,849 analyzed rows", n: 21849 }),
    );
    expect(t1.columns).toEqual([{ key: "overall", label: "Overall", sub: "n = 21,849" }]);
    const female = t1.rows.find((r) => r.kind === "row" && r.key === "gender=female");
    expect(female).toMatchObject({ indent: true, cells: { overall: { kind: "count_pct", n: 11195 } } });
    const pct = female?.kind === "row" && female.cells.overall?.kind === "count_pct" ? female.cells.overall.pct : null;
    expect(pct).toBeCloseTo((100 * 11195) / 21849);
    expect(t1.rows.find((r) => r.kind === "row" && r.key === "age")).toMatchObject({ label: "age, mean (SD)" });
  });
});
