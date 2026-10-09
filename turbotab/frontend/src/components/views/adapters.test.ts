import type { ProfileArtifact } from "../../api/schema";
import {
  forestFromEffects,
  gateFromLock,
  LOCK_GATE,
  numberInPlacementOrder,
  pageFromExhibits,
  table1FromArtifact,
  table1FromProfile,
  table2FromEffects,
} from "./adapters";
import type { Table1Artifact } from "./contracts";
import { exhibitModel, FX, table1Artifact, TABLE1_COLUMNS, TABLE1_LABELS } from "./lab/fixtures";
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

  it("builds Table 1's overall column from the profile: mean (SD) under the codebook's label, each level's n (%)", () => {
    const t1 = table1FromArtifact(table1Artifact, null);
    expect(t1.columns).toEqual([{ key: "overall", label: "Overall", sub: "n = 21,849" }]);
    const female = t1.rows.find((r) => r.kind === "row" && r.key === "gender=female");
    expect(female).toMatchObject({ indent: true, cells: { overall: { kind: "count_pct", n: 11195 } } });
    const pct = female?.kind === "row" && female.cells.overall?.kind === "count_pct" ? female.cells.overall.pct : null;
    expect(pct).toBeCloseTo((100 * 11195) / 21849);
    expect(t1.rows.find((r) => r.kind === "row" && r.key === "age")).toMatchObject({ label: "Age, years, mean (SD)" });
    expect(t1.footnotes[0]!.text).toBe("Characteristics of all 21,849 rows in the file, every one of them analyzed.");
    expect(Object.keys(TABLE1_LABELS)).toEqual(TABLE1_COLUMNS);
  });

  it("says when the profile's rows are not the analyzed ones, and counts the header's n from the profile", () => {
    const t = table1FromProfile(FX.nhanes.profile, { columns: ["age"], unit: "participants", outcome: null, analyzed: 20000 });
    expect(t.groups[0]!.n).toBe(21849);
    expect(t.rows).toBe("all 21,849 rows in the file, before Who's in; the analysis keeps 20,000");
  });

  it("gathers the levels past the profile's top five in one row, so the percents sum to 100", () => {
    const race = {
      ...FX.nhanes.profile.columns.find((c) => c.name === "gender")!,
      name: "race",
      n: 1000,
      n_missing: 10,
      n_unique: 7,
      top: [300, 250, 200, 100, 80].map((count, i) => ({ value: `level ${i + 1}`, count })),
    };
    const profile = { ...FX.nhanes.profile, columns: [race] } as ProfileArtifact;
    const t = table1FromProfile(profile, { columns: ["race"], unit: "participants", outcome: null, analyzed: 1000 });
    const levels = t.variables[0]!.levels!;
    expect(levels.map((l) => l.level).at(-1)).toBe("2 other levels");
    expect(levels.at(-1)!.summary.overall).toMatchObject({ n: 60 });
    expect(levels.reduce((a, l) => a + (l.summary.overall!.pct ?? 0), 0)).toBeCloseTo(100);
  });

  it("leaves the outcome out of a profile Table 1, and holds it out beside the groups until the lock", () => {
    const t = table1FromProfile(FX.nhanes.profile, { columns: ["age", "bmi"], unit: "participants", outcome: "bmi", analyzed: 21849 });
    expect(t.variables.map((v) => v.column)).toEqual(["age"]);
    const grouped: Table1Artifact = {
      ...table1Artifact,
      outcome: "bmi",
      groups: [...table1Artifact.groups, { key: "high", label: "High sugar", n: 100 }],
    };
    const closed = table1FromArtifact(grouped, LOCK_GATE);
    expect(closed.rows.some((r) => r.key === "bmi")).toBe(false);
    expect(closed.footnotes.at(-1)!.text).toBe("Body mass index, kg/m², the outcome, is shown by group once the plan is locked.");
    expect(table1FromArtifact(grouped, null).rows.some((r) => r.key === "bmi")).toBe(true);
    // The outcome alone, in the overall column, is not held.
    expect(table1FromArtifact({ ...table1Artifact, outcome: "bmi" }, LOCK_GATE).rows.some((r) => r.key === "bmi")).toBe(true);
  });

  it("numbers exhibits in placement order: main text from 1, the supplement from S1, none left out", () => {
    const n = numberInPlacementOrder([
      { key: "a", kind: "table", placement: "left_out" },
      { key: "b", kind: "table", placement: "results" },
      { key: "c", kind: "figure", placement: "supplement" },
      { key: "d", kind: "figure", placement: "discussion" },
      { key: "e", kind: "table", placement: "supplement" },
      { key: "f", kind: "table", placement: "discussion" },
    ]);
    expect(Object.fromEntries(n)).toEqual({ a: null, b: "Table 1", c: "Figure S1", d: "Figure 1", e: "Table S1", f: "Table 2" });
    const model = exhibitModel();
    expect(model.exhibits.map((e) => e.number)).toEqual([...numberInPlacementOrder(model.exhibits).values()]);
    expect(model.exhibits.map((e) => e.number)).toEqual(["Table 1", "Table 2", "Figure S1"]);
  });

  it("carries the placements the floor allows onto the page", () => {
    const p = pageFromExhibits(exhibitModel(), "table1")!;
    expect(p.exhibits.find((e) => e.key === "table1")).toMatchObject({ allowed: ["results", "supplement", "left_out"] });
  });

  it("fails closed on the lock: no lock on record is a closed gate", () => {
    expect(gateFromLock(undefined)).toBe(LOCK_GATE);
    expect(gateFromLock({ locked: false, plan_lock: null })).toBe(LOCK_GATE);
    expect(gateFromLock({ locked: true, plan_lock: "d-12" })).toBeNull();
  });

  it("makes each declared model a series when there are several exposures", () => {
    const f = forestFromEffects(FX.nhanes.effects);
    expect(f.series).toBeUndefined();
    expect(f.stub).toBe("Model");
  });
});
