/**
 * The prototype's real logic, proportionally: the fixture copy has not drifted from the engine's,
 * the words fit their budgets, keyed scenes blend by identity, and derived columns find their
 * source (the table and lineage morphs depend on it).
 */
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  BUDGET,
  ENERGY_OPTIONS,
  ENERGY_Q,
  EXCL_OPTIONS,
  EXCL_Q,
  FINDING_COPY,
  FINDING_GROUPS,
  FINDINGS_PUSHED,
  WIDE_OPTIONS,
  WIDE_Q,
  words,
} from "./copy";
import { ENERGY, FINDINGS, fmtR, sourceOf } from "./data";
import { blend, SceneBuilder } from "./engine/morph";
import { energyTable, slotFor } from "./scenarios/model";

const HERE = dirname(fileURLToPath(import.meta.url));

describe("fixture", () => {
  it("is a verbatim copy of docs/turbotab-next/m1/explore/fixtures.json", () => {
    const copy = readFileSync(resolve(HERE, "fixtures.json"));
    const source = readFileSync(resolve(HERE, "../../../../../docs/turbotab-next/m1/explore/fixtures.json"));
    expect(copy.equals(source)).toBe(true);
  });
});

describe("word budgets (BLUEPRINT §11.4)", () => {
  const questions = [ENERGY_Q, EXCL_Q, WIDE_Q];
  const options = [ENERGY_OPTIONS, EXCL_OPTIONS, WIDE_OPTIONS].flatMap((o) => Object.entries(o));

  it.each(questions.map((q) => [q.kicker, q] as const))("%s: question and why", (_, q) => {
    expect(words(q.question)).toBeLessThanOrEqual(BUDGET.question);
    expect(words(q.why)).toBeLessThanOrEqual(BUDGET.why);
  });

  it.each(options)("option %s", (_, o) => {
    expect(words(o.label)).toBeLessThanOrEqual(BUDGET.label);
    expect(words(o.consequence)).toBeLessThanOrEqual(BUDGET.consequence);
    expect(words(o.sidenote)).toBeLessThanOrEqual(BUDGET.sidenote);
  });

  it("every finding has a summary and lever within budget, and each is shown once", () => {
    const ids = FINDINGS.map((f) => f.id).sort();
    const placed = [...FINDINGS_PUSHED, ...FINDING_GROUPS.flatMap((g) => g.ids)].sort();
    expect(placed).toEqual(ids);
    expect(FINDINGS_PUSHED.length).toBeLessThanOrEqual(3);
    for (const id of ids) {
      const c = FINDING_COPY[id]!;
      expect(words(c.summary)).toBeLessThanOrEqual(BUDGET.summary);
      if (c.lever) expect(words(c.lever)).toBeLessThanOrEqual(BUDGET.lever);
      else expect(c.summary).toMatch(/no control|not|yet/i);
    }
  });

  it("the estimands shown under why? fit the in-place budget", () => {
    for (const o of Object.values(ENERGY)) expect(words(o.estimand)).toBeLessThanOrEqual(BUDGET.whyInPlace);
  });
});

describe("keyed scenes", () => {
  const a = new SceneBuilder().set("kept", { x: 0, o: 1 }).set("old", { w: 40, o: 1, grow: 1 }).scene;
  const b = new SceneBuilder().set("kept", { x: 100, o: 1 }).set("new", { w: 80, o: 1, grow: 1, slide: 10 }).scene;

  it("morphs a key present in both, fades a key present in one", () => {
    const m = blend(a, b, 0.25);
    expect(m.items.get("kept")!.x).toBe(25);
    expect(m.items.get("old")!.o).toBeCloseTo(0.75);
    expect(m.items.get("old")!.w).toBeCloseTo(30);
    expect(m.items.get("new")!.o).toBeCloseTo(0.25);
    expect(m.items.get("new")!.w).toBeCloseTo(20);
    expect(m.items.get("new")!.dy).toBeCloseTo(7.5);
  });

  it("hands a swapped label over instead of overlapping it", () => {
    const x = new SceneBuilder().set("tick:a", { o: 1, swap: 1 }).scene;
    const y = new SceneBuilder().set("tick:b", { o: 1, swap: 1 }).scene;
    const early = blend(x, y, 0.3);
    expect(early.items.get("tick:a")!.o).toBeCloseTo(0.4);
    expect(early.items.get("tick:b")!.o).toBe(0);
    const late = blend(x, y, 0.8);
    expect(late.items.get("tick:a")!.o).toBe(0);
    expect(late.items.get("tick:b")!.o).toBeCloseTo(0.6);
  });

  it("composes: a blend interrupted midway is a scene the next blend starts from", () => {
    const mid = blend(a, b, 0.5);
    const c = new SceneBuilder().set("kept", { x: 0, o: 1 }).scene;
    expect(blend(mid, c, 0.5).items.get("kept")!.x).toBe(25);
    expect(blend(a, b, 0)).toBe(a);
    expect(blend(a, b, 1)).toBe(b);
  });
});

describe("derived columns find their source", () => {
  const slots = ["kcal", "protein", "carb", "sugar", "fat_total"];
  it("by name, longest match first", () => {
    expect(sourceOf("fat_total_per_kcal", slots)).toBe("fat_total");
    expect(sourceOf("carb_per_kcal", slots)).toBe("carb");
  });
  it("by the engine's lineage where the name is ambiguous", () => {
    expect(slotFor("kcal_from_carb", slots, ENERGY.partition3)).toBe("carb");
    expect(slotFor("kcal_from_other", slots, ENERGY.partition3)).toBe("kcal");
  });
  it("puts each adjusted column under its nutrient, and kcal out where the method drops it", () => {
    const t = energyTable();
    expect(t.states.residual!.cols.fat_total!.name).toBe("fat_total_adj");
    expect(t.states.residual!.cols.kcal!.status).toBe("dropped");
    expect(t.states.density_multivariate!.cols.kcal!.status).toBe("same");
    expect(t.states.density!.cols.carb!.name).toBe("carb_per_kcal");
    expect(t.states.none!.cols.fat_total!.status).toBe("same");
  });
  it("prints a correlation that rounds to zero as 0.00", () => {
    expect(fmtR(-1.939e-17)).toBe("0.00");
    expect(fmtR(0.1389)).toBe("0.14");
  });
});
