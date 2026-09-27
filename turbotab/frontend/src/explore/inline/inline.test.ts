/**
 * The prototype's real logic: its numbers come from the fixture (and the copy is the fixture),
 * its authored words keep the §11.4 budgets, every finding is presented exactly once, and the
 * lineage reading says what each method does to the columns.
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { ENERGY, EXCLUSIONS, FINDING_SET, WIDE } from "./fixture";
import { fmtR, words } from "./format";
import { touched, tracksOf } from "./lineage";

const HERE = resolve(__dirname);

describe("the fixture", () => {
  it("is a byte-for-byte copy of docs/turbotab-next/m1/explore/fixtures.json", () => {
    const copy = readFileSync(resolve(HERE, "fixtures.json"));
    const source = readFileSync(
      resolve(HERE, "../../../../../docs/turbotab-next/m1/explore/fixtures.json"),
    );
    expect(copy.equals(source)).toBe(true);
  });
});

describe("word budgets (BLUEPRINT §11.4)", () => {
  const questions = [ENERGY, EXCLUSIONS, WIDE];
  it("questions ≤ 14 words, one-line why ≤ 22", () => {
    for (const q of questions) {
      expect(words(q.question)).toBeLessThanOrEqual(14);
      expect(words(q.why)).toBeLessThanOrEqual(22);
    }
  });
  it("option labels ≤ 4 words, consequences and refusals ≤ 16, in-place why ≤ 60", () => {
    const options = [...ENERGY.options, ...EXCLUSIONS.options, ...WIDE.options];
    for (const o of options) {
      expect(words(o.label), o.label).toBeLessThanOrEqual(4);
      expect(words(o.consequence), o.consequence).toBeLessThanOrEqual(16);
      if (o.refused) expect(words(o.refused), o.refused).toBeLessThanOrEqual(16);
    }
    for (const o of ENERGY.options) expect(words(o.why)).toBeLessThanOrEqual(60);
  });
  it("finding claims ≤ 20 words, lever labels ≤ 5, group titles ≤ 8", () => {
    const cards = [...FINDING_SET.pushed, ...FINDING_SET.groups.flatMap((g) => g.pages)];
    for (const f of cards) {
      expect(words(f.claim), f.claim).toBeLessThanOrEqual(20);
      expect(words(f.lever), f.lever).toBeLessThanOrEqual(5);
    }
    for (const g of FINDING_SET.groups) expect(words(g.title)).toBeLessThanOrEqual(8);
  });
});

describe("findings (§11.7)", () => {
  it("push at most three and present every one of the 13 exactly once", () => {
    expect(FINDING_SET.pushed.length).toBeLessThanOrEqual(3);
    const ids = [
      ...FINDING_SET.pushed.flatMap((f) => f.sources),
      ...FINDING_SET.groups.flatMap((g) => g.pages.flatMap((p) => p.sources)),
    ];
    expect(ids.length).toBe(FINDING_SET.total);
    expect(new Set(ids).size).toBe(FINDING_SET.total);
    expect(FINDING_SET.total).toBe(13);
  });
  it("build their claims from the file's numbers", () => {
    const [energy, implausible] = FINDING_SET.pushed;
    expect(energy!.claim).toContain("0.86");
    expect(implausible!.claim).toContain("`501`");
    expect(implausible!.claim).toContain("`194`");
    expect(implausible!.claim).toContain("`307`");
  });
});

describe("the energy shelf", () => {
  it("keeps all six methods; partition stays, refused, with a preview on the subset it accepts", () => {
    expect(ENERGY.options.map((o) => o.method)).toEqual([
      "residual",
      "density_multivariate",
      "standard",
      "density",
      "none",
      "partition",
    ]);
    const partition = ENERGY.options.find((o) => o.method === "partition")!;
    expect(partition.refused).toContain("`sugar`");
    expect(partition.variant?.scatter).not.toBeNull();
    expect(ENERGY.options.filter((o) => o.refused)).toHaveLength(1);
  });
  it("prints a residual's floating-point zero correlation as 0.00", () => {
    const residual = ENERGY.options.find((o) => o.method === "residual")!;
    expect(Math.abs(residual.scatter!.r!)).toBeLessThan(1e-9);
    expect(fmtR(residual.scatter!.r)).toBe("0.00");
    expect(fmtR(-0.004)).toBe("0.00");
    expect(fmtR(0.1389)).toBe("0.14");
  });
});

describe("lineage tracks", () => {
  const base = tracksOf(ENERGY.baseLineage!);
  const lineageOf = (m: string) => ENERGY.options.find((o) => o.method === m)!.lineage!.after;

  it("residual: seven nutrients change, energy feeds them and leaves the model", () => {
    const t = tracksOf(lineageOf("residual"));
    const kcal = t.find((x) => x.raw === "kcal")!;
    expect(kcal.outs).toEqual([]);
    expect(kcal.op).toBe("feeds");
    const hit = touched(t, base);
    expect([...hit].filter((id) => id !== kcal.id)).toHaveLength(7);
    expect(t.find((x) => x.raw === "fat_total")!.outs).toEqual(["fat_total_adj"]);
    expect(t.find((x) => x.raw === "fat_total")!.inputs).toEqual(["raw:kcal"]);
  });
  it("density + energy keeps energy; the standard model touches nothing", () => {
    const dm = tracksOf(lineageOf("density_multivariate"));
    expect(dm.find((x) => x.raw === "kcal")!.outs).toEqual(["kcal"]);
    expect(touched(tracksOf(lineageOf("standard")), base).size).toBe(0);
  });
  it("partition on the macronutrient totals turns energy into energy from everything else", () => {
    const variant = ENERGY.options.find((o) => o.method === "partition")!.variant!;
    const t = tracksOf(variant.lineage!.after);
    expect(t.find((x) => x.raw === "kcal")!.outs).toEqual(["kcal_from_other"]);
    expect(t.find((x) => x.raw === "sugar")!.outs).toEqual(["sugar"]);
  });
  it("the wide transform collapses 495 count columns into one track", () => {
    const t = tracksOf(WIDE.lineage.after);
    const counts = t.find((x) => x.count > 1)!;
    expect(counts.count).toBe(495);
    expect(counts.op).toBe("log2(x + 1)");
    expect(touched(t, tracksOf(WIDE.lineage.before!))).toEqual(new Set([counts.id]));
  });
});
