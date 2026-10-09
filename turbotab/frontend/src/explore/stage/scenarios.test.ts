/**
 * The prototype's promises, checked: word budgets (BLUEPRINT §11.4), the shelf is never
 * shortened, every finding is presented exactly once, numbers come from the fixture, and
 * the fixture copy has not drifted from the one the capture script wrote.
 */
import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { A, B } from "./fixture";
import { ALL_FINDING_IDS, PUSHED, REST } from "./findings";
import { fmtR, words } from "./format";
import { QUESTIONS } from "./scenarios";

const here = dirname(fileURLToPath(import.meta.url));
const questions = Object.values(QUESTIONS);

describe("word budgets", () => {
  it.each(questions.map((q) => [q.id, q] as const))("%s", (_, q) => {
    expect(words(q.question)).toBeLessThanOrEqual(14);
    expect(words(q.why)).toBeLessThanOrEqual(22);
    for (const o of q.options) {
      expect(words(o.label), o.label).toBeLessThanOrEqual(4);
      expect(words(o.line), o.line).toBeLessThanOrEqual(16);
      for (const sv of o.preview.views) {
        expect(words(sv.view.title), sv.view.title).toBeLessThanOrEqual(8);
        expect(words(sv.view.caption), sv.view.caption).toBeLessThanOrEqual(20);
      }
      if (o.preview.note) expect(words(o.preview.note)).toBeLessThanOrEqual(20);
      if (o.why) expect(words(o.why), o.why).toBeLessThanOrEqual(60);
    }
  });

  it("finding claims and levers", () => {
    for (const card of [...PUSHED, ...REST]) {
      if (card.lever) expect(words(card.lever.label)).toBeLessThanOrEqual(5);
      for (const p of card.pages) {
        expect(words(p.claim), p.claim).toBeLessThanOrEqual(20);
        for (const sv of p.stage.views) {
          expect(words(sv.view.title), sv.view.title).toBeLessThanOrEqual(8);
          expect(words(sv.view.caption), sv.view.caption).toBeLessThanOrEqual(20);
        }
      }
    }
  });
});

describe("the shelf is never shortened", () => {
  it("offers every energy method, the refused one included and saying why", () => {
    const keys = QUESTIONS.energy.options.map((o) => o.key).sort();
    expect(keys).toEqual(A.energy_adjustment.options.map((o) => o.method).sort());
    const partition = QUESTIONS.energy.options.find((o) => o.key === "partition")!;
    expect(partition.refused).toBeDefined();
    expect(partition.line).toContain("`sugar`");
    expect(partition.preview.views).toHaveLength(3);
  });

  it("keeps residual and density adjacent, so one key press flips between them", () => {
    const keys = QUESTIONS.energy.options.map((o) => o.key);
    expect(keys.indexOf("density") - keys.indexOf("residual")).toBe(1);
  });

  it("lays every preview of a question out the same way, so options morph rather than swap", () => {
    for (const q of questions) {
      const kinds = q.options.map((o) => o.preview.views.map((v) => v.view.kind).join(","));
      expect(new Set(kinds).size, q.id).toBe(1);
    }
  });
});

describe("findings", () => {
  it("presents all 13, each exactly once, with at most three pushed", () => {
    const covered = [...PUSHED, ...REST].flatMap((c) => c.pages.flatMap((p) => p.sources));
    expect(covered.sort()).toEqual([...ALL_FINDING_IDS].sort());
    expect(ALL_FINDING_IDS).toHaveLength(13);
    expect(PUSHED.length).toBeLessThanOrEqual(3);
  });

  it("pages same-kind findings in one card", () => {
    const flags = PUSHED.find((c) => c.id === "flags")!;
    expect(flags.pages.map((p) => p.key)).toEqual(
      expect.arrayContaining(["imputed_bmi", "imputed_bp_di", "imputed_waist"]),
    );
    expect(flags.pages).toHaveLength(6);
  });
});

describe("numbers are the fixture's", () => {
  it("states exclusion counts as computed", () => {
    const line = (k: string) => QUESTIONS.exclusions.options.find((o) => o.key === k)!.line;
    expect(line("sex_specific")).toContain("1,419");
    expect(line("kcal_500_5000")).toContain("501");
    expect(line("keep_all")).toContain("21,849");
  });

  it("prints the residual correlation (-1.9e-17) as 0.00", () => {
    expect(fmtR(-1.939e-17)).toBe("0.00");
    expect(fmtR(0.1389)).toBe("0.14");
  });

  it("names the count columns the fixture has", () => {
    expect(QUESTIONS.transform.question).toContain(String(B.count_columns.n));
  });
});

describe("the fixture copy", () => {
  const source = resolve(here, "../../../../../docs/turbotab-next/m1/explore/fixtures.json");
  it.skipIf(!existsSync(source))("is byte-identical to docs/turbotab-next/m1/explore/fixtures.json", () => {
    // Buffer.equals compares the bytes at once; toEqual walks 834 KB element by element and can
    // run past the 5-second timeout on a machine running at low priority.
    expect(readFileSync(resolve(here, "fixtures.json")).equals(readFileSync(source))).toBe(true);
  });
});
