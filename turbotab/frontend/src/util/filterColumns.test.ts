import { describe, expect, it } from "vitest";
import { buildColumnIndex, filterColumns } from "./filterColumns";

/** 20,000 names shaped like a wide omics table plus a handful of study columns. */
function wideNames(): string[] {
  const names = ["sample_id", "condition", "hba1c", "HbA1c_baseline", "energy_kcal", "age"];
  for (let i = 0; names.length < 20_000; i++) {
    names.push(`ENSG${String(100000 + i * 7).padStart(11, "0")}`);
  }
  return names;
}

const NAMES = wideNames();
const INDEX = buildColumnIndex(NAMES);
const matched = (q: string) => filterColumns(INDEX, q).map((i) => NAMES[i]);

describe("filterColumns over 20,000 names", () => {
  it("returns every column, in table order, for an empty or blank query", () => {
    expect(filterColumns(INDEX, "")).toHaveLength(20_000);
    expect(filterColumns(INDEX, "   ").slice(0, 3)).toEqual([0, 1, 2]);
  });

  it("matches case-insensitively as a substring", () => {
    expect(matched("hba1c")).toEqual(["hba1c", "HbA1c_baseline"]);
    expect(matched("KCAL")).toEqual(["energy_kcal"]);
  });

  it("requires every whitespace-separated token to appear", () => {
    expect(matched("hba1c base")).toEqual(["HbA1c_baseline"]);
    expect(matched("hba1c zzz")).toEqual([]);
  });

  it("ranks an exact name first, then prefixes, then other matches in table order", () => {
    const names = ["total_age", "age_group", "stage", "age", "Age_at_visit"];
    const idx = buildColumnIndex(names);
    expect(filterColumns(idx, "age").map((i) => names[i])).toEqual([
      "age",
      "age_group",
      "Age_at_visit",
      "total_age",
      "stage",
    ]);
  });

  it("narrows the gene columns by any fragment", () => {
    const hits = matched("ENSG000001000");
    expect(hits.length).toBeGreaterThan(0);
    expect(hits.every((n) => n!.startsWith("ENSG000001000"))).toBe(true);
    expect(matched("00000100007")).toEqual(["ENSG00000100007"]);
  });

  it("stays fast enough to run on every keystroke", () => {
    const queries = [
      "e",
      "en",
      "ens",
      "ensg0000",
      "ensg00000123",
      "hba",
      "a b",
      "zzz",
      "0",
      "kcal",
    ];
    const t0 = performance.now();
    for (let round = 0; round < 5; round++) for (const q of queries) filterColumns(INDEX, q);
    const perQuery = (performance.now() - t0) / (queries.length * 5);
    // A keystroke budget is ~16 ms; this is typically well under 2 ms.
    expect(perQuery).toBeLessThan(16);
  });
});
