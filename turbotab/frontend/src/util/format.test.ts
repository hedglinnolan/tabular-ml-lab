import { codeLike, fmtCode, fmtStat } from "./format";

describe("identifiers and years", () => {
  it("recognizes them by name, or as whole numbers in the range of years", () => {
    expect(codeLike("SEQN")).toBe(true);
    expect(codeLike("participant_id")).toBe(true);
    expect(codeLike("cycle_begin_year")).toBe(true);
    expect(codeLike("kcal", { min: 0, max: 15594, dtype: "numeric" })).toBe(false);
    expect(codeLike("visit_yr")).toBe(true);
    expect(codeLike("exam", { min: 1999, max: 2017, dtype: "integer" })).toBe(true);
    expect(codeLike("age", { min: 18, max: 85, dtype: "integer" })).toBe(false);
  });

  it("writes them without digit grouping, unlike quantities", () => {
    expect(fmtCode(9966)).toBe("9966");
    expect(fmtCode(2001)).toBe("2001");
    expect(fmtStat(2001)).toBe("2,001");
  });
});
