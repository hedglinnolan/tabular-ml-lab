/**
 * The exclusions card's rule by level (SWARM_PLAN §5; decisions.py LevelRule): the levels it
 * excludes and the codes that record a refused or a don't-know answer, each its own list, posted
 * only when the server would accept it. The expected rules are written out here.
 */
import { levelRule, parseLevels } from "./ChoiceQuestions";

describe("a rule by level", () => {
  it("names the excluded levels and the refused and don't-know codes apart", () => {
    expect(levelRule("DIQ010", "1", " 7 ", "9, 99")).toEqual({
      kind: "levels",
      column: "DIQ010",
      exclude: ["1"],
      refused: ["7"],
      dont_know: ["9", "99"],
      reason: "not in the study population (your own levels)",
      missing: "exclude",
    });
  });

  it("is not posted while it excludes no level or names one level twice", () => {
    expect(levelRule("DIQ010", "", "7", "9")).toBeNull();
    expect(levelRule("DIQ010", "1, 7", "7", "9")).toBeNull();
    expect(levelRule("DIQ010", "1", "9.0", "9")).toBeNull(); // 9.0 and 9 are one level
    expect(levelRule("", "1", "", "")).toBeNull();
  });

  it("reads levels by commas, blanks dropped", () => {
    expect(parseLevels(" Yes , ,No")).toEqual(["Yes", "No"]);
    expect(parseLevels("")).toEqual([]);
  });
});
