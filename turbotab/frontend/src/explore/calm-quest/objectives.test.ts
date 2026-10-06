import { expect, it } from "vitest";
import { SCENARIO_ANSWERS, initial, manuscript, reduce } from "../calm-kit";
import { kickerOf, nextObjective, questSections } from "./objectives";

it("lists every asked slot once by guideline section, each objective named apart, and counts them as they are recorded", () => {
  let s = initial();
  const start = questSections(manuscript(s), s.open);
  expect(start.map((x) => [x.id, x.done, x.objectives.length])).toEqual([
    ["participants", 0, 2],
    ["variables", 0, 6],
    ["measurement", 0, 6],
    ["statistics", 0, 4],
    ["results", 0, 0],
  ]);
  for (const sec of start)
    expect(new Set(sec.objectives.map((o) => o.name)).size).toBe(sec.objectives.length);
  expect(start[1]!.objectives.map((o) => o.name)).toContain(
    "Adjustment set: protein, carb and 4 more",
  );
  expect(
    start
      .flatMap((x) => x.objectives)
      .filter((o) => o.status === "open")
      .map((o) => o.id),
  ).toEqual(["unit"]);
  expect(kickerOf("effect")).toBe("Variables · Exposure and estimand · step 2 of 3");

  for (const [step, option] of SCENARIO_ANSWERS.slice(0, 3))
    s = reduce(s, { type: "record", step, option });
  expect(nextObjective(s)).toBeNull();
  s = reduce(s, { type: "open", step: "exclusions" });
  expect(nextObjective(s)).toBe("missing");

  for (const [step, option] of SCENARIO_ANSWERS.slice(3))
    s = reduce(s, { type: "record", step, option });
  const done = questSections(manuscript(s), s.open);
  expect(done.every((x) => x.done === x.objectives.length)).toBe(true);
  expect(done.find((x) => x.id === "results")!.current).toBe(true);
  expect(nextObjective(s)).toBeNull();
});
