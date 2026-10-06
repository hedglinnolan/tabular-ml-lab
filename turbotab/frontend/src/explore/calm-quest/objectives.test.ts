import { expect, it } from "vitest";
import { SCENARIO_ANSWERS, STEPS, initial, reduce } from "../calm-kit";
import { nextObjective, questSections } from "./objectives";

it("lists every question once by section, named by its plain question, and counts them as they are answered", () => {
  let s = initial();
  const start = questSections(s);
  expect(start.map((x) => [x.id, x.done, x.objectives.length])).toEqual([
    ["participants", 0, 2],
    ["variables", 0, 8],
    ["measurement", 0, 6],
    ["statistics", 0, 4],
    ["results", 0, 0],
  ]);
  const names = start.flatMap((x) => x.objectives.map((o) => o.name));
  expect(new Set(names).size).toBe(names.length);
  // The card's register, not the manuscript's: no head ("Adjustment set") and no data-value marks.
  const heads = new Set(STEPS.map((x) => x.head));
  expect(names.filter((n) => heads.has(n) || n.includes("`"))).toEqual([]);
  expect(start[1]!.objectives.map((o) => o.name)).toContain(
    "How do age and gender relate to sugar and glucose?",
  );
  expect(
    start
      .flatMap((x) => x.objectives)
      .filter((o) => o.status === "open")
      .map((o) => o.id),
  ).toEqual(["unit"]);

  for (const [step, option] of SCENARIO_ANSWERS.slice(0, 3))
    s = reduce(s, { type: "record", step, option });
  expect(nextObjective(s)).toBeNull();
  s = reduce(s, { type: "open", step: "exclusions" });
  expect(nextObjective(s)).toBe("missing");

  for (const [step, option] of SCENARIO_ANSWERS.slice(3))
    s = reduce(s, { type: "record", step, option });
  const done = questSections(s);
  expect(done.every((x) => x.done === x.objectives.length)).toBe(true);
  expect(done.find((x) => x.id === "results")!.current).toBe(true);
  expect(nextObjective(s)).toBeNull();
});
