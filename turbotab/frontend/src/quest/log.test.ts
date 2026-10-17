/**
 * The shell's reading of the quest log: "Confirm all" sends exactly what the engine's `sweep_lines`
 * records for the stage (fixtures/quest-contract.json), the card opens on the first stage still
 * working, and the one primary action follows the stage's own fields.
 */
import type { QuestLog, QuestStage } from "../api/schema";
import { CONTRACT } from "./fixtures/contract";
import { openLine, openStageKey, primaryFor, stageOf, sweptLines } from "./log";

const strip = (lines: { id: string; key: string; value: string }[]) =>
  lines.map(({ id, key, value }) => ({ id, key, value }));

describe.each(Object.entries(CONTRACT.logs))("on %s", (_name, { log, swept }) => {
  it("sends, for each sweep, the lines the engine's sweep_lines records", () => {
    const withSweep = log.stages.filter((s) => s.sweep);
    expect(withSweep.map((s) => s.key).sort()).toEqual(Object.keys(swept).sort());
    for (const s of withSweep) expect(strip(sweptLines(s)), s.key).toEqual(strip(swept[s.key]!));
  });

  it("opens on the first reached stage that is not complete", () => {
    const first = log.stages.find((s) => s.reached && !s.progress?.complete)!;
    expect(openStageKey(log)).toBe(first.key);
  });
});

describe("the primary action", () => {
  const log = CONTRACT.logs.results_reached.log;
  const models = stageOf(log, "models")!;

  it("waits, naming what is still to answer, while a counted Decide is open", () => {
    const open = openLine(models)!;
    expect(open.key).toBe("set_selection");
    expect(primaryFor(log, models, { kind: "none" })).toEqual({
      kind: "continue",
      to: null,
      disabled: true,
      hint: `Still to answer: ${open.name}.`,
    });
  });

  it("is the sweep's own action once the Decides are answered", () => {
    const decided: QuestStage = {
      ...models,
      lines: models.lines.map((l) => (l.label === "Decide" ? { ...l, status: "answered" as const } : l)),
    };
    expect(primaryFor(log, decided, { kind: "none" })).toEqual({
      kind: "confirm",
      label: models.sweep!.action,
      heading: models.sweep!.heading,
    });
  });

  it("leaves the card's foot to Fit at Models' end, and is Continue to the next stage otherwise", () => {
    expect(primaryFor(log, models, { kind: "press", held: true, estimate: null })).toEqual({ kind: "fit" });
    const firstLook = stageOf(log, "first_look")!;
    expect(primaryFor(log, firstLook, { kind: "none" })).toEqual({
      kind: "continue",
      to: "whos_in",
      disabled: false,
      hint: null,
    });
  });

  it("is Download in Write-up once it is reached", () => {
    const writeup = stageOf(log, "writeup")!;
    expect(writeup.reached).toBe(true);
    expect(primaryFor(log as QuestLog, writeup, { kind: "none" })).toEqual({ kind: "download" });
  });
});
