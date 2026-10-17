/**
 * The shell's reading of the quest log: "Confirm all" sends exactly what the engine's `sweep_lines`
 * records for the stage (fixtures/quest-contract.json), the card opens on the first stage still
 * working, and the one primary action follows the stage's own fields.
 *
 * The logs are the engine's (the contract's two samples under Predict), with lines' statuses set by
 * construction where a case needs a moment the samples do not hold; each expected value is read off
 * the constructed log (a name, a stage), never off the function under test.
 */
import type { QuestLine, QuestLog, QuestStage } from "../api/schema";
import { CONTRACT } from "./fixtures/contract";
import { type Hint, drawWaitsForSweep, fitWithheld, openLine, openStageKey, primaryFor, settledBy, stageOf, sweptLines } from "./log";

const strip = (lines: { id: string; key: string; value: string }[]) =>
  lines.map(({ id, key, value }) => ({ id, key, value }));

/** The sentence a hint reads as on screen. */
const text = (h: Hint | null) => (h ? `${h.lead}${h.name}${h.tail}` : null);

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

const LOG = CONTRACT.logs.results_reached.log;
const NOT_REACHED = CONTRACT.logs.models_open.log;

/** A stage of the log with each line `set` and the stage `patch`ed, and the log holding it. */
function withStage(log: QuestLog, key: string, set: (l: QuestLine) => QuestLine, patch: Partial<QuestStage> = {}): { log: QuestLog; stage: QuestStage } {
  const stage = { ...stageOf(log, key)!, ...patch };
  const next = { ...stage, lines: stage.lines.map(set) };
  return { log: { ...log, stages: log.stages.map((s) => (s.key === key ? next : s)) }, stage: next };
}
const answered = (l: QuestLine): QuestLine => (l.label === "Decide" ? { ...l, status: "answered", decision_id: `r-${l.key}` } : l);

describe("the primary action", () => {
  const models = stageOf(LOG, "models")!;

  it("waits, naming what is still to answer and linking to it, while a counted Decide is open", () => {
    const open = openLine(models)!;
    expect(open.key).toBe("set_selection");
    expect(primaryFor(LOG, models, { kind: "none" })).toEqual({
      kind: "continue",
      to: null,
      label: "Continue",
      disabled: true,
      hint: { lead: "Still to answer: ", name: open.name, tail: ".", to: { stage: "models", line: "set_selection" } },
    });
  });

  it("is the sweep's own action once the Decides are answered", () => {
    const decided: QuestStage = { ...models, lines: models.lines.map(answered) };
    expect(primaryFor(LOG, decided, { kind: "none" })).toEqual({
      kind: "confirm",
      label: models.sweep!.action,
      heading: models.sweep!.heading,
    });
  });

  it("leaves the card's foot to Fit at Models' end, and is Continue to the next stage otherwise", () => {
    expect(primaryFor(LOG, models, { kind: "press", held: true, estimate: null })).toEqual({ kind: "fit" });
    const firstLook = stageOf(LOG, "first_look")!;
    expect(primaryFor(LOG, firstLook, { kind: "none" })).toEqual({
      kind: "continue",
      to: "whos_in",
      label: "Continue",
      disabled: false,
      hint: null,
    });
  });

  it("is Download in Write-up once it is reached", () => {
    const writeup = stageOf(LOG, "writeup")!;
    expect(writeup.reached).toBe(true);
    expect(primaryFor(LOG as QuestLog, writeup, { kind: "none" })).toEqual({ kind: "download" });
  });
});

describe("a hint's name (P7)", () => {
  // The engine names a noticing by its whole sentence, backticks and closing period included.
  const SENTENCE = "`gender` is two-level text (`female`, `male`); which level counts as 1 is not asked yet.";
  const { log, stage } = withStage(LOG, "whos_in", (l) => (l.source === "finding" ? { ...l, name: SENTENCE } : answered(l)));

  it("is the line's name as running text: no backticks, and no second period", () => {
    const hint = (primaryFor(log, stage, { kind: "none" }) as { hint: Hint }).hint;
    expect(text(hint)).toBe("Still to answer: gender is two-level text (female, male); which level counts as 1 is not asked yet.");
  });

  it("keeps the closing period of a name that has none", () => {
    const plain = withStage(LOG, "whos_in", (l) => (l.source === "finding" ? l : answered(l)));
    const hint = (primaryFor(plain.log, plain.stage, { kind: "none" }) as { hint: Hint }).hint;
    expect(text(hint)).toBe(`Still to answer: ${plain.stage.lines.find((l) => l.source === "finding")!.name}.`);
  });
});

describe("a stage whose lines wait on another stage (P4)", () => {
  const OUTCOME = { key: "target", stage: "question", name: CONTRACT.questions.target!.name };
  const ROLES = { key: "roles", stage: "data", name: CONTRACT.questions.roles!.name };

  it("goes on to the stage that asks what they wait for, the next one, with Continue", () => {
    const { log, stage } = withStage(NOT_REACHED, "data", (l) =>
      l.key === "roles" ? { ...l, status: "waiting", decision_id: null, waiting_for: [OUTCOME] } : l,
    );
    expect(primaryFor(log, stage, { kind: "none" })).toEqual({
      kind: "continue",
      to: "question",
      label: "Continue",
      disabled: false,
      hint: { lead: "Waiting for ", name: OUTCOME.name, tail: ", in Your question.", to: { stage: "question", line: "target" } },
    });
  });

  it("goes back to an earlier stage by its name", () => {
    const { log, stage } = withStage(NOT_REACHED, "whos_in", (l) =>
      l.label === "Decide" && l.counted ? { ...l, status: "waiting", decision_id: null, waiting_for: [ROLES] } : l,
    );
    const p = primaryFor(log, stage, { kind: "none" });
    expect(p).toMatchObject({ kind: "continue", to: "data", label: "Go to Your data", disabled: false });
    expect(text((p as { hint: Hint }).hint)).toBe(`Waiting for ${ROLES.name}, in Your data.`);
  });

  it("stays put, saying what it waits for and linking to it, while the wait is on its own stage", () => {
    const own = { key: "set_selection", stage: "models", name: CONTRACT.declarations.set_selection!.name };
    const { log, stage } = withStage(NOT_REACHED, "models", (l) =>
      l.key === "models" ? { ...l, status: "waiting", waiting_for: [own] } : l.label === "Decide" ? answered(l) : l,
    );
    expect(primaryFor(log, stage, { kind: "none" })).toEqual({
      kind: "continue",
      to: null,
      label: "Continue",
      disabled: true,
      hint: { lead: "Waiting for ", name: own.name, tail: ".", to: { stage: "models", line: "set_selection" } },
    });
  });

  it("stays put, saying so, while a line waits on a result still computing", () => {
    const { log, stage } = withStage(NOT_REACHED, "models", (l) =>
      l.key === "models" ? { ...l, status: "waiting", waiting_for: [] } : l.label === "Decide" ? answered(l) : l,
    );
    const p = primaryFor(log, stage, { kind: "none" }) as { disabled: boolean; hint: Hint };
    expect(p.disabled).toBe(true);
    expect(text(p.hint)).toBe("Waiting for a result still computing.");
    expect(p.hint.to).toBeNull();
  });
});

describe("a stage not reached yet (P4)", () => {
  it("leads to the stage the analysis is at, saying why", () => {
    const results = stageOf(NOT_REACHED, "results")!;
    expect(results.reached).toBe(false);
    const at = NOT_REACHED.stages.find((s) => s.reached && !s.progress?.complete)!;
    const p = primaryFor(NOT_REACHED, results, { kind: "none" });
    expect(p).toMatchObject({ kind: "continue", to: at.key, label: `Go to ${at.name}`, disabled: false });
    expect(text((p as { hint: Hint }).hint)).toBe(`Results is not reached yet; the analysis is at ${at.name}.`);
  });
});

describe("the opening, in Results (P5)", () => {
  const results = stageOf(LOG, "results")!;

  it("is the foot's action while the opening is open, whatever else in Results is", () => {
    expect(results.lines.find((l) => l.key === "open_seal")!.status).toBe("open");
    expect(primaryFor(LOG, results, { kind: "pressed" })).toEqual({ kind: "open" });
  });

  it("gives way to Continue once the rows are opened", () => {
    const { log, stage } = withStage(LOG, "results", answered);
    expect(primaryFor(log, stage, { kind: "pressed" })).toMatchObject({ kind: "continue", to: "writeup", disabled: false });
  });
});

describe("Who's in's sweep before the draw, under Predict (P15)", () => {
  const SWEEP = { lines: 1, answered: false, id: "other:confirm-sweep:whos_in", heading: "Here is the other choice set for you", action: "Confirm it", confirmed_by: null, changed: [] };
  /** Who's in under Predict as the walk meets it: the stated grain in the sweep, every Decide
   *  answered but the draw, which is open. */
  const before = (sweep: typeof SWEEP | null) =>
    withStage(
      LOG,
      "whos_in",
      (l) =>
        l.key === "grain"
          ? { ...l, label: "Confirm", status: sweep?.answered ? "answered" : "set_for_you", counted: false }
          : l.key === "split"
            ? { ...l, status: "open", decision_id: null }
            : answered(l),
      { sweep },
    );

  it("holds the draw until the sweep is confirmed: the sweep is the foot's action, and the draw is not opened", () => {
    const { log, stage } = before(SWEEP);
    expect(drawWaitsForSweep(stage)).toBe(true);
    expect(openLine(stage)).toBeNull();
    expect(primaryFor(log, stage, { kind: "none" })).toEqual({ kind: "confirm", label: SWEEP.action, heading: SWEEP.heading });
  });

  it("opens the draw, the stage's last Decide, once the sweep is confirmed", () => {
    const { log, stage } = before({ ...SWEEP, answered: true });
    expect(drawWaitsForSweep(stage)).toBe(false);
    expect(openLine(stage)?.key).toBe("split");
    expect(primaryFor(log, stage, { kind: "none" })).toMatchObject({ kind: "continue", disabled: true, hint: { to: { stage: "whos_in", line: "split" } } });
  });

  it("holds nothing where the stage has no sweep, or the draw is not a Decide (under Estimate)", () => {
    expect(drawWaitsForSweep(before(null).stage)).toBe(false);
    const estimate = withStage(LOG, "whos_in", (l) => (l.key === "split" ? { ...l, label: "For the record", counted: false } : l), { sweep: SWEEP });
    expect(drawWaitsForSweep(estimate.stage)).toBe(false);
  });
});

describe("the open line (P8)", () => {
  const data = stageOf(LOG, "data")!;

  it("is never a line no card can answer, and is one that a card can", () => {
    // The data stage as the engine serves it: its counted lines answered, the join and the data
    // dictionary open and uncounted.
    expect(data.lines.filter((l) => l.status === "open" && l.label === "Decide").map((l) => l.key)).toEqual(["join_files", "import_codebook"]);
    expect(openLine(data)?.key).toBe("join_files");
    expect(openLine(data, (l) => l.source !== "declaration")).toBeNull();
    const reopened = withStage(LOG, "data", (l) => (l.key === "lens" ? { ...l, status: "open", decision_id: null } : l)).stage;
    expect(openLine(reopened, (l) => l.source !== "declaration")?.key).toBe("lens");
  });
});

describe("the line that settled a noticing (P2)", () => {
  const KEY = "pack::dietary::implausible_intake#1";
  const exclusions = stageOf(LOG, "whos_in")!.lines.find((l) => l.key === "exclusions")!;
  const settle = (id: string, also: (l: QuestLine) => QuestLine = (l) => l) =>
    withStage(LOG, "whos_in", (l) => also(l.key === KEY ? { ...l, status: "answered", decision_id: id } : l));
  const noticing = (log: QuestLog) => stageOf(log, "whos_in")!.lines.find((l) => l.key === KEY)!;

  it("is the question whose record settled it", () => {
    const { log } = settle(exclusions.decision_id!);
    expect(settledBy(log, noticing(log))?.key).toBe("exclusions");
  });

  it("is none for its own disposition, for a record several lines share, or for a line that is not a settled noticing", () => {
    const own = settle("r-own").log;
    expect(settledBy(own, noticing(own))).toBeNull();
    // A record two lines share, as a sweep's confirmation is: neither alone settled it.
    const shared = settle("r-sweep", (l) => (l.key === "exclusions" || l.key === "missing" ? { ...l, decision_id: "r-sweep" } : l)).log;
    expect(stageOf(shared, "whos_in")!.lines.filter((l) => l.source !== "finding" && l.decision_id === "r-sweep").length).toBeGreaterThan(1);
    expect(settledBy(shared, noticing(shared))).toBeNull();
    expect(settledBy(LOG, noticing(LOG))).toBeNull();
    expect(settledBy(LOG, exclusions)).toBeNull();
  });
});

describe("the fit withheld until Fit is pressed (FOUNDATION §7)", () => {
  const { ready, held, pressed, unlocked, no_goal } = CONTRACT.fit_locks;
  it("withholds it under Predict until the press, in the lock's own words", () => {
    expect(fitWithheld({ ...LOG, fit: ready })).toBe(ready.reason);
    expect(fitWithheld({ ...LOG, fit: held })).toBe(held.reason);
  });
  it("withholds nothing once Fit is pressed, under Estimate (the stage's own rule), or with no lock", () => {
    expect(fitWithheld({ ...LOG, fit: pressed })).toBeNull();
    expect(fitWithheld({ ...LOG, fit: unlocked })).toBeNull();
    expect(fitWithheld({ ...LOG, fit: no_goal })).toBeNull();
    expect(fitWithheld({ ...LOG, fit: null })).toBeNull();
  });
});
