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

  it("leaves the card's foot to Fit at Models' end", () => {
    expect(primaryFor(LOG, models, { kind: "press", held: true, estimate: null })).toEqual({ kind: "fit" });
  });

  it("is Download in Write-up once it is reached", () => {
    const writeup = stageOf(LOG, "writeup")!;
    expect(writeup.reached).toBe(true);
    expect(primaryFor(LOG as QuestLog, writeup, { kind: "none" })).toEqual({ kind: "download" });
  });
});

describe("a hint's name (P7)", () => {
  // The engine names the readings a refusal waits on by the ledger's whole sentence (core/quest.py
  // `asks`), backticks and closing period included. Who's in's eligibility is set open by
  // construction and named so; its noticing is answered, so the question is what waits.
  const SENTENCE = "Tell me about this column: `gender`: codes for categories? (two values).";
  const set = (name: (l: QuestLine) => string) =>
    withStage(LOG, "whos_in", (l) => (l.key === "exclusions" ? { ...l, status: "open", decision_id: null, name: name(l) } : answered(l)));

  it("is the line's name as running text: no backticks, and no second period", () => {
    const { log, stage } = set(() => SENTENCE);
    const hint = (primaryFor(log, stage, { kind: "none" }) as { hint: Hint }).hint;
    expect(text(hint)).toBe("Still to answer: Tell me about this column: gender: codes for categories? (two values).");
  });

  it("keeps the closing period of a name that has none", () => {
    const { log, stage } = set((l) => l.name);
    const hint = (primaryFor(log, stage, { kind: "none" }) as { hint: Hint }).hint;
    expect(text(hint)).toBe(`Still to answer: ${stage.lines.find((l) => l.key === "exclusions")!.name}.`);
  });
});

/**
 * Open noticings do not hold Continue (finding 6; FOUNDATION §7): each travels to the triage of the
 * open noticings, which stands before the held-out rows open under Predict (Results) and before the
 * plan's lock otherwise (Models; core/sweep.py `gate_stage`), and the foot says how many wait there.
 * The triage gates the opening, unchanged, so in the triage's own stage a noticing still holds it.
 */
describe("open noticings and Continue (finding 6)", () => {
  // Who's in as the engine serves it under Predict: its questions answered, one noticing open.
  const whos = stageOf(LOG, "whos_in")!;
  const noticings = whos.lines.filter((l) => l.source === "finding" && l.counted && l.status === "open");
  const results = stageOf(LOG, "results")!.name;

  it("enables Continue once the stage's questions are answered, saying how many noticings wait for the triage in Results", () => {
    expect(LOG.fit?.purpose).toBe("prediction");
    expect(noticings.length).toBe(1);
    expect(whos.lines.filter((l) => l.source !== "finding" && l.label === "Decide" && l.counted).every((l) => l.status === "answered")).toBe(true);
    const p = primaryFor(LOG, whos, { kind: "none" });
    expect(p).toMatchObject({ kind: "continue", disabled: false, label: "Continue" });
    expect((p as { note?: string }).note).toBe(`1 noticing waits for the triage in ${results}.`);
  });

  it("counts every open noticing, and says so in the plural", () => {
    const second = { ...noticings[0]!, id: "finding:second", key: "second", order: noticings[0]!.order + 0.1 };
    const two = { ...whos, lines: [...whos.lines, second] };
    const p = primaryFor({ ...LOG, stages: LOG.stages.map((s) => (s.key === "whos_in" ? two : s)) }, two, { kind: "none" });
    expect((p as { note?: string }).note).toBe(`2 noticings wait for the triage in ${results}.`);
  });

  it("says nothing of the triage where no noticing is open", () => {
    const { log, stage } = withStage(LOG, "whos_in", answered);
    const p = primaryFor(log, stage, { kind: "none" });
    expect(p).toMatchObject({ kind: "continue", disabled: false });
    expect((p as { note?: string }).note).toBeUndefined();
  });

  it("still holds Continue on a question left open beside the noticing", () => {
    const { log, stage } = withStage(LOG, "whos_in", (l) => (l.key === "missing" ? { ...l, status: "open", decision_id: null } : l));
    expect(primaryFor(log, stage, { kind: "none" })).toMatchObject({
      kind: "continue",
      disabled: true,
      hint: { lead: "Still to answer: ", to: { stage: "whos_in", line: "missing" } },
    });
  });

  it("names the plan's lock in Models as the triage's place when the goal is not Predict", () => {
    const estimate = { ...LOG, fit: { ...LOG.fit!, purpose: "inference" } };
    const p = primaryFor(estimate, stageOf(estimate, "whos_in")!, { kind: "none" });
    expect((p as { note?: string }).note).toBe(`1 noticing waits for the triage in ${stageOf(LOG, "models")!.name}.`);
  });

  it("keeps a noticing holding Continue in the triage's own stage", () => {
    // Models under Estimate is where the triage stands: its noticing is decided there.
    const estimate = { ...LOG, fit: { ...LOG.fit!, purpose: "inference" } };
    const n = { ...noticings[0]!, id: "finding:models-noticing", key: "models-noticing" };
    const models = stageOf(estimate, "models")!;
    const decided = { ...models, sweep: null, lines: [...models.lines.map(answered), n] };
    const log = { ...estimate, stages: estimate.stages.map((s) => (s.key === "models" ? decided : s)) };
    expect(primaryFor(log, decided, { kind: "none" })).toMatchObject({
      kind: "continue",
      disabled: true,
      hint: { lead: "Still to answer: ", to: { stage: "models", line: n.key } },
    });
  });
});

/**
 * Continue goes to the next open line a card can answer (finding 7), not merely the next stage: the
 * first such line in the seven stages' order outside the stage shown (a counted Decide that is not
 * a noticing, or a stage's Confirm sweep still to confirm), else the next stage. Where that is not
 * the next stage, the foot names where it goes.
 */
describe("where Continue goes (finding 7)", () => {
  const ROLES = { key: "roles", stage: "data", name: CONTRACT.questions.roles!.name };
  // Just after the outcome, as the walk met it: Your question answered and confirmed, the roles open
  // back in Your data, and Who's in's questions waiting on them.
  const afterOutcome = ((): QuestLog => {
    const set = (key: string, f: (l: QuestLine) => QuestLine, patch: Partial<QuestStage> = {}) => (log: QuestLog) =>
      withStage(log, key, f, patch).log;
    return [
      set("question", (l) => (l.label === "Decide" ? answered(l) : l), { sweep: { ...stageOf(NOT_REACHED, "question")!.sweep!, answered: true } }),
      set("data", (l) => (l.key === "roles" ? { ...l, status: "open", decision_id: null } : l)),
      set("whos_in", (l) => (l.label === "Decide" && l.counted && l.source !== "finding" ? { ...l, status: "waiting", decision_id: null, waiting_for: [ROLES] } : l)),
    ].reduce((log, f) => f(log), NOT_REACHED);
  })();

  it("goes back to the roles in Your data once the outcome is answered, naming them", () => {
    const question = stageOf(afterOutcome, "question")!;
    expect(primaryFor(afterOutcome, question, { kind: "none" })).toEqual({
      kind: "continue",
      to: "data",
      line: "roles",
      label: "Continue",
      disabled: false,
      hint: { lead: "Next: ", name: ROLES.name, tail: ", in Your data.", to: { stage: "data", line: "roles" } },
    });
  });

  it("passes over a line no card can answer, and a noticing, to the next one a card can", () => {
    // Your question's intended use is open; a card that cannot answer it is passed over.
    const whos = stageOf(LOG, "whos_in")!;
    const intended = stageOf(LOG, "question")!.lines.find((l) => l.key === "set_intended_use")!;
    expect(intended.status).toBe("open");
    const p = primaryFor(LOG, whos, { kind: "none" });
    expect(p).toMatchObject({ kind: "continue", to: "question", line: "set_intended_use", disabled: false });
    expect(text((p as { hint: Hint }).hint)).toBe(`Next: ${intended.name}, in ${stageOf(LOG, "question")!.name}.`);
    // No card answers the intended use, and Your question's sweep is confirmed: the next is Models'
    // selection, in the next stage, unnamed.
    const swept = withStage(LOG, "question", (l) => l, { sweep: { ...stageOf(LOG, "question")!.sweep!, answered: true } }).log;
    const q = primaryFor(swept, stageOf(swept, "whos_in")!, { kind: "none" }, (l) => l.key !== "set_intended_use");
    expect(q).toMatchObject({ kind: "continue", to: "models", line: "set_selection", label: "Continue", disabled: false, hint: null });
  });

  it("goes to a stage's Confirm sweep still to confirm", () => {
    const { log } = withStage(LOG, "question", (l) => (l.label === "Decide" ? answered(l) : l));
    const sweep = stageOf(log, "question")!.sweep!;
    expect(sweep.answered).toBe(false);
    const p = primaryFor(log, stageOf(log, "whos_in")!, { kind: "none" });
    expect(p).toMatchObject({ kind: "continue", to: "question", disabled: false });
    expect((p as { line?: string }).line).toBeUndefined();
    expect(text((p as { hint: Hint }).hint)).toBe(`Next: the choice set for you, in ${stageOf(log, "question")!.name}.`);
  });

  it("is Continue to the next stage, unnamed, where no open line is left to answer", () => {
    const done = LOG.stages.reduce((log, s) => withStage(log, s.key, answered, { sweep: s.sweep ? { ...s.sweep, answered: true } : null }).log, LOG);
    const p = primaryFor(done, stageOf(done, "first_look")!, { kind: "none" });
    expect(p).toEqual({ kind: "continue", to: "whos_in", label: "Continue", disabled: false, hint: null });
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
    // Every earlier line answered too, so Continue's next open line is none (finding 7): Write-up.
    const done = LOG.stages.reduce((log, s) => withStage(log, s.key, answered, { sweep: s.sweep ? { ...s.sweep, answered: true } : null }).log, LOG);
    const stage = stageOf(done, "results")!;
    expect(primaryFor(done, stage, { kind: "pressed" })).toMatchObject({ kind: "continue", to: "writeup", disabled: false });
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
