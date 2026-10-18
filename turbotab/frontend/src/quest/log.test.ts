/**
 * The shell's reading of the quest log: "Confirm all" sends exactly what the engine's `sweep_lines`
 * records for the stage (fixtures/quest-contract.json), the card opens on the first stage still
 * working, and the one primary action follows the stage's own fields.
 *
 * The logs are the engine's (the contract's two samples under Predict), with lines' statuses set by
 * construction where a case needs a moment the samples do not hold; each expected value is read off
 * the constructed log (a name, a stage), never off the function under test.
 */
import type { Finding, FindingsArtifact, QuestLine, QuestLog, QuestStage } from "../api/schema";
import prediction from "../mocks/fixtures/m3-nhanes-prediction.json";
import { artifactAt, expandView, viewsOf, type M3Fixture } from "../mocks/m3";
import { mockQuest } from "../mocks/quest";
import { CONTRACT } from "./fixtures/contract";
import { type Hint, drawWaitsForSweep, filled, fitPosition, fitWithheld, nextOpen, openLine, openStageKey, primaryFor, readsDone, segments, settledBy, stageOf, sweptLines } from "./log";

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
 * Under Predict open noticings do not hold Continue (finding 6; FOUNDATION §7): each travels to the
 * triage of the open noticings, which stands before the held-out rows open, in Results
 * (core/sweep.py `gate_stage`), and the foot says how many wait there. Under Estimate the engine's
 * triage stands before the plan's lock, in Models, which the shell does not draw yet (W1-R R3 P5):
 * a noticing holds its own stage, as before finding 6, and the foot says it is decided there.
 * Before the goal is chosen no noticing holds Continue either (W1-R M1: under Predict the goal is
 * chosen in Your question, after Your data's noticings), and the foot names no triage, since where
 * they are decided waits on the goal.
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

  it("holds Continue on the noticing under Estimate, naming no triage, since the shell draws none there (W1-R R3 P5)", () => {
    // Under Estimate the engine's triage stands before the plan's lock, in a Models the shell does
    // not draw yet (W3): the noticing is decided in its own stage.
    const log = { ...LOG, fit: { ...LOG.fit!, purpose: "inference" } } as QuestLog;
    const p = primaryFor(log, stageOf(log, "whos_in")!, { kind: "none" });
    expect(p).toEqual({
      kind: "continue",
      to: null,
      label: "Continue",
      disabled: true,
      hint: { lead: "Still to answer: ", name: noticings[0]!.name.replaceAll("`", "").replace(/\.$/, ""), tail: ".", to: { stage: "whos_in", line: noticings[0]!.key } },
      note: "1 noticing is decided in this stage.",
    });
    expect(JSON.stringify(p)).not.toContain("triage");
    // A log with no Fit lock names no goal: the noticing holds its stage there too.
    const unlocked = { ...LOG, fit: null } as QuestLog;
    expect(primaryFor(unlocked, stageOf(unlocked, "whos_in")!, { kind: "none" })).toEqual(p);
  });

  it("enables Continue beside the noticing before the goal is chosen, naming no triage and no stage that decides it (W1-R M1)", () => {
    const log = { ...LOG, fit: { ...LOG.fit!, purpose: null } } as QuestLog;
    const p = primaryFor(log, stageOf(log, "whos_in")!, { kind: "none" });
    expect(p).toMatchObject({ kind: "continue", disabled: false, label: "Continue" });
    expect((p as { note?: string }).note).toBeUndefined();
    expect(JSON.stringify(p)).not.toContain("triage");
    // Continue passes over it, to the next open question, Your question's intended use.
    expect(stageOf(log, "question")!.lines.find((l) => l.key === "set_intended_use")!.status).toBe("open");
    expect(p).toMatchObject({ to: "question", line: "set_intended_use" });
  });

  it("says each of the stage's noticings is decided in it, in the plural", () => {
    const estimate = { ...LOG, fit: { ...LOG.fit!, purpose: "inference" } } as QuestLog;
    const second = { ...noticings[0]!, id: "finding:second", key: "second", order: noticings[0]!.order + 0.1 };
    const whosE = stageOf(estimate, "whos_in")!;
    const two = { ...whosE, lines: [...whosE.lines, second] };
    const p = primaryFor({ ...estimate, stages: estimate.stages.map((s) => (s.key === "whos_in" ? two : s)) }, two, { kind: "none" });
    expect((p as { note?: string }).note).toBe("2 noticings are decided in this stage.");
  });

  it("says nothing of noticings outside Predict where none is open", () => {
    const estimate = { ...LOG, fit: { ...LOG.fit!, purpose: "inference" } } as QuestLog;
    const { log, stage } = withStage(estimate, "whos_in", answered);
    const p = primaryFor(log, stage, { kind: "none" });
    expect(p).toMatchObject({ kind: "continue", disabled: false });
    expect((p as { note?: string }).note).toBeUndefined();
  });

  it("keeps a noticing holding Continue in Models under Estimate, where the engine's triage stands", () => {
    // The shell draws no triage there yet: Models' noticing is decided in Models.
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
 * The walk's own path (W1-R M1; finding 6): the real server's NHANES capture under Predict, at its
 * first moment where Your data holds open noticings and the roles wait for the outcome, before the
 * goal is chosen in Your question. The log is the mock's (dev:mock's /p/m3~nhanes-prediction~n),
 * placed by the engine's registry from the captured view and findings.
 */
describe("Your data before the goal is chosen, on the NHANES capture (W1-R M1)", () => {
  const F = prediction as unknown as M3Fixture;
  const logs = viewsOf(F).map((v) => {
    const st = v.stages.findings;
    const read = st && st.status !== "idle" && st.status !== "blocked";
    const findings = read ? ((artifactAt(F, "findings", st.key) as FindingsArtifact | null)?.findings ?? null) : null;
    return mockQuest(expandView(F, v, "m3-test", "2026-10-17T00:00:00Z"), findings as Finding[] | null, false);
  });
  const atData = logs.find((log) => {
    const data = stageOf(log, "data")!;
    const open = data.lines.filter((l) => l.counted && l.status !== "answered");
    return log.fit?.purpose === null && open.some((l) => l.source === "finding") && open.some((l) => l.key === "roles" && l.status === "waiting");
  })!;
  const data = stageOf(atData, "data")!;
  const roles = data.lines.find((l) => l.key === "roles")!;

  it("is the capture's moment: noticings open, the roles waiting on Your question, no goal", () => {
    expect(atData).toBeDefined();
    expect(atData.fit).toMatchObject({ purpose: null });
    const left = data.lines.filter((l) => l.label === "Decide" && l.counted && l.status !== "answered");
    expect(left.filter((l) => l.source !== "finding").map((l) => l.key)).toEqual(["roles"]);
    expect(roles.waiting_for[0]!.stage).toBe("question");
  });

  it("enables Continue to what the roles wait for, saying no noticing holds it or where one is decided", () => {
    const p = primaryFor(atData, data, { kind: "none" });
    const on = roles.waiting_for[0]!;
    expect(p).toEqual({
      kind: "continue",
      to: on.stage,
      label: "Continue",
      disabled: false,
      hint: { lead: "Waiting for ", name: on.name.replaceAll("`", "").replace(/\.$/, ""), tail: `, in ${stageOf(atData, on.stage)!.name}.`, to: { stage: on.stage, line: on.key } },
    });
  });

  it("holds Continue on the first noticing under Estimate, where Your data decides it", () => {
    const estimate = { ...atData, fit: { ...atData.fit!, purpose: "inference" } } as QuestLog;
    const first = data.lines
      .filter((l) => l.label === "Decide" && l.counted && l.status === "open")
      .sort((a, b) => a.order - b.order)[0]!;
    expect(first.source).toBe("finding");
    expect(primaryFor(estimate, data, { kind: "none" })).toMatchObject({
      kind: "continue",
      disabled: true,
      hint: { lead: "Still to answer: ", to: { stage: "data", line: first.key } },
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

/**
 * Where the analysis is, beside open noticings (finding 6): under Predict they travel to the triage,
 * so outside the triage's stage a stage whose only lines still to answer are noticings is passed
 * (its Continue is enabled) and is not where the analysis is: the shell lands past it, and a stage
 * not reached yet leads past it. Under Predict the engine leaves such a noticing out of its stage's
 * count (ruling R-a, core/quest.py `quest_log`), so the stage is served complete; before the goal
 * is chosen the engine counts it in its stage, as under Estimate, and the shell passes the stage
 * itself (`passedForTriage`, W1-R M1). In the triage's own stage, beside a question still open,
 * and outside Predict (W1-R R3 P5), the noticing holds the stage as before.
 *
 * The logs are the contract's, served under Predict; a log under another goal is served as the
 * engine serves it there, its open noticings counted in their stage's progress (`counted`).
 */
describe("where the analysis is, beside open noticings (finding 6)", () => {
  /** Your question answered and confirmed, so Who's in, with only its noticing open, comes first. */
  const questionDone = (log: QuestLog) => {
    const q = stageOf(log, "question")!;
    return withStage(log, "question", answered, {
      sweep: { ...q.sweep!, answered: true },
      progress: { answered: q.progress!.required, required: q.progress!.required, complete: true },
    }).log;
  };
  const unansweredOf = (s: QuestStage) => s.lines.filter((l) => l.label === "Decide" && l.counted && (l.status === "open" || l.status === "waiting"));
  const openNoticingsOf = (s: QuestStage) => unansweredOf(s).filter((l) => l.source === "finding" && l.status === "open");
  /** The Predict log served under `purpose` instead: outside Predict the engine counts each stage's
   *  open noticings in its progress (ruling R-a holds under Predict alone). */
  const servedUnder = (log: QuestLog, purpose: string | null): QuestLog => ({
    ...log,
    fit: { ...log.fit!, purpose },
    stages: log.stages.map((s) => {
      const n = openNoticingsOf(s).length;
      // Results' own are counted under Predict too (it is the triage's stage).
      if (!s.progress || !n || s.key === "results") return s;
      const required = s.progress.required + n;
      return { ...s, progress: { ...s.progress, required, complete: s.progress.answered >= required } };
    }),
  } as QuestLog);
  /** Who's in's only lines still to answer are its noticings, served as the log's goal has them:
   *  under Predict complete (R-a), otherwise counted and so not complete. */
  const onlyNoticing = (log: QuestLog) => {
    const whos = stageOf(log, "whos_in")!;
    const left = unansweredOf(whos);
    expect(left.length).toBeGreaterThan(0);
    expect(left.every((l) => l.source === "finding" && l.status === "open")).toBe(true);
    expect(whos.progress?.complete).toBe(log.fit?.purpose === "prediction");
  };
  /** The first reached stage after Who's in that is not complete, read off the log. */
  const pastWhosIn = (log: QuestLog) => {
    const i = log.stages.findIndex((s) => s.key === "whos_in");
    return log.stages.slice(i + 1).find((s) => s.reached && !s.progress?.complete)!;
  };

  it("lands past a stage whose only open lines are noticings that travel to the triage", () => {
    const log = questionDone(LOG);
    expect(LOG.fit?.purpose).toBe("prediction");
    onlyNoticing(log);
    const at = pastWhosIn(log);
    expect(at.key).toBe("models");
    expect(openStageKey(log)).toBe(at.key);
  });

  it("leads a stage not reached yet past it too, naming where the analysis is", () => {
    const log = questionDone(NOT_REACHED);
    onlyNoticing(log);
    const at = pastWhosIn(log);
    const results = stageOf(log, "results")!;
    expect(results.reached).toBe(false);
    const p = primaryFor(log, results, { kind: "none" });
    expect(p).toMatchObject({ kind: "continue", to: at.key, label: `Go to ${at.name}` });
    expect(text((p as { hint: Hint }).hint)).toBe(`Results is not reached yet; the analysis is at ${at.name}.`);
  });

  it("stays at a stage whose noticing has a question still open beside it, or a sweep still to confirm", () => {
    // Each served as the engine counts it: the question reopened is one answered fewer, the sweep
    // one more required (core/quest.py `quest_log`).
    const whos = stageOf(LOG, "whos_in")!.progress!;
    const log = withStage(questionDone(LOG), "whos_in", (l) => (l.key === "missing" ? { ...l, status: "open", decision_id: null } : l), {
      progress: { answered: whos.answered - 1, required: whos.required, complete: false },
    }).log;
    expect(stageOf(log, "whos_in")!.lines.some((l) => l.source === "finding" && l.status === "open")).toBe(true);
    expect(openStageKey(log)).toBe("whos_in");
    const sweep = { ...stageOf(LOG, "question")!.sweep!, id: "other:confirm-sweep:whos_in", lines: 1, answered: false };
    const swept = withStage(questionDone(LOG), "whos_in", (l) => l, {
      sweep,
      progress: { answered: whos.answered, required: whos.required + 1, complete: false },
    }).log;
    expect(openStageKey(swept)).toBe("whos_in");
  });

  it("stays at a stage whose only open lines are noticings under Estimate (W1-R R3 P5)", () => {
    const estimate = servedUnder(questionDone(LOG), "inference");
    onlyNoticing(estimate);
    expect(openStageKey(estimate)).toBe("whos_in");
  });

  it("lands past that stage before the goal is chosen, as under Predict (W1-R M1)", () => {
    // The engine counts the noticing in Who's in with no goal chosen, so the stage is served not
    // complete: the shell passes it itself.
    const unchosen = servedUnder(questionDone(LOG), null);
    onlyNoticing(unchosen);
    expect(openStageKey(unchosen)).toBe(pastWhosIn(questionDone(LOG)).key);
  });

  it("stays at Models under Estimate, where the noticing is decided, and passes it for Results under Predict", () => {
    // Under Estimate a noticing in Models, every other line answered.
    const estimate = { ...questionDone(LOG), fit: { ...LOG.fit!, purpose: "inference" } } as QuestLog;
    const n = stageOf(estimate, "whos_in")!.lines.find((l) => l.source === "finding" && l.status === "open")!;
    const whosDone = withStage(estimate, "whos_in", answered, { progress: { answered: 5, required: 5, complete: true } }).log;
    const decided = withStage(whosDone, "models", answered, { sweep: null }).log;
    const models = stageOf(decided, "models")!;
    const log = withStage(decided, "models", (l) => l, {
      lines: [...models.lines, { ...n, id: "finding:models-noticing", key: "models-noticing" }],
      progress: { answered: 3, required: 4, complete: false },
    }).log;
    expect(unansweredOf(stageOf(log, "models")!).map((l) => l.key)).toEqual(["models-noticing"]);
    expect(openStageKey(log)).toBe("models");
    // Under Predict the same Models is passed for Results, the triage's stage there.
    const predict = { ...log, fit: LOG.fit } as QuestLog;
    expect(openStageKey(predict)).toBe("results");
  });
});

/** Each stage in `keys` with every Decide answered, its sweep confirmed and its progress complete. */
function doneIn(log: QuestLog, keys: string[]): QuestLog {
  return keys.reduce((acc, key) => {
    const s = stageOf(acc, key)!;
    const required = s.progress?.required ?? 0;
    return withStage(acc, key, answered, {
      sweep: s.sweep ? { ...s.sweep, answered: true } : null,
      progress: s.reached ? { answered: required, required, complete: true } : s.progress,
    }).log;
  }, log);
}

/**
 * Fit beside a noticing in Models (ruling R-a, W1-R R3 P1): under Predict a noticing's triage is in
 * Results, so Models' noticing does not hold Fit back: Fit is offered once Models' questions and
 * sweep are answered, the foot names Results' triage, and Continue no longer loops between Models
 * and Results. Under Estimate Models' noticing is decided in Models, and Fit waits for it.
 */
describe("Fit beside a noticing in Models (ruling R-a)", () => {
  // Models under Predict before the press, as the engine serves it there: every question and the
  // sweep answered, one open noticing, which the engine counts, so its progress is not complete.
  const NOTICING = stageOf(LOG, "whos_in")!.lines.find((l) => l.source === "finding" && l.status === "open")!;
  const withNoticing = (log: QuestLog, also: (l: QuestLine) => QuestLine = (l) => l) => {
    const before = doneIn(log, ["question", "whos_in", "models"]);
    const models = stageOf(before, "models")!;
    return withStage(before, "models", also, {
      lines: [...models.lines, { ...NOTICING, id: "finding:models-noticing", key: "models-noticing" }],
      progress: { answered: models.progress!.required, required: models.progress!.required + 1, complete: false },
    });
  };

  it("offers Fit under Predict, its foot naming the noticing's triage in Results", () => {
    const { log, stage } = withNoticing(NOT_REACHED);
    expect(log.fit?.purpose).toBe("prediction");
    expect(log.fit?.pressed).toBe(false);
    expect(stage.progress?.complete).toBe(false);
    const fit = fitPosition(log, undefined);
    expect(fit).toEqual({ kind: "press", held: log.fit!.held, estimate: null });
    expect(primaryFor(log, stage, fit)).toEqual({ kind: "fit", note: `1 noticing waits for the triage in ${stageOf(log, "results")!.name}.` });
    // Results, not reached before the press, leads back to Models, where Fit is: no loop.
    expect(primaryFor(log, stageOf(log, "results")!, fit)).toMatchObject({ kind: "continue", to: "models" });
  });

  it("holds Fit back under Estimate, where Models' noticing is decided in Models", () => {
    const { log: predict, stage } = withNoticing(NOT_REACHED);
    const log = { ...predict, fit: { ...predict.fit!, purpose: "inference" } } as QuestLog;
    const fit = fitPosition(log, undefined);
    expect(fit).toEqual({ kind: "none" });
    expect(primaryFor(log, stage, fit)).toMatchObject({
      kind: "continue",
      disabled: true,
      hint: { lead: "Still to answer: ", to: { stage: "models", line: "models-noticing" } },
    });
  });

  it("holds Fit back under Predict while a question in Models is still open beside the noticing", () => {
    const { log } = withNoticing(NOT_REACHED, (l) => (l.key === "set_selection" ? { ...l, status: "open", decision_id: null } : l));
    expect(fitPosition(log, undefined)).toEqual({ kind: "none" });
  });

  it("leaves the fit's foot without a note where Models has no open noticing", () => {
    const log = doneIn(NOT_REACHED, ["question", "whos_in", "models"]);
    const fit = fitPosition(log, undefined);
    expect(fit.kind).toBe("press");
    expect(primaryFor(log, stageOf(log, "models")!, fit)).toEqual({ kind: "fit" });
  });
});

/**
 * nextOpen's three exclusions (W1-R R3 P3), each where it fires and where it stays silent: under
 * Predict an open noticing travels to the triage, so Continue passes over it; the draw while it
 * waits for Who's in's sweep is not the next line, the sweep is; and a stage not reached yet is
 * skipped, whatever line in it is open.
 */
describe("the next open line's exclusions (W1-R R3 P3)", () => {
  it("passes over an open noticing in an earlier stage under Predict, to the next question", () => {
    const log = doneIn(LOG, ["question"]);
    expect(stageOf(log, "whos_in")!.lines.find((l) => l.source === "finding")!.status).toBe("open");
    const to = nextOpen(log, "first_look");
    expect(to?.stage.key).toBe("models");
    expect(to?.line?.key).toBe("set_selection");
  });

  it("leads to that noticing under Estimate, where it is decided in its own stage", () => {
    const log = { ...doneIn(LOG, ["question"]), fit: { ...LOG.fit!, purpose: "inference" } } as QuestLog;
    const noticing = stageOf(log, "whos_in")!.lines.find((l) => l.source === "finding")!;
    const to = nextOpen(log, "first_look");
    expect(to?.stage.key).toBe("whos_in");
    expect(to?.line?.key).toBe(noticing.key);
  });

  it("passes over that noticing before the goal is chosen (W1-R M1)", () => {
    const log = { ...doneIn(LOG, ["question"]), fit: { ...LOG.fit!, purpose: null } } as QuestLog;
    const to = nextOpen(log, "first_look");
    expect(to?.stage.key).toBe("models");
    expect(to?.line?.key).toBe("set_selection");
  });

  const SWEEP = { lines: 1, answered: false, id: "other:confirm-sweep:whos_in", heading: "Here is the other choice set for you", action: "Confirm it", confirmed_by: null, changed: [] };
  /** Who's in under Predict with every Decide answered but the draw, and its sweep as `sweep`. */
  const drawOpen = (sweep: typeof SWEEP) =>
    withStage(doneIn(LOG, ["question"]), "whos_in", (l) => (l.key === "split" ? { ...l, status: "open", decision_id: null } : answered(l)), { sweep }).log;

  it("leads to Who's in's sweep, not to the draw that waits for it", () => {
    const log = drawOpen(SWEEP);
    expect(nextOpen(log, "first_look")).toEqual({ stage: stageOf(log, "whos_in"), line: null });
  });

  it("leads to the draw once the sweep is confirmed", () => {
    const log = drawOpen({ ...SWEEP, answered: true });
    const to = nextOpen(log, "first_look");
    expect(to?.stage.key).toBe("whos_in");
    expect(to?.line?.key).toBe("split");
  });

  /** Every stage before Results done, and Results' updating question open, Results `reached` or not. */
  const resultsOpen = (reached: boolean) => {
    const log = doneIn(NOT_REACHED, ["question", "whos_in", "models"]);
    return withStage(log, "results", (l) => (l.key === "set_updating" ? { ...l, status: "open", waiting_for: [] } : l), { reached }).log;
  };

  it("skips an open line in a stage not reached yet", () => {
    const log = resultsOpen(false);
    expect(stageOf(log, "results")!.lines.find((l) => l.key === "set_updating")).toMatchObject({ status: "open", counted: true });
    expect(nextOpen(log, "first_look")).toBeNull();
  });

  it("leads to that line once its stage is reached", () => {
    const log = resultsOpen(true);
    const to = nextOpen(log, "first_look");
    expect(to?.stage.key).toBe("results");
    expect(to?.line?.key).toBe("set_updating");
  });
});

/**
 * A stage reads as done only once reached (finding 25): under Predict the engine reaches Write-up
 * with Results, at 0 of 0, and First look with the stages around it; neither reads as done while
 * the analysis is at an earlier stage. A stage whose own objectives are all answered reads as done.
 */
describe("a stage that reads as done (finding 25)", () => {
  /** Each stage in `keys` with no objective answered, as before the researcher came to it. */
  const untouched = (log: QuestLog, keys: string[]) =>
    keys.reduce((acc, key) => {
      const s = stageOf(acc, key)!;
      return withStage(acc, key, (l) => (l.label === "Decide" && l.counted ? { ...l, status: "open", decision_id: null } : l), {
        sweep: s.sweep ? { ...s.sweep, answered: false } : null,
        progress: { answered: 0, required: s.progress!.required, complete: false },
      }).log;
    }, log);

  it("is never a stage that asks nothing while the analysis is before it and nothing past it is answered", () => {
    const log = untouched(LOG, ["whos_in", "models"]);
    const writeup = stageOf(log, "writeup")!;
    expect(writeup).toMatchObject({ reached: true, progress: { answered: 0, required: 0, complete: true } });
    expect(stageOf(log, "question")!.progress?.complete).toBe(false);
    expect(stageOf(log, "results")!.progress?.answered).toBe(0);
    expect(readsDone(log, writeup)).toBe(false);
    expect(readsDone(log, stageOf(log, "first_look")!)).toBe(false);
    const seg = segments(log).find((x) => x.key === "writeup")!;
    // It does not read as done, while the served field says what GET /quest says
    expect(seg.done).toBe(false);
    expect(seg.complete).toBe(true);
    expect(filled(seg)).toBe(0);
  });

  it("is a stage that asks nothing once a stage past it has answers, an earlier line still open (W1-R B1)", () => {
    // The engine's log: Your question's intended use is open, and Who's in and Models hold answers.
    expect(stageOf(LOG, "question")!.lines.find((l) => l.key === "set_intended_use")!.status).toBe("open");
    expect(stageOf(LOG, "whos_in")!.progress!.answered).toBeGreaterThan(0);
    const first = stageOf(LOG, "first_look")!;
    expect(first).toMatchObject({ reached: true, progress: { required: 0, complete: true } });
    expect(readsDone(LOG, first)).toBe(true);
    const seg = segments(LOG).find((x) => x.key === "first_look")!;
    expect(seg.done).toBe(true);
    expect(seg.complete).toBe(true);
    expect(filled(seg)).toBe(1);
    // Only Who's in answered past it is enough.
    const whosOnly = untouched(LOG, ["models"]);
    expect(readsDone(whosOnly, stageOf(whosOnly, "first_look")!)).toBe(true);
  });

  it("is not Write-up while Results' lines are not all answered, under Predict (W1-R B1)", () => {
    expect(stageOf(LOG, "results")!.progress).toMatchObject({ answered: 0, complete: false });
    expect(readsDone(LOG, stageOf(LOG, "writeup")!)).toBe(false);
    const before = doneIn(LOG, ["question", "whos_in", "models"]);
    expect(readsDone(before, stageOf(before, "writeup")!)).toBe(false);
    const one = withStage(before, "results", (l) => (l.key === "set_updating" ? answered(l) : l), {
      progress: { answered: 1, required: 2, complete: false },
    }).log;
    expect(readsDone(one, stageOf(one, "writeup")!)).toBe(false);
  });

  it("is a stage whose objectives are all answered, and one that asks nothing once the analysis comes to it", () => {
    const data = stageOf(LOG, "data")!;
    expect(data.progress).toMatchObject({ complete: true });
    expect(data.progress!.required).toBeGreaterThan(0);
    expect(readsDone(LOG, data)).toBe(true);
    const done = doneIn(LOG, ["question", "whos_in", "models", "results"]);
    expect(readsDone(done, stageOf(done, "first_look")!)).toBe(true);
    expect(readsDone(done, stageOf(done, "writeup")!)).toBe(true);
    const seg = segments(done).find((x) => x.key === "writeup")!;
    expect(seg.done).toBe(true);
    expect(seg.complete).toBe(true);
    expect(filled(seg)).toBe(1);
  });

  it("is never a stage not reached", () => {
    const results = stageOf(NOT_REACHED, "results")!;
    expect(results.reached).toBe(false);
    expect(readsDone(NOT_REACHED, { ...results, progress: { answered: 0, required: 0, complete: true } })).toBe(false);
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
