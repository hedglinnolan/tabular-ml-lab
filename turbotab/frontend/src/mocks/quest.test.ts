/**
 * The mock's quest log (dev:mock only) places each line where the engine's registry does
 * (src/quest/fixtures/quest-contract.json), and keeps the walk's gates: under Predict, Results is
 * not reached and the opening waits until Fit is pressed, and the families wait for the
 * predictor-selection question. Read on the real server's captured NHANES walk under prediction.
 */
import type { DecisionRecord, ProjectView } from "../api/schema";
import { CONTRACT } from "../quest/fixtures/contract";
import prediction from "./fixtures/m3-nhanes-prediction.json";
import { expandView, viewsOf, type M3Fixture } from "./m3";
import { mockQuest, mockSections } from "./quest";

const F = prediction as unknown as M3Fixture;
const VIEWS = viewsOf(F).map((v) => expandView(F, v, "mock-test", "2026-10-17T00:00:00Z"));
/** The captured view where the opening is the Router's open question. */
const AT_OPENING = VIEWS.find((v) => v.interview.some((s) => s.key === "open_seal" && s.status === "open"))!;

function record(view: ProjectView, decision: DecisionRecord["decision"]): ProjectView {
  const seq = view.decisions.length + 1;
  const rec = { ...view.decisions[0]!, id: `m${seq}`, seq, decision, sentence: null };
  return { ...view, decisions: [...view.decisions, rec] };
}

describe("the mock's quest log", () => {
  it("places every question the Router lists in the engine's stage, by the engine's name", () => {
    const log = mockQuest(AT_OPENING, null, false);
    for (const step of AT_OPENING.interview) {
      if (step.status === "not_applicable") continue;
      const place = CONTRACT.questions[step.key]!;
      const stage = log.stages.find((s) => s.lines.some((l) => l.source === "question" && l.key === step.key));
      expect(stage?.key, step.key).toBe(place.stage);
      expect(stage!.lines.find((l) => l.key === step.key)!.name).toBe(place.name);
    }
    expect(log.stages.map((s) => s.name)).toEqual(CONTRACT.stages.map((s) => s.name));
  });

  it("lists Predict's declarations where the engine places them", () => {
    const log = mockQuest(AT_OPENING, null, false);
    for (const kind of ["set_intended_use", "set_selection", "set_validation", "set_levers", "set_explain"]) {
      const stage = log.stages.find((s) => s.lines.some((l) => l.source === "declaration" && l.key === kind));
      expect(stage?.key, kind).toBe(CONTRACT.declarations[kind]!.stage);
    }
  });

  it("keeps Results unreached, and the opening waiting, until Fit is pressed", () => {
    const before = mockQuest(AT_OPENING, null, false);
    const results = (log: typeof before) => log.stages.find((s) => s.key === "results")!;
    expect(results(before).reached).toBe(false);
    expect(results(before).lines.find((l) => l.key === "open_seal")!.status).toBe("waiting");
    expect(before.fit).toMatchObject({ purpose: "prediction", pressed: false, reason: CONTRACT.fit_locks.ready.reason });
    const after = mockQuest(AT_OPENING, null, true);
    expect(results(after).reached).toBe(true);
    expect(results(after).lines.find((l) => l.key === "open_seal")!.status).toBe("open");
    expect(after.fit).toMatchObject({ pressed: true, reason: CONTRACT.fit_locks.pressed.reason });
  });

  it("holds the families for the predictor-selection question under Predict, and frees them once it is answered", () => {
    const open = VIEWS.find((v) => v.state.purpose === "prediction" && v.interview.some((s) => s.key === "models" && s.status === "open"))!;
    const models = (view: ProjectView) =>
      mockQuest(view, null, false).stages.find((s) => s.key === "models")!.lines.find((l) => l.key === "models")!;
    expect(models(open)).toMatchObject({ status: "waiting", waiting_for: [{ key: "set_selection", stage: "models" }] });
    expect(models(record(open, { kind: "set_selection", method: "none" } as never)).status).toBe("open");
  });

  it("answers a stage's Confirm lines with its Confirm all, and counts the sweep once", () => {
    const view = record(AT_OPENING, { kind: "confirm_sweep", stage: "models", sweep: "defaults", lines: [] } as never);
    const stage = mockQuest(view, null, false).stages.find((s) => s.key === "models")!;
    const confirms = stage.lines.filter((l) => l.label === "Confirm");
    expect(confirms.length).toBeGreaterThan(0);
    expect(confirms.every((l) => l.status === "answered")).toBe(true);
    expect(stage.sweep).toMatchObject({ answered: true, action: CONTRACT.sweep_words[String(confirms.length)]![1] });
    const counted = stage.lines.filter((l) => l.label === "Decide" && l.counted).length;
    expect(stage.progress!.required).toBe(counted + 1);
  });

  it("groups the methods' sentences by the stage that records each kind, every recorded one once", () => {
    const sections = mockSections(AT_OPENING);
    const recorded = AT_OPENING.decisions.filter((r) => r.sentence).length;
    expect(sections.count).toBe(recorded);
    expect(sections.guideline).toBe("TRIPOD+AI");
  });
});
