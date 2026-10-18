/**
 * The mock's quest log (dev:mock only) places each line where the engine's registry does
 * (src/quest/fixtures/quest-contract.json), and keeps the walk's gates: under Predict, Results is
 * not reached and the opening waits until Fit is pressed, and the families wait for the
 * predictor-selection question. Read on the real server's captured NHANES walk under prediction.
 */
import type { DecisionRecord, Finding, ProjectView, QuestLine } from "../api/schema";
import ROLES from "../quest/cards/roles/fixture.json";
import { CONTRACT } from "../quest/fixtures/contract";
import prediction from "./fixtures/m3-nhanes-prediction.json";
import { expandView, viewsOf, type M3Fixture } from "./m3";
import { mockQuest, mockRoleLines, mockSections, openingRefusal, servedFit } from "./quest";

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

  it("lists each role the capture's roles answer recorded below high as the engine's own line, open until it is confirmed", () => {
    // The capture's roles answer recorded 16 columns unconfirmed (the walkthrough's 16), which its
    // driver then confirmed in one block; the engine's lines for bp_sys (waiting) and waist
    // (confirmed) are roles/fixture.json's (core/quest.py `_role_lines`). Its place in the stage
    // follows the table's order, so `order` is the capture's own.
    const waitingAt = VIEWS.findIndex((v) => (v.state.roles_unconfirmed ?? []).includes("bp_sys") && !v.state.role_confirmations?.bp_sys);
    expect(waitingAt).toBeGreaterThan(-1);
    const view = VIEWS[waitingAt]!;
    const engine = (column: string) => (ROLES.lines as unknown as QuestLine[]).find((l) => l.items[0]!.column === column)!;
    const { order: _engineOrder, ...bp } = engine("bp_sys");
    const reasons = { bp_sys: bp.items[0]!.evidence };
    const data = (v: ProjectView) => mockQuest(v, null, false, reasons).stages.find((s) => s.key === "data")!;
    const before = data(view);
    const roleLines = before.lines.filter((l) => l.id === ROLES.place.id);
    const unconfirmed = view.state.roles_unconfirmed!;
    expect(roleLines.map((l) => l.items[0]!.column)).toEqual(Object.keys(view.state.roles!).filter((c) => unconfirmed.includes(c)));
    expect(roleLines).toHaveLength(16);
    expect(roleLines.every((l) => l.status === "open" && l.counted)).toBe(true);
    const { order, ...mockBp } = roleLines.find((l) => l.key === "confirm_role:bp_sys")!;
    expect(mockBp).toEqual(bp);
    // Why a line waits names its own presses, so a flag's (the capture's six imputed_* columns)
    // differs from a covariate's: it is the engine's for a flag.
    const flag = roleLines.find((l) => l.key === "confirm_role:imputed_waist")!;
    expect(flag.reason).toBe(engine("imputed_waist").reason);
    expect(flag.items[0]!.options).toEqual(engine("imputed_waist").items[0]!.options);
    expect(flag.reason).not.toBe(bp.reason!.replaceAll("bp_sys", "imputed_waist"));
    expect(order).toBeGreaterThanOrEqual(ROLES.place.order);
    expect(before.progress?.complete).toBe(false);
    expect(before.progress!.required - before.progress!.answered).toBeGreaterThanOrEqual(16);

    const settledAt = VIEWS.findIndex((v, i) => i > waitingAt && !!v.state.role_confirmations?.waist);
    const after = data(VIEWS[settledAt]!).lines.filter((l) => l.id === ROLES.place.id);
    const { decision_id: _engineRecord, order: _o, items: [engineItem], ...waist } = engine("waist");
    const { decision_id: recordId, order: _m, items: [mockItem], ...mockWaist } = after.find((l) => l.key === "confirm_role:waist")!;
    expect(mockWaist).toEqual(waist);
    expect({ ...mockItem!, evidence: "" }).toEqual({ ...engineItem!, evidence: "" });
    expect(mockItem!.evidence).toBe(ROLES.unread_evidence);
    expect(VIEWS[settledAt]!.decisions.find((r) => r.id === recordId)?.decision.kind).toBe("confirm_readings");
    expect(after.every((l) => l.status === "answered")).toBe(true);
  });

  it("leaves an open noticing out of its stage's count under Predict, and counts it under Estimate (ruling R-a)", () => {
    // One constructed open noticing with a repair, so it is a Decide of Your data.
    const noticing = { id: "test::r-a", summary: "A noticing.", title: "A noticing.", routes_to: null, repairs: [{}] } as unknown as Finding;
    const data = (view: ProjectView, findings: Finding[]) => mockQuest(view, findings, false).stages.find((s) => s.key === "data")!;
    const estimate = { ...AT_OPENING, state: { ...AT_OPENING.state, purpose: "inference" as const } };
    for (const view of [AT_OPENING, estimate]) {
      const without = data(view, []).progress!;
      const withIt = data(view, [noticing]);
      expect(withIt.lines.find((l) => l.key === noticing.id)).toMatchObject({ label: "Decide", status: "open", counted: true });
      const predict = view.state.purpose === "prediction";
      expect(withIt.progress!.required, view.state.purpose ?? "none").toBe(without.required + (predict ? 0 : 1));
      expect(withIt.progress!.complete).toBe(predict ? without.complete : false);
    }
  });

  it("lists no role line where the roles answer left nothing unconfirmed", () => {
    const view = { ...AT_OPENING, state: { ...AT_OPENING.state, roles_unconfirmed: [] } };
    expect(mockRoleLines(view)).toEqual([]);
  });

  it("groups the methods' sentences by the stage that records each kind, every recorded one once", () => {
    const sections = mockSections(AT_OPENING);
    const recorded = AT_OPENING.decisions.filter((r) => r.sentence).length;
    expect(sections.count).toBe(recorded);
    expect(sections.guideline).toBe("TRIPOD+AI");
  });
});

/**
 * Fit's gate under Predict, as core/fit_press.py and core/seal.py keep it (P1): the fit is served
 * withheld until Fit is pressed, and the opening waits for the press, then for the triage of every
 * open noticing, in seal.py's order. Under Estimate the mock keeps its walk (W3). The noticing is
 * one constructed open finding, under an id of its own so no captured disposition settles it.
 */
describe("the mock's Fit gate under Predict", () => {
  const NOTICED = {
    id: "test::noticed",
    severity: "info",
    title: "9 columns hold SAS export zeros.",
    summary: "152 cells in bp_di hold 5.4e-79.",
    detail: "",
    why_it_matters: null,
    affected_columns: ["bp_di"],
    source: "profile",
    lens: null,
    evidence: null,
    routes_to: null,
    lever_label: null,
    group: null,
    repairs: [],
    disposition: null,
    answered_by: null,
  } as unknown as Finding;
  const opening = (view: ProjectView, pressed: boolean) => openingRefusal(view, mockQuest(view, [NOTICED], pressed), pressed);
  const TRIAGED = { kind: "confirm_sweep", stage: "results", sweep: "noticings", lines: [] } as never;

  it("is where the capture's opening stands: under Predict, the fit computed, rows held out, not opened", () => {
    expect(AT_OPENING.state.purpose).toBe("prediction");
    expect(AT_OPENING.stages.fit?.status).toBe("fresh");
    expect(AT_OPENING.state.split?.holdout).toBeGreaterThan(0);
    expect(AT_OPENING.state.seal_opened).toBeNull();
  });

  it("refuses the opening until Fit is pressed, with the way to press it", () => {
    const r = opening(AT_OPENING, false);
    expect(r?.error.code).toBe("fit_not_yet");
    expect(r?.error.exits).toEqual([{ label: "Press Fit on the analysis flowchart", decision: null }]);
  });

  it("then refuses it while a noticing is open, offering its triage's Confirm all, and lets it through once that is recorded", () => {
    const r = opening(AT_OPENING, true);
    expect(r?.error.code).toBe("triage_open");
    expect(r?.error.message).toContain("1 is still open");
    expect(r?.error.message).toContain("152 cells in bp_di hold 5.4e-79 (doesn't change your numbers here)");
    expect(r?.error.exits.map((x) => x.decision)).toEqual([null, TRIAGED]);
    expect(opening(record(AT_OPENING, TRIAGED), true)).toBeNull();
  });

  it("leaves the refusals that come first in seal.py, and every other goal, to the mock's own validator", () => {
    const opened = { ...AT_OPENING, state: { ...AT_OPENING.state, seal_opened: { at: "x" } } } as unknown as ProjectView;
    expect(opening(opened, false)).toBeNull();
    const stale = { ...AT_OPENING, stages: { ...AT_OPENING.stages, fit: { ...AT_OPENING.stages.fit!, status: "running" as const } } };
    expect(opening(stale, false)).toBeNull();
    const estimate = { ...AT_OPENING, state: { ...AT_OPENING.state, purpose: "inference" as const } };
    expect(opening(estimate, false)).toBeNull();
  });

  it("serves the fit withheld until Fit is pressed: every score and coefficient removed, the reason first", () => {
    const fit = {
      withheld: null,
      withheld_for: null,
      comparison: { metric: "r2" },
      comparisons: [{ a: 1 }],
      models: [{ family: "linear", cv: { r2: { mean: 0.08 } }, holdout: { r2: 0.07 }, coefficients: [{ term: "kcal", estimate: 0.1 }], concerns: ["c"] }],
    };
    const withheld = servedFit(fit, "prediction", false) as unknown as typeof fit & { models: Record<string, unknown>[] };
    expect(withheld).toMatchObject({ withheld_for: "fit", comparison: null, comparisons: [] });
    expect(withheld.withheld).toMatch(/^No score or estimate is shown until Fit is pressed/);
    expect(withheld.models[0]).toMatchObject({ family: "linear", cv: {}, holdout: null, coefficients: null });
    expect(withheld.models[0]!.concerns).toEqual([withheld.withheld, "c"]);
    expect(servedFit(fit, "prediction", true)).toBe(fit);
    expect(servedFit(fit, "inference", false)).toBe(fit);
  });
});
