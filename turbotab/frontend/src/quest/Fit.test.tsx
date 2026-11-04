/**
 * Fit at the flowchart's end (ruling 3; FOUNDATION §7): held, it is Fit with how long the fit is
 * expected to take and no Cancel, since nothing runs; while the fit job runs, its progress and a
 * Cancel where Fit was. The lock is the engine's (fixtures/quest-contract.json: core/fit_press.py
 * `fit_lock` for a fit estimated at 1,800 s), so the estimate on the button is the engine's words.
 * A fit the engine refused in words (StageStatus.refusal, the audit's G04) is said beneath Fit with
 * its ways forward that name no decision in words (a decision is the card's), on the statuses the
 * engine serves (components/fixtures/stages.json). A press of Fit the engine refused is said so too,
 * once where it is the fit's own refusal, and while what it answered stands: no decision recorded
 * since, and the fit's refusal unchanged.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import { api, RefusalError } from "../api/client";
import { keys } from "../api/queries";
import type { DecisionRecord, FitLock, ProjectView, QuestLog, Refusal, StageStatus } from "../api/schema";
import { SERVED } from "../components/fixtures/stages";
import { plain } from "../kit";
import { expandView, viewsOf, type M3Fixture } from "../mocks/m3";
import { FitControl } from "./Fit";
import { CONTRACT } from "./fixtures/contract";
import { fitPosition, heldFor } from "./log";

const BASE = CONTRACT.logs.results_reached.log;
const { held, pressed, estimate_seconds } = CONTRACT.fit_locks;
const PID = "p";
const FIXTURES = import.meta.glob<M3Fixture>("../mocks/fixtures/m3-nhanes-inference.json", { eager: true, import: "default" });
const JOURNEY = Object.values(FIXTURES)[0]!;
/** The project as the server serves it at Fit: the NHANES journey under Estimate, its models
 *  chosen, its plan not locked yet (the last view the capture holds before its `lock_plan`). */
const VIEW = ((): ProjectView => {
  const before = viewsOf(JOURNEY).filter((v) => v.decisions.every((id) => JOURNEY.log[id]?.decision.kind !== "lock_plan"));
  return expandView(JOURNEY, before[before.length - 1]!, PID, "2026-10-17T00:00:00Z");
})();

/** The sample log with Models complete and the given lock. */
function logWith(lock: FitLock): QuestLog {
  return {
    ...BASE,
    fit: lock,
    stages: BASE.stages.map((s) =>
      s.key === "models" ? { ...s, progress: { answered: 5, required: 5, complete: true } } : s,
    ),
  };
}

function fitStatus(status: StageStatus["status"], progress: number | null): StageStatus {
  return {
    stage: "fit",
    status,
    key: "k",
    fresh: status === "fresh",
    missing: [],
    error: null,
    job_id: null,
    progress,
    updated_at: null,
    cancelled: false,
    held: status === "idle" ? "fit" : null,
    refusal: null,
  };
}

/** A query cache as the app's (nothing refetches unasked), holding the project's view. */
function client(): QueryClient {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  qc.setQueryData(keys.view(PID), VIEW);
  return qc;
}

function wrap(node: ReactNode, qc = client()) {
  return render(<QueryClientProvider client={qc}>{node}</QueryClientProvider>);
}

/** Fit pressed with the plan not locked yet (the engine's `unlocked` lock), over the fit's status
 *  `fit`, and refused with `refusal`, settled: what is said beneath Fit then (`said`), and Fit drawn
 *  again over a new status of the fit (`redraw`). */
async function pressedOver(fit: StageStatus, refusal: Refusal) {
  vi.spyOn(api, "pressFit").mockRejectedValue(new RefusalError(refusal));
  const qc = client();
  const log = logWith(CONTRACT.fit_locks.unlocked);
  const view = wrap(<FitControl pid={PID} log={log} fit={fit} />, qc);
  await userEvent.setup().click(screen.getByTestId("fit"));
  await waitFor(() => expect(qc.getMutationCache().getAll().map((m) => m.state.status)).toEqual(["error"]));
  const said = () => within(screen.getByTestId("fit-position")).queryAllByRole("alert").map((a) => a.textContent);
  const redraw = (next: StageStatus) =>
    view.rerender(
      <QueryClientProvider client={qc}>
        <FitControl pid={PID} log={log} fit={next} />
      </QueryClientProvider>,
    );
  return { qc, said, redraw };
}

describe("Fit, held", () => {
  it("is Fit with the engine's estimate, and no Cancel", () => {
    // 1,800 seconds is 30 minutes (core/models/cost.py `duration`: rounded to the minute under an hour).
    const minutes = estimate_seconds / 60;
    expect(heldFor(held)).toBe(`about ${minutes} minutes`);
    wrap(<FitControl pid="p" log={logWith(held)} fit={fitStatus("idle", null)} />);
    expect(screen.getByTestId("fit")).toHaveTextContent(`Fit · about ${minutes} minutes`);
    expect(screen.queryByTestId("fit-cancel")).toBeNull();
    expect(screen.queryByRole("progressbar")).toBeNull();
  });

  it("is Fit alone when the fit is not held, and nothing before Models is done", () => {
    const notHeld = { ...held, held: false };
    expect(fitPosition(logWith(notHeld), fitStatus("fresh", 1))).toEqual({ kind: "press", held: false, estimate: null });
    wrap(<FitControl pid="p" log={logWith(notHeld)} fit={fitStatus("fresh", 1)} />);
    expect(screen.getByTestId("fit")).toHaveTextContent(/^Fit$/);
    // The sample log's Models is not complete: no Fit yet.
    expect(fitPosition({ ...BASE, fit: held }, fitStatus("idle", null))).toEqual({ kind: "none" });
  });
});

describe("Fit, running", () => {
  it("shows the job's progress and a Cancel where Fit was", () => {
    wrap(<FitControl pid="p" log={logWith(pressed)} fit={fitStatus("running", 0.4)} />);
    expect(screen.queryByTestId("fit")).toBeNull();
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "40");
    expect(screen.getByTestId("fit-cancel")).toHaveTextContent("Cancel");
  });

  it("is gone once the pressed fit is computed: Results opens", () => {
    expect(fitPosition(logWith(pressed), fitStatus("fresh", 1))).toEqual({ kind: "pressed" });
    const { container } = wrap(<FitControl pid="p" log={logWith(pressed)} fit={fitStatus("fresh", 1)} />);
    expect(container).toBeEmptyDOMElement();
  });
});

describe("Fit, over a fit the engine refused", () => {
  const notHeld = { ...held, held: false };
  /** The fixture's status as the fit serves it; a refusal of its own is the fit's. */
  const asFit = (from: StageStatus): StageStatus => ({
    ...from,
    stage: "fit",
    refusal: from.refusal && from.refusal.stage === from.stage ? { ...from.refusal, stage: "fit" } : from.refusal,
  });
  afterEach(() => vi.restoreAllMocks());

  it("says the refusal beneath Fit with its way forward that names no decision, none a press", () => {
    // Fit sits in the tapestry, which records nothing (FOUNDATION §1; Stage's recordHere): the
    // way forward that is a decision is the card's to offer, so it is neither a press nor a
    // sentence here; the one that names no decision is said in words.
    const fit = asFit(SERVED.R);
    const refusal = fit.refusal!;
    const decide = vi.spyOn(api, "decide");
    wrap(<FitControl pid="p" log={logWith(notHeld)} fit={fit} />);
    expect(screen.getByTestId("fit")).toBeInTheDocument();
    const note = screen.getByTestId("fit-refusal");
    expect(note).toHaveAttribute("role", "alert");
    expect(note).toHaveAttribute("data-stage", "fit");
    expect(note).toHaveAttribute("data-code", refusal.code);
    const [take, told] = refusal.exits;
    expect([take!.decision === null, told!.decision === null]).toEqual([false, true]);
    expect(note.textContent).toBe(`${refusal.message} ${String(told!.label)}.`);
    for (const x of refusal.exits) expect(screen.queryByRole("button", { name: String(x.label) })).toBeNull();
    expect(screen.getAllByRole("button").map((b) => b.dataset.testid)).toEqual(["fit"]);
    expect(decide).not.toHaveBeenCalled();
  });

  it("says no confirmation of an unsettled reading as a sentence of fact", () => {
    // U's ways forward are the two answers to one reading ("`age` is an amount", "`age` holds
    // codes"), the card's to ask: said one after the other they would state both as facts.
    const fit = asFit(SERVED.U);
    const refusal = fit.refusal!;
    wrap(<FitControl pid="p" log={logWith(notHeld)} fit={fit} />);
    const note = screen.getByTestId("fit-refusal");
    expect(note).toHaveAttribute("data-code", refusal.code);
    expect(refusal.exits.every((x) => x.decision !== null)).toBe(true);
    expect(note.textContent).toBe(plain(refusal.message));
    for (const x of refusal.exits) expect(note).not.toHaveTextContent(plain(String(x.label)));
    expect(screen.getAllByRole("button").map((b) => b.dataset.testid)).toEqual(["fit"]);
  });

  it("says the refusal of the stage the fit waits on", () => {
    // AR waits on RH, which refused: as the fit waits on the design.
    const fit = { ...SERVED.AR, stage: "fit" };
    wrap(<FitControl pid="p" log={logWith(notHeld)} fit={fit} />);
    const note = screen.getByTestId("fit-refusal");
    expect(note).toHaveAttribute("data-stage", SERVED.AR.refusal!.stage);
    expect(note).toHaveTextContent(SERVED.AR.refusal!.message);
  });

  it("says nothing of a refusal over a fit that crashed, or one that is computed", () => {
    wrap(<FitControl pid="p" log={logWith(notHeld)} fit={asFit(SERVED.C)} />);
    expect(screen.queryByTestId("fit-refusal")).toBeNull();
    wrap(<FitControl pid="p" log={logWith(notHeld)} fit={fitStatus("fresh", 1)} />);
    expect(screen.queryByTestId("fit-refusal")).toBeNull();
  });
});

describe("Fit, a press the engine refused", () => {
  // Under Estimate and Describe the engine refuses a press of Fit while the open noticings' triage
  // before the lock is not confirmed (core/fit_press.py `triage_refusal`, code `triage_open`), in
  // the served shape (turbotab/server/tests/test_fit_press_holds_api.py): its words, built by the
  // engine's rule from a noticing the triage serves (results/fixtures/nhanes-predict.json,
  // `binary_text__gender`), and two ways forward, the triage in Models (no decision) and its
  // Confirm all (a decision). The shell draws that triage at Models' end; Fit sits in the tapestry,
  // which records nothing (FOUNDATION §1), so the press's refusal is said as Fit's other lines are:
  // its words and the ways forward that name no decision, in sentences, none a press.
  const { unlocked } = CONTRACT.fit_locks;
  const LEAD = "Fit waits: pressing it locks the analysis plan, and the open noticings are triaged before the plan is locked.";
  const SUMMARY = "`gender` is two-level text (`female`, `male`); which level counts as 1 is not asked yet.";
  const said = (summary: string) => summary.trim().replace(/\.$/, "");
  const TRIAGE_IN_MODELS = "Triage the open noticings in Models";
  const KEEP = "Keep the recommended dispositions";
  const OPEN: Refusal = {
    error: {
      code: "triage_open",
      message: `${LEAD} 1 noticing is still open: ${said(SUMMARY)}. Recommended: act on it. Triage it in Models first, so it is decided before any estimate is seen.`,
      exits: [
        { label: TRIAGE_IN_MODELS, decision: null },
        { label: KEEP, decision: { kind: "confirm_sweep", stage: "models", sweep: "noticings", lines: [] } },
      ],
    },
  };
  const BLOCKING_SUMMARY = "`glucose` takes one value in every row.";
  const BLOCKING: Refusal = {
    error: {
      code: "triage_open",
      message: `${LEAD} 1 noticing must be resolved first: ${said(BLOCKING_SUMMARY)}.`,
      exits: [{ label: `Decide: ${BLOCKING_SUMMARY}`, decision: null }],
    },
  };
  afterEach(() => vi.restoreAllMocks());

  async function pressRefused(refusal: Refusal | Error) {
    const press = vi.spyOn(api, "pressFit").mockRejectedValue(refusal instanceof Error ? refusal : new RefusalError(refusal));
    const decide = vi.spyOn(api, "decide");
    wrap(<FitControl pid="p" log={logWith(unlocked)} fit={fitStatus("fresh", 1)} />);
    await userEvent.setup().click(screen.getByTestId("fit"));
    const alert = await within(screen.getByTestId("fit-position")).findByRole("alert");
    expect(press).toHaveBeenCalledTimes(1);
    return { alert, decide };
  }

  it("says the triage's refusal with the way forward that names where the triage is, and no press", async () => {
    expect(unlocked.purpose).not.toBe("prediction");
    const { alert, decide } = await pressRefused(OPEN);
    expect(alert.textContent).toBe(`${plain(OPEN.error.message)} ${TRIAGE_IN_MODELS}.`);
    expect(alert).not.toHaveTextContent(KEEP);
    expect(screen.getAllByRole("button").map((b) => b.dataset.testid)).toEqual(["fit"]);
    expect(decide).not.toHaveBeenCalled();
  });

  it("says a blocking noticing's refusal with its way forward in words", async () => {
    const { alert, decide } = await pressRefused(BLOCKING);
    expect(alert.textContent).toBe(`${plain(BLOCKING.error.message)} ${plain(BLOCKING.error.exits[0]!.label)}`);
    expect(screen.getAllByRole("button").map((b) => b.dataset.testid)).toEqual(["fit"]);
    expect(decide).not.toHaveBeenCalled();
  });

  it("says a refusal with no way forward, and a failure, in their words alone, as before", async () => {
    const ordinary: Refusal = { error: { code: "fit_not_yet", message: "There is nothing to fit yet.", exits: [] } };
    const first = await pressRefused(ordinary);
    expect(first.alert.textContent).toBe(ordinary.error.message);
    cleanup();
    vi.restoreAllMocks();
    const second = await pressRefused(new Error("The server stopped."));
    expect(second.alert.textContent).toBe("The server stopped.");
    expect(screen.getAllByRole("button").map((b) => b.dataset.testid)).toEqual(["fit"]);
  });

  const OPEN_SAID = `${plain(OPEN.error.message)} ${TRIAGE_IN_MODELS}.`;

  it("says the triage's refusal no longer once the triage is confirmed, though the quest log reads the same", async () => {
    // The person takes the way past where the triage is drawn, at Models' end on this same screen:
    // its Confirm all records the decision the refusal's own way forward names, each noticing as the
    // engine recommends (core/sweep.py `_confirm_the_triage`), worded by `confirm_sentence`. A
    // noticing recorded to act on stays open in the quest log for its own decision (core/sweep.py
    // `cover_noticings`; core/tests/test_confirm_sweep.py, `note__SEQN`), so the log can read the
    // same; the project's decisions do not.
    const p = await pressedOver(fitStatus("fresh", 1), OPEN);
    expect(p.said()).toEqual([OPEN_SAID]);
    const last = VIEW.decisions[VIEW.decisions.length - 1]!;
    const confirmed: DecisionRecord = {
      id: "triage-confirmed",
      seq: last.seq + 1,
      at: "2026-10-17T00:00:01Z",
      note: null,
      sentence: "The open noticings were triaged in Models: 1 to act on.",
      post_seal: false,
      after_estimates: false,
      recorded_by: "you",
      early: null,
      decision: {
        kind: "confirm_sweep",
        stage: "models",
        sweep: "noticings",
        lines: [{ id: "finding:binary_text__gender", key: "binary_text__gender", value: "act_on_it", basis: null }],
      },
    };
    expect(OPEN.error.exits[1]!.decision).toMatchObject({ kind: "confirm_sweep", stage: "models", sweep: "noticings" });
    act(() => {
      p.qc.setQueryData<ProjectView>(keys.view(PID), (v) => v && { ...v, decisions: [...v.decisions, confirmed] });
    });
    await waitFor(() => expect(p.said()).toEqual([]));
    expect(screen.getByTestId("fit")).toBeEnabled();
  });

  it("keeps saying it while nothing it answered moved: the view read again the same, the fit computing beside it", async () => {
    // Under Estimate a fit not held may compute before the press, its progress mirrored onto its
    // status (api/events.ts `applyJob`); a re-read of the view that brings the same decisions keeps
    // them as they were (the query cache's structural sharing).
    const p = await pressedOver(fitStatus("running", 0.4), OPEN);
    expect(p.said()).toEqual([OPEN_SAID]);
    await act(async () => {
      p.qc.setQueryData<ProjectView>(keys.view(PID), structuredClone(p.qc.getQueryData<ProjectView>(keys.view(PID))!));
      await new Promise((settled) => setTimeout(settled, 20)); // the cache tells its observers on a later tick
    });
    p.redraw(fitStatus("running", 0.5));
    expect(p.said()).toEqual([OPEN_SAID]);
  });

  describe("over a fit the engine refused", () => {
    // Under Estimate a press is refused, while the fit or a stage it waits on is refused, with that
    // stage's own refusal as the fit's status serves it, its code, words and ways forward
    // (server/service.py `press_fit`; turbotab/server/tests/test_fit_press_holds_api.py), which Fit
    // says beneath it already.
    const fit: StageStatus = { ...SERVED.R, stage: "fit", refusal: { ...SERVED.R.refusal!, stage: "fit" } };
    const own = fit.refusal!;
    const told = own.exits.find((x) => x.decision === null)!;
    const FIT_SAID = `${own.message} ${String(told.label)}.`;
    // The status types its ways forward loosely; the press serves the same ones.
    const AS_PRESSED: Refusal = { error: { code: own.code, message: own.message, exits: own.exits as Refusal["error"]["exits"] } };

    it("says the fit's own refusal once", async () => {
      const p = await pressedOver(fit, AS_PRESSED);
      expect(p.said()).toEqual([FIT_SAID]);
      expect(screen.getByTestId("fit-refusal").textContent).toBe(FIT_SAID);
    });

    it("says a press refused otherwise beside it", async () => {
      // The fit's status as last read, and a press the server refused on a newer state: the triage's.
      const p = await pressedOver(fit, BLOCKING);
      expect(p.said()).toEqual([FIT_SAID, `${plain(BLOCKING.error.message)} ${plain(BLOCKING.error.exits[0]!.label)}`]);
    });

    it("says neither once the fit is refused no more, a stage run again with no decision recorded", async () => {
      const p = await pressedOver(fit, AS_PRESSED);
      p.redraw(fitStatus("fresh", 1));
      expect(p.said()).toEqual([]);
    });
  });
});
