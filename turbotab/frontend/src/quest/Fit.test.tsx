/**
 * Fit at the flowchart's end (ruling 3; FOUNDATION §7): held, it is Fit with how long the fit is
 * expected to take and no Cancel, since nothing runs; while the fit job runs, its progress and a
 * Cancel where Fit was. The lock is the engine's (fixtures/quest-contract.json: core/fit_press.py
 * `fit_lock` for a fit estimated at 1,800 s), so the estimate on the button is the engine's words.
 * A fit the engine refused in words (StageStatus.refusal, the audit's G04) is said beneath Fit with
 * its ways forward that name no decision in words (a decision is the card's), on the statuses the
 * engine serves (components/fixtures/stages.json).
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import type { ReactNode } from "react";
import { api } from "../api/client";
import type { FitLock, QuestLog, StageStatus } from "../api/schema";
import { SERVED } from "../components/fixtures/stages";
import { plain } from "../kit";
import { FitControl } from "./Fit";
import { CONTRACT } from "./fixtures/contract";
import { fitPosition, heldFor } from "./log";

const BASE = CONTRACT.logs.results_reached.log;
const { held, pressed, estimate_seconds } = CONTRACT.fit_locks;

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

function wrap(node: ReactNode) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={qc}>{node}</QueryClientProvider>);
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
