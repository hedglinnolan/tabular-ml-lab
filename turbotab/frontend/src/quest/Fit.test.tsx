/**
 * Fit at the flowchart's end (ruling 3; FOUNDATION §7): held, it is Fit with how long the fit is
 * expected to take and no Cancel, since nothing runs; while the fit job runs, its progress and a
 * Cancel where Fit was. The lock is the engine's (fixtures/quest-contract.json: core/fit_press.py
 * `fit_lock` for a fit estimated at 1,800 s), so the estimate on the button is the engine's words.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import type { ReactNode } from "react";
import type { FitLock, QuestLog, StageStatus } from "../api/schema";
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
