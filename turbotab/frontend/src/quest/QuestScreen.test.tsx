/**
 * Fit's press, followed through the shell (FOUNDATION §7; ruling 3). A short fit computed before
 * the press opens Results at once. A fit that must run keeps the screen on Models' analysis
 * flowchart, where the Fit position shows the job's progress and a Cancel where Fit was, and
 * Results opens once the fit is computed. A screen opened while the pressed fit runs lands there.
 *
 * The view is a real server's captured NHANES walk under Predict (src/mocks/fixtures), and the
 * quest log and Fit's locks are the engine's (fixtures/quest-contract.json), with Models complete
 * and the fit stage's status set for each step, as the client sees them after the press: the lock
 * first (usePressFit keeps the returned lock), then the fit job's status from the view.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, vi } from "vitest";
import { keys } from "../api/queries";
import type { FitLock, ProjectView, QuestLog, StageStatus } from "../api/schema";
import { expandView, viewsOf, type M3Fixture } from "../mocks/m3";
import { CONTRACT } from "./fixtures/contract";
import { QuestScreen } from "./QuestScreen";

const FIXTURES = import.meta.glob<M3Fixture>("../mocks/fixtures/m3-nhanes-prediction.json", { eager: true, import: "default" });
const JOURNEY = Object.values(FIXTURES)[0]!;
const PID = "quest-fit";
const VIEW = ((): ProjectView => {
  const views = viewsOf(JOURNEY);
  return expandView(JOURNEY, views[views.length - 1]!, PID, "2026-10-17T00:00:00Z");
})();
const BASE = CONTRACT.logs.results_reached.log;
const { held, pressed, ready, estimate_seconds } = CONTRACT.fit_locks;

/** The engine's sample log with Models complete and the given lock. */
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
  };
}

/** The query cache tells its observers on a timer. */
const notified = () => new Promise((r) => setTimeout(r, 0));

function shell(lock: FitLock, fit: StageStatus) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  const serve = (l: FitLock, f: StageStatus) => {
    qc.setQueryData(keys.quest(PID), logWith(l));
    qc.setQueryData(keys.view(PID), { ...VIEW, stages: { ...VIEW.stages, fit: f } });
  };
  serve(lock, fit);
  render(
    <QueryClientProvider client={qc}>
      <QuestScreen pid={PID} stream="open" />
    </QueryClientProvider>,
  );
  return {
    /** The lock alone, as usePressFit keeps it before the view is read again. */
    lock: (l: FitLock) =>
      act(async () => {
        qc.setQueryData<QuestLog>(keys.quest(PID), (prev) => ({ ...prev!, fit: l }));
        await notified();
      }),
    serve: (l: FitLock, f: StageStatus) =>
      act(async () => {
        serve(l, f);
        await notified();
      }),
  };
}

/** The stage the card shows (StageCard's `card-<stage>`). */
const STAGE_CARD = new RegExp(`^card-(${CONTRACT.stages.map((s) => s.key).join("|")})$`);
const card = () => screen.getByTestId(STAGE_CARD).dataset.testid;

// Whatever is not seeded stays pending: nothing here reaches a server.
beforeEach(() => {
  vi.stubGlobal("fetch", vi.fn(() => new Promise(() => {})));
});
afterEach(() => {
  vi.unstubAllGlobals();
});

describe("pressing a held Fit", () => {
  it("keeps the Fit position, with its progress and Cancel, while the fit runs; Results opens once it is computed", async () => {
    const s = shell(held, fitStatus("idle", null));
    fireEvent.click(screen.getByTestId("stage-models"));
    expect(card()).toBe("card-models");
    expect(screen.getByTestId("fit")).toHaveTextContent(`Fit · about ${estimate_seconds / 60} minutes`);
    expect(screen.queryByTestId("fit-cancel")).toBeNull();

    // The press: the returned lock first, the view's fit job not read yet.
    await s.lock(pressed);
    expect(card()).toBe("card-models");
    expect(screen.queryByTestId("tapestry-results")).toBeNull();

    // The fit job runs: progress and a Cancel where Fit was, on Models' flowchart.
    await s.serve(pressed, fitStatus("running", 0.4));
    expect(card()).toBe("card-models");
    const position = screen.getByTestId("fit-position");
    expect(within(position).getByRole("progressbar")).toHaveAttribute("aria-valuenow", "40");
    expect(within(position).getByTestId("fit-cancel")).toHaveTextContent("Cancel");
    expect(screen.getByTestId("tapestry")).toHaveAccessibleName("Your analysis");
    expect(screen.queryByTestId("tapestry-results")).toBeNull();

    // Computed: Results opens.
    await s.serve(pressed, fitStatus("fresh", 1));
    expect(card()).toBe("card-results");
    expect(screen.getByTestId("tapestry-results")).toBeInTheDocument();
    expect(screen.queryByTestId("fit-cancel")).toBeNull();
  });

  it("leaves the screen where the person went while the fit ran", async () => {
    const s = shell(held, fitStatus("idle", null));
    fireEvent.click(screen.getByTestId("stage-models"));
    await s.lock(pressed);
    await s.serve(pressed, fitStatus("running", 0.4));
    fireEvent.click(screen.getByTestId("stage-whos_in"));
    await s.serve(pressed, fitStatus("fresh", 1));
    expect(card()).toBe("card-whos_in");
  });
});

describe("pressing Fit on a short fit already computed", () => {
  it("opens Results at once", async () => {
    const s = shell(ready, fitStatus("fresh", 1));
    fireEvent.click(screen.getByTestId("stage-models"));
    expect(screen.getByTestId("fit")).toHaveTextContent(/^Fit$/);
    await s.lock(pressed);
    expect(card()).toBe("card-results");
    expect(screen.getByTestId("tapestry-results")).toBeInTheDocument();
  });
});

describe("a screen opened while the pressed fit runs", () => {
  it("lands on Models' flowchart, with the fit's progress and Cancel", () => {
    shell(pressed, fitStatus("running", 0.4));
    expect(card()).toBe("card-models");
    const position = screen.getByTestId("fit-position");
    expect(within(position).getByRole("progressbar")).toHaveAttribute("aria-valuenow", "40");
    expect(within(position).getByTestId("fit-cancel")).toBeInTheDocument();
  });

  it("lands on the open stage once the fit is computed", () => {
    shell(pressed, fitStatus("fresh", 1));
    const open = BASE.stages.find((st) => st.reached && !st.progress?.complete)!.key;
    expect(card()).toBe(`card-${open}`);
    expect(screen.queryByTestId("fit-position")).toBeNull();
  });
});
