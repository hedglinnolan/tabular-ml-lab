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
 *
 * Then a card's recorded answer, followed through the shell: the tapestry returns to live, as it
 * does under the Record, and keeps its preview while nothing is recorded.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, vi } from "vitest";
import { api, RefusalError } from "../api/client";
import type { TeachingEntry } from "../api/m1-types";
import type { PreviewResult } from "../api/m1-stage-types";
import { keys } from "../api/queries";
import type { FitLock, ProjectView, QuestLog, Refusal, StageStatus } from "../api/schema";
import { artifactAt, expandView, viewsOf, type M3Fixture } from "../mocks/m3";
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

/**
 * A recorded answer returns the tapestry to the live data, as the Record's recording does
 * (record/Record.tsx resets the stage focus on success): the option a card previewed is not shown
 * as "nothing is recorded" beside an answer that is. The view is the capture's first, the lens
 * question open, and the next captured view is what the server returns for the recording; the
 * quest log is the engine's, with the lens line open in Your data and no later stage reached.
 */
describe("a card's recorded answer in the shell", () => {
  const views = viewsOf(JOURNEY);
  const OPEN = expandView(JOURNEY, views[0]!, PID, "2026-10-17T00:00:00Z");
  const RECORDED = expandView(JOURNEY, views[1]!, PID, "2026-10-17T00:00:01Z");
  const SHARED = Object.values(
    import.meta.glob<{ teaching: TeachingEntry[] }>("../mocks/fixtures/m3-shared.json", { eager: true, import: "default" }),
  )[0]!;
  const MODELS_OPEN = CONTRACT.logs.models_open.log;
  const LENS_OPEN: QuestLog = {
    ...MODELS_OPEN,
    stages: MODELS_OPEN.stages.map((s) =>
      s.key === "data"
        ? {
            ...s,
            progress: { answered: 1, required: 2, complete: false },
            lines: s.lines.map((l) => (l.key === "lens" ? { ...l, status: "open", decision_id: null } : l)),
          }
        : { ...s, reached: false, progress: null, sweep: null },
    ),
  };
  const PREVIEW: PreviewResult = { kind: "none", views: [], basis: "No views.", note: null, caution: null };

  function lensOpen() {
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
    qc.setQueryData(keys.teaching(), SHARED.teaching);
    qc.setQueryData(keys.quest(PID), LENS_OPEN);
    qc.setQueryData(keys.view(PID), OPEN);
    for (const [name, status] of Object.entries(OPEN.stages)) {
      if (!status || status.status !== "fresh") continue;
      qc.setQueryData(keys.stage(PID, name), {
        stage: name,
        key: status.key,
        fresh: true,
        status: status.status,
        artifact: artifactAt(JOURNEY, name, status.key),
      });
    }
    vi.spyOn(api, "preview").mockResolvedValue(PREVIEW);
    render(
      <QueryClientProvider client={qc}>
        <QuestScreen pid={PID} stream="open" />
      </QueryClientProvider>,
    );
    return qc;
  }

  /** Keyboard focus on the dietary lens: the tapestry previews it, with no record button of its
   *  own, since the card's is the one action (finding 4; Classic's stage keeps it). */
  async function previewDietary() {
    expect(card()).toBe("card-data");
    const option = await screen.findByTestId("option-dietary");
    fireEvent.focus(option);
    await waitFor(() => expect(screen.getByTestId("stage")).toHaveAttribute("data-scene", "preview"));
    await waitFor(() => expect(screen.getByTestId("stage-loading")).not.toHaveAttribute("data-on"));
    expect(screen.getByTestId("stage")).toHaveAttribute("data-focus", "option");
    expect(screen.queryByTestId("stage-record")).toBeNull();
    return option;
  }

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("returns the tapestry to live once the card records", async () => {
    expect(RECORDED.decisions.length).toBe(OPEN.decisions.length + 1);
    lensOpen();
    const option = await previewDietary();
    const decide = vi.spyOn(api, "decide").mockResolvedValue(RECORDED);
    fireEvent.keyDown(option, { key: "Enter" });
    await waitFor(() => expect(decide).toHaveBeenCalledTimes(1));
    expect(decide.mock.calls[0]![1]).toMatchObject({ kind: "set_lens", lenses: ["dietary"] });
    await waitFor(() => expect(screen.getByTestId("stage")).toHaveAttribute("data-focus", "live"));
    expect(screen.queryByTestId("stage-record")).toBeNull();
    expect(screen.getByTestId("stage")).not.toHaveAttribute("data-scene", "preview");
  });

  it("keeps the preview while nothing is recorded: a refused press, or a view that records nothing new", async () => {
    const qc = lensOpen();
    const option = await previewDietary();
    const refusal: Refusal = { error: { code: "not_yet", message: "Not yet.", exits: [] } };
    const decide = vi.spyOn(api, "decide").mockRejectedValue(new RefusalError(refusal));
    fireEvent.keyDown(option, { key: "Enter" });
    await waitFor(() => expect(decide).toHaveBeenCalledTimes(1));
    await screen.findByText("Not yet.");
    // A view read again with the same decisions (a stage's status moved) records nothing either.
    const before = qc.getQueryData<ProjectView>(keys.view(PID))!;
    await act(async () => {
      qc.setQueryData<ProjectView>(keys.view(PID), (prev) => ({
        ...prev!,
        stages: { ...prev!.stages, ingest: { ...prev!.stages.ingest!, updated_at: "2026-10-17T00:00:02Z" } },
      }));
      await notified();
    });
    expect(qc.getQueryData(keys.view(PID))).not.toBe(before);
    expect(screen.getByTestId("stage")).toHaveAttribute("data-focus", "option");
    expect(screen.getByTestId("stage")).toHaveAttribute("data-scene", "preview");
    expect(screen.queryByTestId("stage-record")).toBeNull();
  });
});

/**
 * What waits is what is shown (FOUNDATION §7): under Predict the fit may already be computed
 * before Fit is pressed, but until the quest log says it was pressed (`fit.pressed`), the Models
 * tapestry shows no fit scene: no cross-validated comparison, no coefficients, no substitution and
 * no seal card. The view is the capture's where the opening is the Router's open question (the
 * fit computed and served whole, the held-out rows still sealed), each fresh stage's artifact as
 * captured; Models is left one objective short, so its tapestry is the live stage, not the
 * analysis flowchart.
 */
describe("the Models tapestry under Predict, before and after Fit is pressed", () => {
  const AT_OPENING = viewsOf(JOURNEY)
    .map((v, i) => expandView(JOURNEY, v, PID, `2026-10-17T00:00:${String(i).padStart(2, "0")}Z`))
    .find((v) => v.interview.some((s) => s.key === "open_seal" && s.status === "open"))!;
  /** The fit scene's parts, as Results (components/stage/results) marks each. */
  const FIT_SCENE = ['[data-purpose="model_comparison"]', '[data-purpose="coefficients"]', '[data-purpose="substitution"]', '[data-testid="open-seal-card"]'];
  beforeEach(() => {
    // The views measure their box; jsdom has no layout to measure.
    vi.stubGlobal("ResizeObserver", class { observe() {} unobserve() {} disconnect() {} });
  });

  function modelsBefore(lock: FitLock) {
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
    qc.setQueryData(keys.quest(PID), {
      ...BASE,
      fit: lock,
      stages: BASE.stages.map((s) =>
        s.key === "models" ? { ...s, progress: { answered: 4, required: 5, complete: false } } : s,
      ),
    } satisfies QuestLog);
    qc.setQueryData(keys.view(PID), AT_OPENING);
    for (const [name, status] of Object.entries(AT_OPENING.stages)) {
      if (!status || status.status !== "fresh") continue;
      qc.setQueryData(keys.stage(PID, name), {
        stage: name,
        key: status.key,
        fresh: true,
        status: status.status,
        artifact: artifactAt(JOURNEY, name, status.key),
      });
    }
    render(
      <QueryClientProvider client={qc}>
        <QuestScreen pid={PID} stream="open" />
      </QueryClientProvider>,
    );
    fireEvent.click(screen.getByTestId("stage-models"));
    expect(card()).toBe("card-models");
  }
  const fitScene = () => FIT_SCENE.filter((sel) => screen.getByTestId("tapestry").querySelector(sel));

  it("shows no fit scene until Fit is pressed, and says why in the engine's words", async () => {
    expect(AT_OPENING.stages.fit?.status).toBe("fresh");
    modelsBefore(ready);
    await waitFor(() => expect(screen.getByTestId("stage")).toHaveAttribute("data-scene", "now"));
    expect(fitScene()).toEqual([]);
    expect(screen.queryByTestId("results")).toBeNull();
    expect(screen.getByTestId("tapestry")).toHaveTextContent(ready.reason);
    expect(screen.getByTestId("tapestry")).not.toHaveTextContent("Nothing has been fit yet");
  });

  it("shows the fit scene once Fit is pressed", async () => {
    modelsBefore(pressed);
    await waitFor(() => expect(screen.getByTestId("stage")).toHaveAttribute("data-scene", "results"));
    await waitFor(() => expect(fitScene()).toEqual(FIT_SCENE));
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
