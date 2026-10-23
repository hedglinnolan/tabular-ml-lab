/**
 * Fit, reachable on the stage (W1-B): with the fit stage held by the scheduler, the live stage
 * carries Fit captioned with the engine's reason and estimate, instead of "Nothing has been fit
 * yet" alone; and Fit is offered once per screen, also when an older fit is served withheld.
 *
 * The reason is what `fit_press.fit_lock` says for 480 s (480 / 60 = 8, about 8 minutes, by hand).
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { FitLock, ProjectView, QuestLog, StageStatus } from "../../api/schema";
import { Stage } from "./Stage";

const PID = "p1";

const LOCK_FIRST =
  "No estimate is shown until Fit is pressed: pressing it locks the analysis plan, so the plan is declared before any estimate is seen.";

const lock: FitLock = {
  purpose: "inference",
  locks: true,
  locked: false,
  at: null,
  sha256: null,
  pressed: false,
  opened: false,
  held: true,
  estimate_seconds: 480,
  reason:
    "The plan is not locked yet. Pressing Fit locks it and shows the first estimates; until then no estimate is shown. The fit waits for it: it is expected to take about 8 minutes.",
};

const stage = (name: string, patch: Partial<StageStatus>): StageStatus => ({
  stage: name,
  status: "fresh",
  key: `k-${name}`,
  fresh: true,
  missing: [],
  error: null,
  job_id: null,
  progress: null,
  updated_at: "2026-10-10T10:00:00.000Z",
  cancelled: false,
  held: null,
  refusal: null,
  ...patch,
});

function viewWith(fit: StageStatus): ProjectView {
  return {
    summary: { id: PID, name: "NHANES", n_rows: 5000, n_cols: 40 },
    state: { purpose: "inference", target: "LBXGLU", roles: null, models: ["ridge"] },
    decisions: [],
    interview: [],
    stages: { fit },
  } as unknown as ProjectView;
}

/** The older fit as GET /stages/fit serves it while Fit is not pressed: every estimate withheld. */
const withheldFit = {
  stage: "fit",
  key: "k-fit-old",
  fresh: false,
  status: "stale",
  artifact: { withheld_for: "fit", withheld: LOCK_FIRST, models: [] },
};

function mount(view: ProjectView, fitResult: unknown = {}) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) => {
      const body: unknown = url.endsWith("/quest")
        ? ({ version: 4, stages: [], kinds: {}, fit: lock } satisfies QuestLog)
        : url.endsWith("/stages/fit")
          ? fitResult
          : {};
      return new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });
    }),
  );
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={qc}>
      <Stage pid={PID} view={view} focus={{ kind: "live" }} onFocus={() => {}} />
    </QueryClientProvider>,
  );
}

const durations = (text: string) => text.match(/\d+\s*(?:minutes?|min|hours?|h|seconds?|s)(?![a-z])/g) ?? [];

afterEach(() => vi.unstubAllGlobals());

describe("the live stage when the fit is held", () => {
  it("shows Fit with the engine's reason and estimate", async () => {
    mount(viewWith(stage("fit", { status: "idle", fresh: false, held: "fit" })));
    expect(await screen.findByRole("button", { name: "Fit" })).toBeEnabled();
    await waitFor(() => expect(screen.getByTestId("fit-held")).toHaveTextContent("expected to take about 8 minutes."));
    expect(screen.queryByRole("button", { name: "Cancel" })).toBeNull();
    expect(durations(screen.getByTestId("stage").textContent ?? "")).toEqual(["8 minutes"]);
  });

  it("offers Fit once when an older fit is served withheld for it", async () => {
    // Fit was fit live before; the models changed to a set expected to take 8 minutes: the stage
    // is stale and held, and the older artifact comes back withheld for Fit.
    mount(viewWith(stage("fit", { status: "stale", fresh: false, held: "fit" })), withheldFit);
    await waitFor(() => expect(screen.getByTestId("stage")).toHaveAttribute("data-scene", "results"));
    await waitFor(() => expect(screen.getByTestId("stage")).toHaveTextContent("expected to take about 8 minutes."));
    expect(screen.getAllByRole("button", { name: "Fit" })).toHaveLength(1);
    expect(screen.getAllByTestId("fit-press")).toHaveLength(1);
    expect(durations(screen.getByTestId("stage").textContent ?? "")).toEqual(["8 minutes"]);
  });

  it("shows no Fit when nothing holds the fit", () => {
    mount(viewWith(stage("fit", { status: "blocked", fresh: false, key: null, missing: ["models"] })));
    expect(screen.queryByRole("button", { name: "Fit" })).toBeNull();
    expect(screen.queryByTestId("fit-held")).toBeNull();
  });
});
