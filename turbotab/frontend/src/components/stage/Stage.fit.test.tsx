/**
 * Fit, reachable on the stage (W1-B): with the fit stage held by the scheduler, the live stage
 * carries Fit with the engine's estimate and a Cancel, instead of "Nothing has been fit yet" alone.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { FitLock, ProjectView, QuestLog, StageStatus } from "../../api/schema";
import { Stage } from "./Stage";

const PID = "p1";

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
  reason: "The plan is not locked yet. Pressing Fit locks it. The fit waits for it: it is expected to take 8 min.",
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

function mount(view: ProjectView) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) => {
      const body: unknown = url.endsWith("/quest")
        ? ({ version: 4, stages: [], kinds: {}, fit: lock } satisfies QuestLog)
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

afterEach(() => vi.unstubAllGlobals());

describe("the live stage when the fit is held", () => {
  it("shows the estimate, Fit and Cancel", async () => {
    mount(viewWith(stage("fit", { status: "idle", fresh: false, held: "fit" })));
    expect(await screen.findByRole("button", { name: "Fit" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Cancel" })).toBeEnabled();
    expect(await screen.findByTestId("fit-estimate")).toHaveTextContent("about 8 min");
  });

  it("shows no Fit when nothing holds the fit", () => {
    mount(viewWith(stage("fit", { status: "blocked", fresh: false, key: null, missing: ["models"] })));
    expect(screen.queryByRole("button", { name: "Fit" })).toBeNull();
    expect(screen.queryByTestId("fit-held")).toBeNull();
  });
});
