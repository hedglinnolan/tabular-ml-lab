/**
 * Fit, reachable (W1-B). A fit expected to take over about two minutes waits in the scheduler for
 * Fit (core/fit_press.py): its stage is idle with `held: "fit"` and serves no artifact, so the
 * Results had nothing to show and nothing to press. The stage now offers Fit, captioned with the
 * engine's own words for why it waits and how long it is expected to take.
 *
 * The lock reasons below are what the engine's `fit_press.fit_lock` says (its `cost.duration`
 * rounding); the expected durations are hand computations: 312 s / 60 = 5.2, about 5 minutes;
 * 150 s / 60 = 2.5, which Python rounds half to even, about 2 minutes (JavaScript's Math.round
 * would say 3, so a page that rounds again disagrees with the engine on the same panel).
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { FitLock, QuestLog, StageStatus } from "../../../api/schema";
import { FitHeld } from "./Results";

const PID = "p1";

const PREDICT =
  "Under prediction nothing locks: the held-out rows are the seal. Pressing Fit opens Results; until then no score or estimate is shown.";

const lock = (patch: Partial<FitLock> = {}): FitLock => ({
  purpose: "prediction",
  locks: false,
  locked: false,
  at: null,
  sha256: null,
  pressed: false,
  opened: false,
  held: true,
  estimate_seconds: 312,
  reason: `${PREDICT} The fit waits for it: it is expected to take about 5 minutes.`,
  ...patch,
});

const quest = (fit: FitLock | null): QuestLog => ({ version: 4, stages: [], kinds: {}, fit });

const status = (patch: Partial<StageStatus> = {}): StageStatus => ({
  stage: "fit",
  status: "idle",
  key: "k-fit",
  fresh: false,
  missing: [],
  error: null,
  job_id: null,
  progress: null,
  updated_at: "2026-10-10T10:00:00.000Z",
  cancelled: false,
  held: "fit",
  refusal: null,
  ...patch,
});

interface Call {
  path: string;
  method: string;
}

function serve(fit: FitLock | null) {
  const calls: Call[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, init?: RequestInit) => {
      const method = init?.method ?? "GET";
      calls.push({ path: url, method });
      const body = url.endsWith("/quest") ? quest(fit) : fit;
      return new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });
    }),
  );
  return calls;
}

function mount(node: ReactNode) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={qc}>{node}</QueryClientProvider>);
}

/** Every duration the panel states: the time must be said once, in the engine's words. */
const durations = (text: string) => text.match(/\d+\s*(?:minutes?|min|hours?|h|seconds?|s)(?![a-z])/g) ?? [];

afterEach(() => vi.unstubAllGlobals());

describe("a fit the scheduler holds", () => {
  it("offers Fit, captioned with the engine's reason and its estimate, said once", async () => {
    serve(lock());
    mount(<FitHeld pid={PID} status={status()} />);
    const panel = await screen.findByTestId("fit-held");
    await waitFor(() => expect(panel).toHaveTextContent("it is expected to take about 5 minutes."));
    expect(panel).toHaveTextContent(PREDICT);
    expect(durations(panel.textContent ?? "")).toEqual(["5 minutes"]);
    expect(within(panel).getByRole("button", { name: "Fit" })).toBeEnabled();
  });

  it("never re-rounds the engine's estimate: 150 s is about 2 minutes, not 3", async () => {
    serve(lock({ estimate_seconds: 150, reason: `${PREDICT} The fit waits for it: it is expected to take about 2 minutes.` }));
    mount(<FitHeld pid={PID} status={status()} />);
    const panel = await screen.findByTestId("fit-held");
    await waitFor(() => expect(panel).toHaveTextContent("about 2 minutes"));
    expect(panel).not.toHaveTextContent("3 min");
    expect(durations(panel.textContent ?? "")).toEqual(["2 minutes"]);
  });

  it("offers no Cancel: a held fit runs nothing, and the fit-wide Cancel would stop other work", async () => {
    // Re-armed (pressed before, the estimate grew past 1.5 times): POST /fit/cancel would also
    // withdraw the press and the plan's lock, and stop every estimate stage's job.
    serve(lock({ pressed: true, opened: true }));
    mount(<FitHeld pid={PID} status={status()} />);
    const panel = await screen.findByTestId("fit-held");
    await waitFor(() => expect(panel).toHaveTextContent("about 5 minutes"));
    expect(within(panel).queryByRole("button", { name: "Cancel" })).toBeNull();
    expect(within(panel).getAllByRole("button")).toHaveLength(1);
  });

  it("Fit is POST /fit", async () => {
    const calls = serve(lock());
    mount(<FitHeld pid={PID} status={status()} />);
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "Fit" }));
    await waitFor(() => expect(calls.filter((c) => c.method === "POST").map((c) => c.path)).toEqual(["/api/projects/p1/fit"]));
  });

  it("before the quest log arrives, offers Fit with no words the engine did not say", () => {
    vi.stubGlobal("fetch", vi.fn(() => new Promise<Response>(() => {})));
    mount(<FitHeld pid={PID} status={status()} />);
    const panel = screen.getByTestId("fit-held");
    expect(within(panel).getByRole("button", { name: "Fit" })).toBeEnabled();
    expect(panel.textContent).toBe("Fit");
  });

  it("says nothing when the fit is not held", () => {
    serve(lock({ held: false }));
    mount(<FitHeld pid={PID} status={status({ held: null })} />);
    expect(screen.queryByTestId("fit-held")).toBeNull();
  });
});

describe("a fit running after Fit was pressed", () => {
  it("adds no second Cancel: the job chip's Cancel is the one", async () => {
    serve(lock({ held: false, pressed: true, opened: true }));
    mount(<FitHeld pid={PID} status={status({ status: "running", held: null, job_id: "j1", progress: 0.2 })} />);
    // Long enough for a quest log, were one asked for, to arrive and render.
    await new Promise((r) => setTimeout(r, 50));
    expect(screen.queryByTestId("fit-held")).toBeNull();
    expect(screen.queryByTestId("cancel-fit")).toBeNull();
  });
});
