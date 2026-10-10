/**
 * Fit, reachable (W1-B). A fit expected to take over about two minutes waits in the scheduler for
 * Fit (core/fit_press.py): its stage is idle with `held: "fit"` and serves no artifact, so the
 * Results had nothing to show and nothing to press. The stage now says how long it will take,
 * offers Fit, and offers Cancel, which is POST /fit/cancel.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { FitLock, QuestLog, StageStatus } from "../../../api/schema";
import { FitHeld } from "./Results";

const PID = "p1";

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
  reason:
    "Under prediction nothing locks: the held-out rows are the seal. Pressing Fit opens Results; until then no score or estimate is shown. The fit waits for it: it is expected to take 5 min.",
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

afterEach(() => vi.unstubAllGlobals());

describe("a fit the scheduler holds", () => {
  it("shows what it will take, Fit and Cancel", async () => {
    serve(lock());
    mount(<FitHeld pid={PID} status={status()} />);
    const panel = await screen.findByTestId("fit-held");
    // 312 s is the engine's estimate; the page rounds it to minutes as the models card does.
    await waitFor(() => expect(within(panel).getByTestId("fit-estimate")).toHaveTextContent("about 5 min"));
    expect(within(panel).getByRole("button", { name: "Fit" })).toBeEnabled();
    expect(within(panel).getByRole("button", { name: "Cancel" })).toBeEnabled();
    // The server's own words for why it waits are the caption.
    expect(panel).toHaveTextContent("Pressing Fit opens Results");
  });

  it("Fit is POST /fit and Cancel is POST /fit/cancel", async () => {
    const calls = serve(lock());
    mount(<FitHeld pid={PID} status={status()} />);
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "Fit" }));
    await waitFor(() => expect(calls.filter((c) => c.method === "POST").map((c) => c.path)).toEqual(["/api/projects/p1/fit"]));
    await user.click(screen.getByRole("button", { name: "Cancel" }));
    await waitFor(() =>
      expect(calls.filter((c) => c.method === "POST").map((c) => c.path)).toEqual([
        "/api/projects/p1/fit",
        "/api/projects/p1/fit/cancel",
      ]),
    );
  });

  it("still offers Fit and Cancel before the quest log has said how long it takes", () => {
    vi.stubGlobal("fetch", vi.fn(() => new Promise<Response>(() => {})));
    mount(<FitHeld pid={PID} status={status()} />);
    expect(screen.getByRole("button", { name: "Fit" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Cancel" })).toBeEnabled();
    expect(screen.queryByTestId("fit-estimate")).toBeNull();
  });

  it("says nothing when the fit is not held, and nothing is running", () => {
    serve(lock({ held: false }));
    mount(<FitHeld pid={PID} status={status({ held: null })} />);
    expect(screen.queryByTestId("fit-held")).toBeNull();
  });
});

describe("a fit running after Fit was pressed", () => {
  it("offers Cancel and not Fit again", async () => {
    serve(lock({ held: false, pressed: true, opened: true }));
    mount(<FitHeld pid={PID} status={status({ status: "running", held: null, job_id: "j1", progress: 0.2 })} />);
    const panel = await screen.findByTestId("fit-held");
    expect(within(panel).getByRole("button", { name: "Cancel" })).toBeEnabled();
    expect(within(panel).queryByRole("button", { name: "Fit" })).toBeNull();
  });
});
