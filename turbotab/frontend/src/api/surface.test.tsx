/**
 * The API surface the walkable shell reads (W1-A): the quest log and the record's outputs as
 * hooks, Fit's cancel, the confirm-sweep decision, `decide_now`, and the export. Each hook is
 * checked against the URL and method it sends (fetch mocked), and the SSE events that change
 * what the quest log says are checked to refetch it.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { api, RefusalError } from "./client";
import { applyProjectEvent } from "./events";
import {
  keys,
  useCancelFit,
  useChecklist,
  useConfirmSweep,
  useDecideNow,
  useExport,
  useMateriality,
  useProjectView,
  useQuest,
  useRecord,
  useTriage,
} from "./queries";
import type { DecisionRecord, StageStatus } from "./schema";

const PID = "p 1"; // a space: the id is encoded in every path

interface Call {
  url: string;
  method: string;
  body: unknown;
}

function serve(replies: Record<string, { status?: number; body: unknown; type?: string }> = {}) {
  const calls: Call[] = [];
  const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
    calls.push({
      url,
      method: init?.method ?? "GET",
      body: typeof init?.body === "string" ? (JSON.parse(init.body) as unknown) : null,
    });
    const path = url.split("?")[0]!;
    const reply = replies[path] ?? { body: {} };
    return new Response(JSON.stringify(reply.body), {
      status: reply.status ?? 200,
      headers: { "Content-Type": reply.type ?? "application/json" },
    });
  });
  vi.stubGlobal("fetch", fetchMock);
  return calls;
}

function wrapper() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const Wrap = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={qc}>{children}</QueryClientProvider>
  );
  return { qc, Wrap };
}

afterEach(() => vi.unstubAllGlobals());

const BASE = "/api/projects/p%201";

describe("the quest log and the record's outputs are read from their routes", () => {
  const reads: [string, () => unknown, string][] = [
    ["quest", () => useQuest(PID), "/quest"],
    ["record", () => useRecord(PID), "/record"],
    ["triage", () => useTriage(PID), "/triage"],
    ["materiality", () => useMateriality(PID), "/materiality"],
    ["checklist", () => useChecklist(PID), "/checklist"],
  ];
  for (const [name, hook, path] of reads) {
    it(`${name} is GET ${path}`, async () => {
      const calls = serve({ [`${BASE}${path}`]: { body: { marker: name } } });
      const { Wrap } = wrapper();
      const { result } = renderHook(() => hook() as { data?: unknown }, { wrapper: Wrap });
      await waitFor(() => expect(result.current.data).toEqual({ marker: name }));
      expect(calls).toEqual([{ url: `${BASE}${path}`, method: "GET", body: null }]);
    });
  }

  it("a hook given enabled=false asks for nothing", async () => {
    const calls = serve();
    const { Wrap } = wrapper();
    renderHook(() => useQuest(PID, false), { wrapper: Wrap });
    await new Promise((r) => setTimeout(r, 20));
    expect(calls).toEqual([]);
  });
});

describe("Fit", () => {
  it("cancel is POST /fit/cancel and the lock it returns replaces the quest log's", async () => {
    const lock = { held: true, locked: false, estimate_seconds: 300 };
    const calls = serve({ [`${BASE}/fit/cancel`]: { body: lock } });
    const { qc, Wrap } = wrapper();
    qc.setQueryData(keys.quest(PID), { version: 4, stages: [], kinds: {}, fit: { ...lock, locked: true } });
    const { result } = renderHook(() => useCancelFit(PID), { wrapper: Wrap });
    await act(async () => {
      await result.current.mutateAsync();
    });
    expect(calls).toEqual([{ url: `${BASE}/fit/cancel`, method: "POST", body: null }]);
    expect((qc.getQueryData(keys.quest(PID)) as { fit: typeof lock }).fit).toEqual(lock);
  });
});

describe("decisions", () => {
  it("confirm_sweep is posted through /decisions with the gate's stage, the sweep and the lines", async () => {
    const calls = serve({ [`${BASE}/decisions`]: { body: { decisions: [], stages: {} } } });
    const { Wrap } = wrapper();
    const { result } = renderHook(() => useConfirmSweep(PID), { wrapper: Wrap });
    await act(async () => {
      await result.current.mutateAsync({
        stage: "whos_in",
        sweep: "noticings",
        lines: [{ id: "n1", key: "disposition", value: "act_on_it" }],
      });
    });
    expect(calls).toEqual([
      {
        url: `${BASE}/decisions`,
        method: "POST",
        body: {
          kind: "confirm_sweep",
          stage: "whos_in",
          sweep: "noticings",
          lines: [{ id: "n1", key: "disposition", value: "act_on_it" }],
        },
      },
    ]);
  });

  it("confirm_sweep with no lines confirms the defaults as listed", async () => {
    const calls = serve({ [`${BASE}/decisions`]: { body: { decisions: [], stages: {} } } });
    await api.confirmSweep(PID, { stage: "data", sweep: "defaults" });
    expect(calls[0]!.body).toEqual({ kind: "confirm_sweep", stage: "data", sweep: "defaults" });
  });

  it("decide_now rides as a query parameter, and only when asked for", async () => {
    const calls = serve({ [`${BASE}/decisions`]: { body: { decisions: [], stages: {} } } });
    const d = { kind: "set_target", column: "LBXGLU" } as const;
    await api.decide(PID, d);
    await api.decide(PID, d, { decideNow: true });
    expect(calls.map((c) => c.url)).toEqual([`${BASE}/decisions`, `${BASE}/decisions?decide_now=true`]);
    const { Wrap } = wrapper();
    const { result } = renderHook(() => useDecideNow(PID), { wrapper: Wrap });
    await act(async () => {
      await result.current.mutateAsync(d);
    });
    expect(calls[2]!.url).toBe(`${BASE}/decisions?decide_now=true`);
  });
});

describe("the export", () => {
  it("is GET /export and arrives as a file with the name the server gave it", async () => {
    const fetchMock = vi.fn(
      async (_url: string) =>
        new Response("PK", {
          status: 200,
          headers: {
            "Content-Type": "application/zip",
            "Content-Disposition": 'attachment; filename="nhanes-turbotab-export.zip"',
          },
        }),
    );
    vi.stubGlobal("fetch", fetchMock);
    const file = await api.exportBundle(PID);
    expect(fetchMock.mock.calls[0]![0]).toBe(`${BASE}/export`);
    expect(file.filename).toBe("nhanes-turbotab-export.zip");
    expect(file.blob.size).toBe(2);
  });

  it("a refusal (409) is thrown with what is missing, as every other refusal is", async () => {
    serve({
      [`${BASE}/export`]: {
        status: 409,
        body: { error: { code: "export_not_ready", message: "The plan is open.", exits: [] } },
      },
    });
    const { Wrap } = wrapper();
    const { result } = renderHook(() => useExport(PID), { wrapper: Wrap });
    const err = await act(async () => result.current.mutateAsync().catch((e: unknown) => e));
    expect(err).toBeInstanceOf(RefusalError);
  });
});

describe("the quest log follows the record", () => {
  const record: DecisionRecord = {
    id: "d2",
    seq: 2,
    at: "2026-10-10T10:00:00.000Z",
    note: null,
    sentence: null,
    post_seal: false,
    after_estimates: false,
    recorded_by: "you",
    early: null,
    decision: { kind: "set_target", column: "LBXGLU" },
  };
  const status = (patch: Partial<StageStatus>): StageStatus => ({
    stage: "fit",
    status: "idle",
    key: "k",
    fresh: false,
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

  it("a decision event marks the quest log, the record, the triage, the ledger and the checklist out of date", () => {
    const qc = new QueryClient();
    const keysOut = [keys.quest, keys.record, keys.triage, keys.materiality, keys.checklist].map((k) => k(PID));
    for (const k of keysOut) qc.setQueryData(k, { marker: 1 });
    applyProjectEvent(qc, PID, { type: "decision", data: record });
    for (const k of keysOut) expect(qc.getQueryState(k)!.isInvalidated, String(k)).toBe(true);
  });

  it("a stage event that changes a stage's state marks the quest log out of date", () => {
    const qc = new QueryClient();
    qc.setQueryData(keys.quest(PID), { marker: 1 });
    const running = status({ status: "running", progress: 0.1 });
    qc.setQueryData(keys.view(PID), {
      summary: {}, state: {}, decisions: [], interview: [], stages: { fit: running },
    });
    applyProjectEvent(qc, PID, { type: "stage", data: status({ status: "fresh", fresh: true, updated_at: "2026-10-10T10:00:05.000Z" }) });
    expect(qc.getQueryState(keys.quest(PID))!.isInvalidated).toBe(true);
  });

  it("a progress tick alone does not", () => {
    const qc = new QueryClient();
    qc.setQueryData(keys.quest(PID), { marker: 1 });
    const running = status({ status: "running", progress: 0.1 });
    qc.setQueryData(keys.view(PID), {
      summary: {}, state: {}, decisions: [], interview: [], stages: { fit: running },
    });
    applyProjectEvent(qc, PID, {
      type: "stage",
      data: { ...running, progress: 0.2, updated_at: "2026-10-10T10:00:01.000Z" },
    });
    expect(qc.getQueryState(keys.quest(PID))!.isInvalidated).toBe(false);
  });

  // A read of the view can hold a stage's new state before its event: the engine reports a stage
  // fresh once its result is written, before the run's end publishes that it finished (stamped
  // then), and a response can beat an event on the wire. The event that follows then finds the view
  // holding its state already. The quest log was read before either, while Who's in's results were
  // out of date (CI's quest-nhanes-predict journey: the bar kept Who's in reopened, GET /quest none).
  const viewWith = (fit: StageStatus) => ({ summary: {}, state: {}, decisions: [], interview: [], stages: { fit } });
  // [what, the view held, the view read, the event that follows it]
  const moves: [string, StageStatus, StageStatus, StageStatus][] = [
    [
      "a result computed again (its stage no longer out of date)",
      status({ status: "running", progress: 0.9 }),
      status({ status: "fresh", fresh: true }), // written, stamped as when it started running
      status({ status: "fresh", fresh: true, updated_at: "2026-10-10T10:00:05.000Z" }), // the run's end
    ],
    [
      "a result out of date (a reopening)",
      status({ status: "fresh", fresh: true }),
      status({ status: "stale", key: "k2", updated_at: "2026-10-10T10:00:01.000Z" }),
      status({ status: "stale", key: "k2", updated_at: "2026-10-10T10:00:01.000Z" }),
    ],
  ];
  for (const [what, held, read, event] of moves) {
    it(`a view read that brings ${what} before its event marks the quest log out of date`, async () => {
      serve({ [BASE]: { body: viewWith(read) } });
      const { qc, Wrap } = wrapper();
      qc.setQueryData(keys.view(PID), viewWith(held));
      qc.setQueryData(keys.quest(PID), { marker: "read before the change" });
      const { result } = renderHook(() => useProjectView(PID), { wrapper: Wrap });
      await waitFor(() => expect(result.current.data?.stages.fit?.status).toBe(read.status));
      applyProjectEvent(qc, PID, { type: "stage", data: event });
      expect(qc.getQueryState(keys.quest(PID))!.isInvalidated).toBe(true);
    });
  }

  it("a view read that brings no new state (a progress tick, or a status older than the cache's) leaves it as read", async () => {
    const running = status({ status: "running", progress: 0.4, updated_at: "2026-10-10T10:00:02.000Z" });
    const older = status({ status: "queued", updated_at: "2026-10-10T10:00:01.000Z" });
    serve({ [BASE]: { body: { ...viewWith(older), stages: { fit: older, split: { ...running, stage: "split", progress: 0.6 } } } } });
    const { qc, Wrap } = wrapper();
    qc.setQueryData(keys.view(PID), { ...viewWith(running), stages: { fit: running, split: { ...running, stage: "split" } } });
    qc.setQueryData(keys.quest(PID), { marker: 1 });
    const { result } = renderHook(() => useProjectView(PID), { wrapper: Wrap });
    await waitFor(() => expect(result.current.data?.stages.split?.progress).toBe(0.6));
    expect(result.current.data?.stages.fit?.status).toBe("running"); // the cache's newer status kept
    expect(qc.getQueryState(keys.quest(PID))!.isInvalidated).toBe(false);
  });

  it("a resync refetches it with everything else under the project", () => {
    const qc = new QueryClient();
    qc.setQueryData(keys.quest(PID), { marker: 1 });
    applyProjectEvent(qc, PID, { type: "resync", data: {} });
    expect(qc.getQueryState(keys.quest(PID))!.isInvalidated).toBe(true);
  });
});
