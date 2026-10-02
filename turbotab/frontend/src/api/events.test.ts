import { QueryClient } from "@tanstack/react-query";
import { beforeEach, describe, expect, it } from "vitest";
import { applyProjectEvent, parseEvent, reconcileStages } from "./events";
import { keys, mergeView } from "./queries";
import type { DecisionRecord, JobView, ProjectView, StageResult, StageStatus } from "./schema";

const PID = "p1";

function status(stage: string, patch: Partial<StageStatus> = {}): StageStatus {
  return {
    stage,
    status: "fresh",
    key: "k1",
    fresh: true,
    missing: [],
    error: null,
    job_id: null,
    progress: null,
    updated_at: "2026-09-27T10:00:00.000Z",
    cancelled: false,
    ...patch,
  };
}

function record(seq: number): DecisionRecord {
  return {
    id: `d${seq}`,
    seq,
    at: "2026-09-27T10:00:00.000Z",
    note: null,
    sentence: null,
    post_seal: false,
    decision: { kind: "set_target", column: `c${seq}` },
  };
}

function view(): ProjectView {
  return {
    summary: {
      id: PID,
      name: "dietary recalls",
      created_at: "2026-09-27T09:00:00.000Z",
      source_kind: "path",
      source_name: "dietary_recalls.csv",
      n_rows: 600,
      n_cols: 17,
      ingest: null,
    },
    state: {
      lens: null,
      target: "c1",
      task: null,
      purpose: null,
      roles: null,
      energy_adjustment: null,
      exclusions: null,
      missing: null,
      split: null,
      models: null,
      substitution: null,
      orientation: null,
      event: null,
      grain: null,
      repeat_kind: null,
      unit: null,
      aggregation: null,
      temporal: null,
      seal_opened: null,
      findings: null,
      feature_table: null,
      categorical: null,
      exposure_forms: null,
      outcome_order: null,
    },
    decisions: [record(1)],
    stages: {
      ingest: status("ingest"),
      target_info: status("target_info", { key: "t1" }),
    },
    interview: [],
  };
}

let qc: QueryClient;
const cachedView = () => qc.getQueryData<ProjectView>(keys.view(PID))!;
const isInvalid = (key: readonly unknown[]) => qc.getQueryState(key)?.isInvalidated === true;

beforeEach(() => {
  qc = new QueryClient({ defaultOptions: { queries: { staleTime: Infinity } } });
  qc.setQueryData(keys.view(PID), view());
});

describe("a decision event", () => {
  it("appends a record the cache does not have, in seq order, and refetches the folded state", () => {
    applyProjectEvent(qc, PID, { type: "decision", data: record(2) });
    expect(cachedView().decisions.map((d) => d.id)).toEqual(["d1", "d2"]);
    expect(isInvalid(keys.view(PID))).toBe(true);
  });

  it("is a no-op for a record already in the cache (our own POST answered first)", () => {
    applyProjectEvent(qc, PID, { type: "decision", data: record(1) });
    expect(cachedView().decisions).toHaveLength(1);
    expect(isInvalid(keys.view(PID))).toBe(false);
  });
});

describe("a stage event", () => {
  it("patches the stage status in place", () => {
    const stale = status("target_info", {
      status: "stale",
      fresh: false,
      key: "t2",
      updated_at: "2026-09-27T10:00:01.000Z",
    });
    applyProjectEvent(qc, PID, { type: "stage", data: stale });
    expect(cachedView().stages.target_info).toEqual(stale);
    expect(cachedView().stages.ingest!.status).toBe("fresh"); // others untouched
  });

  it("never lets an older status overwrite a newer one", () => {
    const newer = status("target_info", {
      status: "running",
      updated_at: "2026-09-27T10:00:05.000Z",
    });
    const older = status("target_info", {
      status: "stale",
      updated_at: "2026-09-27T10:00:02.000Z",
    });
    applyProjectEvent(qc, PID, { type: "stage", data: newer });
    applyProjectEvent(qc, PID, { type: "stage", data: older });
    expect(cachedView().stages.target_info!.status).toBe("running");
  });

  it("keeps the older artifact while stale, and refetches only when fresh with a new key", () => {
    const old: StageResult = {
      stage: "target_info",
      key: "t1",
      fresh: true,
      status: "fresh",
      artifact: { column: "c1" },
    };
    qc.setQueryData(keys.stage(PID, "target_info"), old);

    applyProjectEvent(qc, PID, {
      type: "stage",
      data: status("target_info", {
        status: "stale",
        fresh: false,
        key: "t2",
        updated_at: "2026-09-27T10:00:01.000Z",
      }),
    });
    expect(isInvalid(keys.stage(PID, "target_info"))).toBe(false);
    expect(qc.getQueryData(keys.stage(PID, "target_info"))).toBe(old);

    applyProjectEvent(qc, PID, {
      type: "stage",
      data: status("target_info", {
        status: "fresh",
        key: "t2",
        updated_at: "2026-09-27T10:00:02.000Z",
      }),
    });
    expect(isInvalid(keys.stage(PID, "target_info"))).toBe(true);
  });

  it("does not refetch a fresh stage whose key the cache already holds", () => {
    qc.setQueryData(keys.stage(PID, "target_info"), {
      stage: "target_info",
      key: "t1",
      fresh: true,
      status: "fresh",
      artifact: {},
    } satisfies StageResult);
    applyProjectEvent(qc, PID, {
      type: "stage",
      data: status("target_info", { key: "t1", updated_at: "2026-09-27T10:00:03.000Z" }),
    });
    expect(isInvalid(keys.stage(PID, "target_info"))).toBe(false);
  });

  it("refetches the view when a status changes, so the Router's interview stays current", () => {
    applyProjectEvent(qc, PID, {
      type: "stage",
      data: status("target_info", {
        status: "running",
        fresh: false,
        key: "t2",
        updated_at: "2026-09-27T10:00:01.000Z",
      }),
    });
    expect(isInvalid(keys.view(PID))).toBe(true);
  });

  it("does not refetch the view for a repeat of the same status", () => {
    applyProjectEvent(qc, PID, {
      type: "stage",
      data: status("target_info", { key: "t1", updated_at: "2026-09-27T10:00:02.000Z" }),
    });
    expect(isInvalid(keys.view(PID))).toBe(false);
  });

  it("refetches the table and column summaries when ingest turns fresh", () => {
    qc.setQueryData(keys.table(PID, 0, 50, "a,b"), {
      columns: [],
      rows: [],
      total_rows: 0,
      offset: 0,
    });
    qc.setQueryData(keys.columns(PID), []);
    applyProjectEvent(qc, PID, {
      type: "stage",
      data: status("ingest", { key: "i2", updated_at: "2026-09-27T10:00:04.000Z" }),
    });
    expect(isInvalid(keys.table(PID, 0, 50, "a,b"))).toBe(true);
    expect(isInvalid(keys.columns(PID))).toBe(true);
  });
});

describe("a job event", () => {
  const job = (patch: Partial<JobView>): JobView => ({
    job_id: "j1",
    label: "Reading the outcome column",
    stage: "target_info",
    state: "running",
    progress: 0.4,
    message: null,
    error: null,
    ...patch,
  });

  it("stores the job and mirrors its progress onto the stage it serves", () => {
    qc.setQueryData(keys.view(PID), {
      ...view(),
      stages: {
        ...view().stages,
        target_info: status("target_info", { status: "running", job_id: "j1", progress: 0.1 }),
      },
    });
    applyProjectEvent(qc, PID, { type: "job", data: job({}) });
    expect(qc.getQueryData<JobView>(keys.job(PID, "j1"))!.progress).toBe(0.4);
    expect(cachedView().stages.target_info!.progress).toBe(0.4);
  });

  it("does not touch a stage that is running a different job", () => {
    qc.setQueryData(keys.view(PID), {
      ...view(),
      stages: {
        ...view().stages,
        target_info: status("target_info", { status: "running", job_id: "j9", progress: 0.1 }),
      },
    });
    applyProjectEvent(qc, PID, { type: "job", data: job({}) });
    expect(cachedView().stages.target_info!.progress).toBe(0.1);
  });
});

describe("resync", () => {
  it("invalidates every query under the project and nothing else", () => {
    qc.setQueryData(keys.stage(PID, "findings"), {
      stage: "findings",
      key: null,
      fresh: false,
      status: "idle",
      artifact: null,
    });
    qc.setQueryData(keys.stage("other", "findings"), {
      stage: "findings",
      key: null,
      fresh: false,
      status: "idle",
      artifact: null,
    });
    qc.setQueryData(keys.projects(), []);
    applyProjectEvent(qc, PID, { type: "resync", data: {} });
    expect(isInvalid(keys.view(PID))).toBe(true);
    expect(isInvalid(keys.stage(PID, "findings"))).toBe(true);
    expect(isInvalid(keys.stage("other", "findings"))).toBe(false);
    expect(isInvalid(keys.projects())).toBe(false);
  });
});

describe("parseEvent", () => {
  it("parses known frames and ignores unknown or malformed ones", () => {
    expect(parseEvent("stage", JSON.stringify(status("ingest")))?.type).toBe("stage");
    expect(parseEvent("resync", "{}")).toEqual({ type: "resync", data: {} });
    expect(parseEvent("ping", "")).toEqual({ type: "ping", data: {} });
    expect(parseEvent("mystery", "{}")).toBeNull();
    expect(parseEvent("stage", "{not json")).toBeNull();
  });
});

describe("mergeView", () => {
  it("keeps newer stage statuses and the union of decisions when an older response lands", () => {
    const prev = view();
    prev.stages.target_info = status("target_info", {
      status: "running",
      updated_at: "2026-09-27T10:00:09.000Z",
    });
    prev.decisions = [record(1), record(2)];
    const next = view();
    next.stages.target_info = status("target_info", {
      status: "queued",
      updated_at: "2026-09-27T10:00:08.000Z",
    });
    next.decisions = [record(1), record(3)];
    const merged = mergeView(prev, next);
    expect(merged.stages.target_info!.status).toBe("running");
    expect(merged.decisions.map((d) => d.seq)).toEqual([1, 2, 3]);
  });

  it("does not let a view built before the last decision roll the state back", () => {
    // The cache holds the POST's answer (decision 2 folded in, the stage already fresh);
    // a GET that the server answered before decision 2 lands afterwards.
    const prev = view();
    prev.decisions = [record(1), record(2)];
    prev.state = { ...prev.state, target: "c2" };
    prev.stages.target_info = status("target_info", {
      key: "t2",
      updated_at: "2026-09-27T10:00:05.000Z",
    });
    const late = view();
    late.stages.target_info = status("target_info", {
      status: "running",
      fresh: false,
      key: "t1",
      updated_at: "2026-09-27T10:00:04.000Z",
    });
    const merged = mergeView(prev, late);
    expect(merged.state.target).toBe("c2");
    expect(merged.stages.target_info!.status).toBe("fresh");
    expect(merged.stages.target_info!.key).toBe("t2");
    // A response that has seen every decision is taken as it is.
    const current = view();
    current.decisions = [record(1), record(2)];
    current.state = { ...current.state, target: "c2", purpose: "prediction" };
    expect(mergeView(prev, current).state.purpose).toBe("prediction");
  });
});

describe("reconcileStages", () => {
  it("reads again a result fetched while the stage ran, once the view says it is fresh", () => {
    // The race behind "the file is still being read" forever: the result was fetched while
    // ingest ran, and the event that it finished never refetched it.
    const running: StageResult = { stage: "ingest", key: null, fresh: false, status: "running", artifact: null };
    qc.setQueryData(keys.stage(PID, "ingest"), running);
    const current: StageResult = { stage: "target_info", key: "t1", fresh: true, status: "fresh", artifact: {} };
    qc.setQueryData(keys.stage(PID, "target_info"), current);
    expect(reconcileStages(qc, PID, cachedView())).toEqual(["ingest"]);
    expect(isInvalid(keys.stage(PID, "ingest"))).toBe(true);
    expect(isInvalid(keys.stage(PID, "target_info"))).toBe(false); // already the fresh key
  });

  it("reads again a fresh result under another key, and leaves a stage that is not fresh", () => {
    qc.setQueryData(keys.stage(PID, "ingest"), { stage: "ingest", key: "old", fresh: true, status: "fresh", artifact: {} });
    const v = cachedView();
    v.stages.target_info = status("target_info", { status: "running", fresh: false });
    qc.setQueryData(keys.stage(PID, "target_info"), { stage: "target_info", key: null, fresh: false, status: "running", artifact: null });
    expect(reconcileStages(qc, PID, v)).toEqual(["ingest"]);
    expect(isInvalid(keys.stage(PID, "target_info"))).toBe(false);
  });
});
