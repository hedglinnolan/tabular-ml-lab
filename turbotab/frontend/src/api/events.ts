/**
 * Server-sent events -> the query cache. The server pushes; the client never polls.
 *
 *   decision  append the record to the cached view; refetch the view for the folded state
 *   stage     patch the stage status in place; refetch the view when the status changed
 *             (the Router's interview reads it); when a stage turns fresh with a key the
 *             cached result does not have, refetch that stage (and the table after ingest)
 *   job       store the JobView and mirror its progress onto the stage it serves
 *   resync    refetch everything under [pid]
 */
import { useEffect, useState } from "react";
import { useQueryClient, type QueryClient } from "@tanstack/react-query";
import { eventsUrl } from "./client";
import { keys } from "./queries";
import type {
  DecisionRecord,
  JobView,
  ProjectEvent,
  ProjectView,
  StageResult,
  StageStatus,
} from "./schema";

export const EVENT_TYPES = ["decision", "stage", "job", "resync", "ping"] as const;

function newer(a: StageStatus | undefined, b: StageStatus): boolean {
  // True when `b` should replace `a`. Unknown times never beat a known one.
  if (!a) return true;
  if (!b.updated_at) return !a.updated_at;
  if (!a.updated_at) return true;
  return Date.parse(b.updated_at) >= Date.parse(a.updated_at);
}

function applyDecision(qc: QueryClient, pid: string, record: DecisionRecord): void {
  const view = qc.getQueryData<ProjectView>(keys.view(pid));
  if (view && view.decisions.some((d) => d.id === record.id)) return; // already have it
  if (view) {
    const decisions = [...view.decisions, record].sort((a, b) => a.seq - b.seq);
    qc.setQueryData<ProjectView>(keys.view(pid), { ...view, decisions });
  }
  // The folded state (and any revert) is the server's to compute.
  void qc.invalidateQueries({ queryKey: keys.view(pid), exact: true });
}

function applyStage(qc: QueryClient, pid: string, status: StageStatus): void {
  const view = qc.getQueryData<ProjectView>(keys.view(pid));
  if (view) {
    const current = view.stages[status.stage];
    if (!newer(current, status)) return;
    qc.setQueryData<ProjectView>(keys.view(pid), {
      ...view,
      stages: { ...view.stages, [status.stage]: status },
    });
    // The Router's interview reads stage statuses (a question waits while a stage it needs
    // computes), and no event carries it: refetch the view when a status really changes.
    if (current?.status !== status.status || current?.cancelled !== status.cancelled) {
      void qc.invalidateQueries({ queryKey: keys.view(pid), exact: true });
    }
  }
  if (status.status !== "fresh") return; // an older artifact stays cached and is shown veiled
  const cached = qc.getQueryData<StageResult>(keys.stage(pid, status.stage));
  if (!cached || !cached.fresh || cached.key !== status.key) {
    void qc.invalidateQueries({ queryKey: keys.stage(pid, status.stage), exact: true });
  }
  // The rows and columns the UI reads are the working table's (M2_CONTRACT §2): the raw file
  // until the table is turned around or combined, so each of these stages changes them.
  if (status.stage === "ingest" || status.stage === "oriented" || status.stage === "working") {
    void qc.invalidateQueries({ queryKey: keys.tableAll(pid) });
    void qc.invalidateQueries({ queryKey: keys.columns(pid), exact: true });
    void qc.invalidateQueries({ queryKey: keys.view(pid), exact: true }); // n_rows, n_cols
  }
}

function applyJob(qc: QueryClient, pid: string, job: JobView): void {
  qc.setQueryData(keys.job(pid, job.job_id), job);
  if (!job.stage) return;
  const view = qc.getQueryData<ProjectView>(keys.view(pid));
  const status = view?.stages[job.stage];
  if (!view || !status || status.job_id !== job.job_id) return;
  if (job.state !== "running" && job.state !== "queued") return;
  qc.setQueryData<ProjectView>(keys.view(pid), {
    ...view,
    stages: { ...view.stages, [job.stage]: { ...status, progress: job.progress } },
  });
}

/** Apply one event to the cache. Pure with respect to everything but `qc`. */
export function applyProjectEvent(qc: QueryClient, pid: string, event: ProjectEvent): void {
  switch (event.type) {
    case "decision":
      applyDecision(qc, pid, event.data);
      return;
    case "stage":
      applyStage(qc, pid, event.data);
      return;
    case "job":
      applyJob(qc, pid, event.data);
      return;
    case "resync":
      void qc.invalidateQueries({ queryKey: keys.project(pid) });
      return;
    case "ping":
      return;
  }
}

/** Parse one SSE frame into a typed event, or null if it is not one we know. */
export function parseEvent(type: string, raw: string): ProjectEvent | null {
  if (!(EVENT_TYPES as readonly string[]).includes(type)) return null;
  let data: unknown = {};
  if (raw) {
    try {
      data = JSON.parse(raw);
    } catch {
      return null;
    }
  }
  return { type, data } as ProjectEvent;
}

export type StreamState = "connecting" | "open" | "reconnecting";

/** Subscribe to a project's event stream for as long as the component is mounted. */
export function useProjectEvents(pid: string): StreamState {
  const qc = useQueryClient();
  const [state, setState] = useState<StreamState>("connecting");

  useEffect(() => {
    if (typeof EventSource === "undefined") return;
    const source = new EventSource(eventsUrl(pid));
    let opened = false;
    const listeners = EVENT_TYPES.map((type) => {
      const listener = (e: Event) => {
        const event = parseEvent(type, (e as MessageEvent<string>).data);
        if (event) applyProjectEvent(qc, pid, event);
      };
      source.addEventListener(type, listener);
      return [type, listener] as const;
    });
    source.onopen = () => {
      // After a reconnect, events may have been missed: catch up once.
      if (opened) void qc.invalidateQueries({ queryKey: keys.project(pid) });
      opened = true;
      setState("open");
    };
    source.onerror = () => setState("reconnecting");
    return () => {
      for (const [type, listener] of listeners) source.removeEventListener(type, listener);
      source.close();
    };
  }, [pid, qc]);

  return state;
}
