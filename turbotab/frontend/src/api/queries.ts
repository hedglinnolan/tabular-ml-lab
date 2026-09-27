/**
 * TanStack Query hooks. Every project-scoped key starts with the project id, so
 * a `resync` can invalidate `[pid]` and refetch everything that project shows.
 * Nothing here polls: freshness comes from SSE (see events.ts).
 */
import {
  QueryClient,
  useMutation,
  useQueries,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import { api } from "./client";
import type {
  Decision,
  JobView,
  ProjectView,
  StageArtifacts,
  StageName,
  StageResult,
  StageStatus,
} from "./schema";

export const keys = {
  health: () => ["health"] as const,
  projects: () => ["projects"] as const,
  fs: (path: string | null) => ["fs", path ?? ""] as const,
  project: (pid: string) => [pid] as const,
  view: (pid: string) => [pid, "view"] as const,
  stage: (pid: string, stage: string) => [pid, "stage", stage] as const,
  columns: (pid: string) => [pid, "columns"] as const,
  tableAll: (pid: string) => [pid, "table"] as const,
  table: (pid: string, offset: number, limit: number, columns: string) =>
    [pid, "table", offset, limit, columns] as const,
  histogram: (pid: string, column: string, bins: number | undefined) =>
    [pid, "histogram", column, bins ?? null] as const,
  job: (pid: string, jid: string) => [pid, "job", jid] as const,
};

export function makeQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: {
        // Server pushes changes over SSE; nothing refetches on a timer or on focus.
        staleTime: Infinity,
        refetchOnWindowFocus: false,
        refetchOnReconnect: false,
        retry: 1,
      },
    },
  });
}

export function useHealth() {
  return useQuery({ queryKey: keys.health(), queryFn: ({ signal }) => api.health(signal) });
}

export function useProjects() {
  return useQuery({
    queryKey: keys.projects(),
    queryFn: ({ signal }) => api.listProjects(signal),
    staleTime: 0,
  });
}

export function useFsListing(path: string | null, enabled = true) {
  return useQuery({
    queryKey: keys.fs(path),
    queryFn: ({ signal }) => api.listDir(path ?? undefined, signal),
    enabled,
    staleTime: 0,
  });
}

export function useProjectView(pid: string) {
  return useQuery({ queryKey: keys.view(pid), queryFn: ({ signal }) => api.project(pid, signal) });
}

export function useStageResult<S extends StageName>(pid: string, stage: S, enabled = true) {
  return useQuery<StageResult<StageArtifacts[S]>>({
    queryKey: keys.stage(pid, stage),
    queryFn: ({ signal }) => api.stage(pid, stage, signal),
    enabled,
  });
}

export function useColumnSummaries(pid: string, enabled = true) {
  return useQuery({
    queryKey: keys.columns(pid),
    queryFn: ({ signal }) => api.columns(pid, signal),
    enabled,
  });
}

export interface TablePage {
  offset: number;
  limit: number;
  columns: string[];
}

/** Several row-and-column windows of the working table, fetched in parallel. */
export function useTablePages(pid: string, pages: TablePage[], enabled = true) {
  return useQueries({
    queries: pages.map((p) => ({
      queryKey: keys.table(pid, p.offset, p.limit, p.columns.join(",")),
      queryFn: ({ signal }: { signal: AbortSignal }) =>
        api.table(pid, { offset: p.offset, limit: p.limit, columns: p.columns }, signal),
      enabled,
    })),
  });
}

export function useHistogram(pid: string, column: string | null, bins?: number) {
  return useQuery({
    queryKey: keys.histogram(pid, column ?? "", bins),
    queryFn: ({ signal }) => api.histogram(pid, column ?? "", bins, signal),
    enabled: column !== null,
  });
}

export function useJobs(pid: string, jobIds: string[]) {
  return useQueries({
    queries: jobIds.map((jid) => ({
      queryKey: keys.job(pid, jid),
      queryFn: ({ signal }: { signal: AbortSignal }) => api.job(pid, jid, signal),
    })),
  });
}

function stamp(s: StageStatus | undefined): number {
  if (!s?.updated_at) return -Infinity;
  const t = Date.parse(s.updated_at);
  return Number.isNaN(t) ? -Infinity : t;
}

/**
 * Merge a freshly received view into the cached one without letting an older
 * response overwrite newer stage statuses that already arrived over SSE.
 */
export function mergeView(prev: ProjectView | undefined, next: ProjectView): ProjectView {
  if (!prev) return next;
  const stages: ProjectView["stages"] = { ...next.stages };
  for (const [name, old] of Object.entries(prev.stages)) {
    const incoming = next.stages[name];
    if (incoming && stamp(old) > stamp(incoming)) stages[name] = old;
  }
  const byId = new Map(prev.decisions.map((d) => [d.id, d]));
  for (const d of next.decisions) byId.set(d.id, d);
  const decisions = [...byId.values()].sort((a, b) => a.seq - b.seq);
  return { ...next, stages, decisions };
}

export function useDecide(pid: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (decision: Decision) => api.decide(pid, decision),
    onSuccess: (view) => {
      qc.setQueryData<ProjectView>(keys.view(pid), (prev) => mergeView(prev, view));
    },
  });
}

export function useCancelJob(pid: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (jid: string) => api.cancelJob(pid, jid),
    onSuccess: (job: JobView) => {
      qc.setQueryData(keys.job(pid, job.job_id), job);
    },
  });
}

export function useOpenPath() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (path: string) => api.openPath(path),
    onSuccess: () => qc.invalidateQueries({ queryKey: keys.projects() }),
  });
}

export function useUpload() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (file: File) => api.upload(file),
    onSuccess: () => qc.invalidateQueries({ queryKey: keys.projects() }),
  });
}
