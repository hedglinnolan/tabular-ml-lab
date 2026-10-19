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
  ConfirmSweepInput,
  Decision,
  FitLock,
  JobView,
  ProjectView,
  QuestLog,
  StageResult,
  StageStatus,
} from "./schema";
import type { AnyStageArtifacts, AnyStageName } from "./m1-types";
import type { CodebookRequest, JoinPreviewRequest } from "./m3-types";

export const keys = {
  health: () => ["health"] as const,
  /** The words for a goal (`purpose`), else the words with no goal named. */
  teaching: (purpose?: string | null): readonly string[] => (purpose ? ["teaching", purpose] : ["teaching"]),
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
  // M3: each is read from the decision log, so a decision event invalidates it (events.ts).
  readings: (pid: string) => [pid, "readings"] as const,
  methods: (pid: string) => [pid, "methods"] as const,
  // P8: the methods in the bundle's sections, read from the log and the fresh stages as the bundle
  // reads them, so a decision event invalidates it with the other log-derived keys, and a stage
  // event changing a stage's state or fresh key invalidates it too (events.ts).
  methodsSections: (pid: string) => [pid, "methods", "sections"] as const,
  plan: (pid: string) => [pid, "plan"] as const,
  files: (pid: string) => [pid, "files"] as const,
  models: () => ["models"] as const,
  // W1-A: the quest log and the record's outputs. All are derived from the decision log and the
  // stage statuses, so a decision event (and a stage changing state) invalidates them (events.ts);
  // the quest log also when a read of the view brings a stage's new state first (useProjectView).
  quest: (pid: string) => [pid, "quest"] as const,
  record: (pid: string) => [pid, "record"] as const,
  triage: (pid: string) => [pid, "triage"] as const,
  materiality: (pid: string) => [pid, "materiality"] as const,
  checklist: (pid: string) => [pid, "checklist"] as const,
};

/** The reads that follow the decision log: stale after any decision. */
export function logDerivedKeys(pid: string) {
  return [
    keys.quest(pid),
    keys.record(pid),
    keys.triage(pid),
    keys.materiality(pid),
    keys.checklist(pid),
    keys.methodsSections(pid),
  ];
}

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
  const qc = useQueryClient();
  return useQuery({
    queryKey: keys.view(pid),
    // A refetch can answer with statuses older than the SSE events that arrived while it
    // was in flight (the server builds the view before the stage finishes). Merge newest-wins
    // so a late response never puts a finished stage back to "running".
    queryFn: async ({ signal }) => {
      const next = await api.project(pid, signal);
      // Read the cache after the response, not before: SSE may have patched it meanwhile.
      const prev = qc.getQueryData<ProjectView>(keys.view(pid));
      const merged = mergeView(prev, next);
      // The quest log reads each stage's state (a result out of date drops its stage back until it
      // is computed again), and a stage event reads it again only when it changes the state the view
      // holds (events.ts). This read can bring that state first: the engine reports a stage fresh
      // once its result is written, before the run's end publishes the event, and a response can
      // beat an event. The event then finds nothing new, so this read reads the quest log again.
      if (prev && stagesMoved(prev, merged)) void qc.invalidateQueries({ queryKey: keys.quest(pid), exact: true });
      return merged;
    },
  });
}

/** The words every question carries (GET /api/teaching): static, fetched once per goal. With the
 *  project's `purpose` the server words the cards it words by goal for that goal (under prediction
 *  the roles card's predictors, W1-R finding 3c); with none, as they stand with no goal named. */
export function useTeaching(purpose: string | null = null) {
  return useQuery({ queryKey: keys.teaching(purpose), queryFn: ({ signal }) => api.teaching(purpose, signal) });
}

export function useStageResult<S extends AnyStageName>(pid: string, stage: S, enabled = true) {
  return useQuery<StageResult<AnyStageArtifacts[S]>>({
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
  // The folded state belongs to the newest decision each side has seen. A response
  // built before a decision the cache already holds (from the POST that made it, or a
  // response that came back first) must not roll the state back.
  const behind = lastSeq(next) < lastSeq(prev);
  // The size comes from the ingest artifact; keep the side whose ingest status won.
  const summary = stages.ingest === prev.stages.ingest ? prev.summary : next.summary;
  return { ...next, summary, state: behind ? prev.state : next.state, stages, decisions };
}

/** Whether a stage really changed (its state, a stop, the hold), as against a progress tick. */
export function changedState(a: StageStatus | undefined, b: StageStatus): boolean {
  return a?.status !== b.status || a?.cancelled !== b.cancelled || a?.held !== b.held;
}

/** Whether `next` holds a stage in a state `prev` did not (a progress tick is no change). */
function stagesMoved(prev: ProjectView, next: ProjectView): boolean {
  return Object.entries(next.stages).some(([name, s]) => !!s && changedState(prev.stages[name], s));
}

function lastSeq(view: ProjectView): number {
  return view.decisions.reduce((m, d) => Math.max(m, d.seq), 0);
}

function recorded(qc: QueryClient, pid: string, view: ProjectView): void {
  qc.setQueryData<ProjectView>(keys.view(pid), (prev) => mergeView(prev, view));
  // The SSE decision event does this too; a response that beats it must not leave them old.
  for (const key of logDerivedKeys(pid)) void qc.invalidateQueries({ queryKey: key, exact: true });
}

export function useDecide(pid: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (decision: Decision) => api.decide(pid, decision),
    onSuccess: (view) => recorded(qc, pid, view),
  });
}

/** Answer a later question ahead of the Router (`decide_now`): refused, with what it waits for,
 * where its card is not computed or the earlier answers it reads are not in. */
export function useDecideNow(pid: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (decision: Decision) => api.decide(pid, decision, { decideNow: true }),
    onSuccess: (view) => recorded(qc, pid, view),
  });
}

/** Confirm a gate's sweep (kind confirm_sweep): the defaults or the noticing dispositions as
 * listed, with any line the person changed in `lines`. */
export function useConfirmSweep(pid: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (sweep: Omit<ConfirmSweepInput, "kind">) => api.confirmSweep(pid, sweep),
    onSuccess: (view) => recorded(qc, pid, view),
  });
}

/** Try a failed stage again, or recompute one whose work was cancelled. */
export function useRunStage(pid: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (stage: AnyStageName) => api.runStage(pid, stage),
    onSuccess: (status: StageStatus) => {
      qc.setQueryData<ProjectView>(keys.view(pid), (prev) =>
        prev && stamp(status) >= stamp(prev.stages[status.stage])
          ? { ...prev, stages: { ...prev.stages, [status.stage]: status } }
          : prev,
      );
    },
  });
}

/** Fit, pressed: every stage result is fetched again (the estimates are served now), and the
 * view, which carries the lock under Estimate. */
export function usePressFit(pid: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api.pressFit(pid),
    onSuccess: (lock) => {
      keepLock(qc, pid, lock);
      void qc.invalidateQueries({ queryKey: [pid, "stage"] });
      void qc.invalidateQueries({ queryKey: keys.view(pid) });
    },
  });
}

/** The lock a Fit or Cancel returned, as the quest log shows it, until the log is read again. */
function keepLock(qc: QueryClient, pid: string, lock: FitLock): void {
  qc.setQueryData<QuestLog>(keys.quest(pid), (prev) => (prev ? { ...prev, fit: lock } : prev));
  void qc.invalidateQueries({ queryKey: keys.quest(pid), exact: true });
}

/** Cancel, where Fit was: running estimates stop; before any estimate is served the press (and,
 * under Estimate, the plan's lock) is withdrawn, so Fit is asked for again. */
export function useCancelFit(pid: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api.cancelFit(pid),
    onSuccess: (lock) => {
      keepLock(qc, pid, lock);
      void qc.invalidateQueries({ queryKey: [pid, "stage"] });
      void qc.invalidateQueries({ queryKey: keys.view(pid) });
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

// ── M3: the readings ledger, the record's outputs, assembly ─────────────────

/** "Read from your data": the readings the values settled, each with the answers that change it. */
export function useReadings(pid: string, enabled = true) {
  return useQuery({
    queryKey: keys.readings(pid),
    queryFn: ({ signal }) => api.readings(pid, signal),
    enabled,
  });
}

/** The methods text the decision log builds. */
export function useMethods(pid: string, enabled = true) {
  return useQuery({
    queryKey: keys.methods(pid),
    queryFn: ({ signal }) => api.methods(pid, signal),
    enabled,
  });
}

/** P8: the methods in the export bundle's own sections and words, for the manuscript rail. */
export function useMethodsSections(pid: string, enabled = true) {
  return useQuery({
    queryKey: keys.methodsSections(pid),
    queryFn: ({ signal }) => api.methodsSections(pid, signal),
    enabled,
  });
}

/** The analysis plan for registration, with its hash. */
export function usePlan(pid: string, enabled = true) {
  return useQuery({ queryKey: keys.plan(pid), queryFn: ({ signal }) => api.plan(pid, signal), enabled });
}

/** Every model family the engine can fit (static). */
export function useModels() {
  return useQuery({ queryKey: keys.models(), queryFn: ({ signal }) => api.models(signal) });
}

/** The files added to the project to join to its table. */
export function useFiles(pid: string, enabled = true) {
  return useQuery({ queryKey: keys.files(pid), queryFn: ({ signal }) => api.files(pid, signal), enabled });
}

/** Add a file to join: by its path on this machine, or uploaded. */
export function useAddFile(pid: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (source: { path: string } | { file: File }) =>
      "path" in source ? api.addFile(pid, source.path) : api.uploadFile(pid, source.file),
    onSuccess: () => qc.invalidateQueries({ queryKey: keys.files(pid), exact: true }),
  });
}

/** The row counts a join would give; nothing is recorded (`join_files` records it). */
export function useJoinPreview(pid: string) {
  return useMutation({ mutationFn: (body: JoinPreviewRequest) => api.joinPreview(pid, body) });
}

/** Read a codebook (by path, the table's own XPT labels, or uploaded) and say what importing it
 *  would settle; nothing is recorded (`import_codebook` records it). */
export function useCodebook(pid: string) {
  return useMutation({
    mutationFn: (source: CodebookRequest | { file: File }) =>
      "file" in source ? api.uploadCodebook(pid, source.file) : api.codebook(pid, source),
  });
}

// ── W1-A: the quest log, the record's outputs, the export ───────────────────

/** The quest log (seven stages, their lines, progress) and the plan's lock under `fit`. */
export function useQuest(pid: string, enabled = true) {
  return useQuery({ queryKey: keys.quest(pid), queryFn: ({ signal }) => api.quest(pid, signal), enabled });
}

/** Each stage's For the record lines. */
export function useRecord(pid: string, enabled = true) {
  return useQuery({ queryKey: keys.record(pid), queryFn: ({ signal }) => api.record(pid, signal), enabled });
}

/** The triage of the open noticings at the gate. */
export function useTriage(pid: string, enabled = true) {
  return useQuery({ queryKey: keys.triage(pid), queryFn: ({ signal }) => api.triage(pid, signal), enabled });
}

/** The materiality ledger. */
export function useMateriality(pid: string, enabled = true) {
  return useQuery({
    queryKey: keys.materiality(pid),
    queryFn: ({ signal }) => api.materiality(pid, signal),
    enabled,
  });
}

/** The reporting checklist and what the export still waits for. */
export function useChecklist(pid: string, enabled = true) {
  return useQuery({
    queryKey: keys.checklist(pid),
    queryFn: ({ signal }) => api.checklist(pid, signal),
    enabled,
  });
}

/** Build the manuscript bundle on request. A mutation, not a query: it is a download the person
 * asks for, and a refusal (everything the export still waits for) comes back as a RefusalError. */
export function useExport(pid: string) {
  return useMutation({ mutationFn: () => api.exportBundle(pid) });
}
