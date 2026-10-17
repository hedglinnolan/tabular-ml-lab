/**
 * The only module that talks to the network. One typed function per route.
 * A 409 is parsed into a typed `Refusal` and thrown as `RefusalError`; any other
 * non-2xx becomes `ApiError` carrying the status and whatever body came back.
 */
import type {
  ChecklistReport,
  ColumnSummary,
  ConfirmSweepInput,
  Decision,
  DecisionInput,
  ExportFile,
  FitLock as FitLockView,
  ForTheRecord,
  FsListing,
  Health,
  Histogram,
  JobView,
  Ledger,
  ProjectSummary,
  ProjectView,
  QuestLog,
  Refusal,
  StageResult,
  StageStatus,
  TableWindow,
  Triage,
} from "./schema";
import type { AnyStageArtifacts, AnyStageName, TeachingEntry } from "./m1-types";
import type { PreviewResult } from "./m1-stage-types";
import type {
  AddedFile,
  CodebookPreview,
  CodebookRequest,
  FamilyInfo,
  FitLock,
  JoinPreview,
  JoinPreviewRequest,
  MethodsSections,
  MethodsText,
  PlanExport,
  ReadingsCard,
} from "./m3-types";

export const API_BASE = "/api";

export class ApiError extends Error {
  readonly status: number;
  readonly body: unknown;
  constructor(status: number, message: string, body: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }
}

/** HTTP 409: the server declined, said why, and offered ways out. Not recorded. */
export class RefusalError extends Error {
  readonly refusal: Refusal;
  constructor(refusal: Refusal) {
    super(refusal.error.message);
    this.name = "RefusalError";
    this.refusal = refusal;
  }
}

export function isRefusalError(err: unknown): err is RefusalError {
  return err instanceof RefusalError;
}

function isObject(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

/** Narrow an unknown 409 body to a `Refusal`, or null if it is not one. */
export function parseRefusal(body: unknown): Refusal | null {
  if (!isObject(body) || !isObject(body.error)) return null;
  const { code, message, exits } = body.error;
  if (typeof code !== "string" || typeof message !== "string") return null;
  if (exits !== undefined && !Array.isArray(exits)) return null;
  const parsedExits: Refusal["error"]["exits"] = [];
  for (const exit of (exits as unknown[] | undefined) ?? []) {
    if (!isObject(exit) || typeof exit.label !== "string") return null;
    const decision = exit.decision;
    if (decision !== null && decision !== undefined) {
      if (!isObject(decision) || typeof decision.kind !== "string") return null;
    }
    parsedExits.push({
      label: exit.label,
      decision: (decision ?? null) as Decision | null,
    });
  }
  return { error: { code, message, exits: parsedExits } };
}

type Query = Record<string, string | number | undefined | null>;

function withQuery(path: string, query?: Query): string {
  if (!query) return path;
  const params = new URLSearchParams();
  for (const [k, v] of Object.entries(query)) {
    if (v !== undefined && v !== null && v !== "") params.set(k, String(v));
  }
  const qs = params.toString();
  return qs ? `${path}?${qs}` : path;
}

async function readBody(res: Response): Promise<unknown> {
  const text = await res.text();
  if (!text) return null;
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return text;
  }
}

function errorMessage(status: number, body: unknown): string {
  if (isObject(body)) {
    if (isObject(body.error) && typeof body.error.message === "string") return body.error.message;
    if (typeof body.detail === "string") return body.detail;
  }
  if (typeof body === "string" && body.length < 300) return body;
  return `The server answered ${status}.`;
}

/** The sign-in page's address, returning to `here` (a path on this server) afterward. */
export function signInUrl(here: string): string {
  return `/login?next=${encodeURIComponent(here)}`;
}

/**
 * Server mode: a 401 means the session ended (signed out elsewhere, expired, a new password).
 * The client announces it on `window`; main.tsx sends the page to sign in and back.
 */
export const UNAUTHENTICATED_EVENT = "turbotab:unauthenticated";

interface RequestOptions {
  method?: "GET" | "POST";
  query?: Query;
  json?: unknown;
  form?: FormData;
  signal?: AbortSignal;
  accept?: string;
}

/** What a non-2xx answer becomes: the sign-in event, a typed refusal, or an ApiError. */
async function fail(res: Response): Promise<never> {
  const body = await readBody(res);
  if (res.status === 401) window.dispatchEvent(new Event(UNAUTHENTICATED_EVENT));
  if (res.status === 409) {
    const refusal = parseRefusal(body);
    if (refusal) throw new RefusalError(refusal);
  }
  throw new ApiError(res.status, errorMessage(res.status, body), body);
}

async function send(path: string, opts: RequestOptions): Promise<Response> {
  const init: RequestInit = { method: opts.method ?? "GET", signal: opts.signal ?? null };
  const headers: Record<string, string> = { Accept: opts.accept ?? "application/json" };
  if (opts.json !== undefined) {
    headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(opts.json);
  } else if (opts.form) {
    init.body = opts.form;
  }
  init.headers = headers;
  return fetch(withQuery(`${API_BASE}${path}`, opts.query), init);
}

async function request<T>(path: string, opts: RequestOptions = {}): Promise<T> {
  const res = await send(path, opts);
  if (res.ok) return (await readBody(res)) as T;
  return fail(res);
}

/** The file name a `Content-Disposition: attachment; filename="..."` header gives, if any. */
function attachmentName(res: Response): string | null {
  const header = res.headers.get("Content-Disposition") ?? "";
  const found = /filename="?([^";]+)"?/i.exec(header);
  return found ? found[1]! : null;
}

async function requestFile(path: string, fallbackName: string, opts: RequestOptions = {}): Promise<ExportFile> {
  const res = await send(path, { ...opts, accept: "*/*" });
  if (!res.ok) return fail(res);
  return { blob: await res.blob(), filename: attachmentName(res) ?? fallbackName };
}

const enc = encodeURIComponent;

export const api = {
  health: (signal?: AbortSignal) => request<Health>("/health", { signal }),

  listProjects: (signal?: AbortSignal) => request<ProjectSummary[]>("/projects", { signal }),

  openPath: (path: string) =>
    request<ProjectSummary>("/projects", { method: "POST", json: { path } }),

  upload: (file: File) => {
    const form = new FormData();
    form.append("file", file, file.name);
    return request<ProjectSummary>("/projects/upload", { method: "POST", form });
  },

  project: (pid: string, signal?: AbortSignal) =>
    request<ProjectView>(`/projects/${enc(pid)}`, { signal }),

  /**
   * Record a decision. `decideNow` answers a later question ahead of the Router, where the server
   * allows it (its card is computed and the earlier answers it reads are in); otherwise it is
   * refused (409) with what the question is waiting for.
   */
  decide: (pid: string, decision: Decision | DecisionInput, opts?: { decideNow?: boolean }) =>
    request<ProjectView>(`/projects/${enc(pid)}/decisions`, {
      method: "POST",
      json: decision,
      query: { decide_now: opts?.decideNow ? "true" : undefined },
    }),

  /** Confirm a gate's sweep: every default ("defaults") or noticing disposition ("noticings") as
   * listed, with any line the person changed in `lines` (POST /decisions, kind confirm_sweep). */
  confirmSweep: (pid: string, sweep: Omit<ConfirmSweepInput, "kind">) =>
    request<ProjectView>(`/projects/${enc(pid)}/decisions`, {
      method: "POST",
      json: { kind: "confirm_sweep", ...sweep } satisfies ConfirmSweepInput,
    }),

  stage: <S extends AnyStageName>(pid: string, stage: S, signal?: AbortSignal) =>
    request<StageResult<AnyStageArtifacts[S]>>(`/projects/${enc(pid)}/stages/${enc(stage)}`, {
      signal,
    }),

  /** Compute a stage again after it failed or was cancelled (and whatever it waits on). */
  runStage: (pid: string, stage: AnyStageName) =>
    request<StageStatus>(`/projects/${enc(pid)}/stages/${enc(stage)}/run`, { method: "POST" }),

  table: (
    pid: string,
    window: { offset: number; limit: number; columns?: string[] },
    signal?: AbortSignal,
  ) =>
    request<TableWindow>(`/projects/${enc(pid)}/table`, {
      query: {
        offset: window.offset,
        limit: window.limit,
        columns: window.columns?.join(","),
      },
      signal,
    }),

  columns: (pid: string, signal?: AbortSignal) =>
    request<ColumnSummary[]>(`/projects/${enc(pid)}/columns`, { signal }),

  histogram: (pid: string, column: string, bins?: number, signal?: AbortSignal) =>
    request<Histogram>(`/projects/${enc(pid)}/columns/${enc(column)}/histogram`, {
      query: { bins },
      signal,
    }),

  job: (pid: string, jid: string, signal?: AbortSignal) =>
    request<JobView>(`/projects/${enc(pid)}/jobs/${enc(jid)}`, { signal }),

  cancelJob: (pid: string, jid: string) =>
    request<JobView>(`/projects/${enc(pid)}/jobs/${enc(jid)}/cancel`, { method: "POST" }),

  listDir: (path?: string, signal?: AbortSignal) =>
    request<FsListing>("/fs/list", { query: { path }, signal }),

  /** The words every interview question carries, in asking order (M1_CONTRACT §5). */
  teaching: (signal?: AbortSignal) => request<TeachingEntry[]>("/teaching", { signal }),

  /** What recording `decision` would change, on the user's own data. Nothing is recorded. */
  preview: (pid: string, decision: Decision, signal?: AbortSignal) =>
    request<PreviewResult>(`/projects/${enc(pid)}/preview`, {
      method: "POST",
      json: decision,
      signal,
    }),

  /** The views that show why a finding was raised (M1_CONTRACT §12.3). */
  findingEvidence: (pid: string, findingId: string, signal?: AbortSignal) =>
    request<PreviewResult>(`/projects/${enc(pid)}/findings/${enc(findingId)}/evidence`, {
      signal,
    }),

  // ── M3: the readings ledger, the record's outputs, assembly ──────────────

  /** The readings the values settled with no question asked ("read from your data", §14.3). */
  readings: (pid: string, signal?: AbortSignal) =>
    request<ReadingsCard>(`/projects/${enc(pid)}/readings`, { signal }),

  /** The methods text built from the decision log (audit WP16). */
  methods: (pid: string, signal?: AbortSignal) =>
    request<MethodsText>(`/projects/${enc(pid)}/methods`, { signal }),

  /** P8: the methods in the export bundle's own sections and words, each line with its record id
   * and the lines counted (the manuscript rail; TRIPOD+AI's sections under Predict). */
  methodsSections: (pid: string, signal?: AbortSignal) =>
    request<MethodsSections>(`/projects/${enc(pid)}/methods/sections`, { signal }),

  /** The analysis plan for registration: canonical JSON with its SHA-256 (MODELING_SEQUENCE §1). */
  plan: (pid: string, signal?: AbortSignal) =>
    request<PlanExport>(`/projects/${enc(pid)}/plan`, { signal }),

  /** Fit, pressed (SIZING P0.8): a job command. Under Estimate it locks the plan; until it is
   * pressed no estimate (and, under Predict, no score) is served. */
  pressFit: (pid: string) => request<FitLock>(`/projects/${enc(pid)}/fit`, { method: "POST" }),

  // ── the quest log, the record's outputs, Fit, the export (W1-A) ──────────

  /** The quest log: seven stages, their lines, progress, and the plan's lock (SIZING P0.4). */
  quest: (pid: string, signal?: AbortSignal) =>
    request<QuestLog>(`/projects/${enc(pid)}/quest`, { signal }),

  /** Each stage's For the record lines: collapsed in the quest log, never counted. */
  record: (pid: string, signal?: AbortSignal) =>
    request<ForTheRecord>(`/projects/${enc(pid)}/record`, { signal }),

  /** The triage of the open noticings at the gate, with each one's recommended disposition. */
  triage: (pid: string, signal?: AbortSignal) =>
    request<Triage>(`/projects/${enc(pid)}/triage`, { signal }),

  /** The materiality ledger: how far each noticing's alternative moves the numbers. */
  materiality: (pid: string, signal?: AbortSignal) =>
    request<Ledger>(`/projects/${enc(pid)}/materiality`, { signal }),

  /** The reporting checklist of the declared purpose, and what the export still waits for. */
  checklist: (pid: string, signal?: AbortSignal) =>
    request<ChecklistReport>(`/projects/${enc(pid)}/checklist`, { signal }),

  /** Cancel, where Fit was: the estimates' work still running stops; before any estimate is
   * served the press (and, under Estimate, the plan's lock) is withdrawn. Returns the lock. */
  cancelFit: (pid: string) =>
    request<FitLockView>(`/projects/${enc(pid)}/fit/cancel`, { method: "POST" }),

  /** The manuscript bundle (a zip). Refused (409) with everything missing while it is not ready. */
  exportBundle: (pid: string, signal?: AbortSignal) =>
    requestFile(`/projects/${enc(pid)}/export`, "turbotab-export.zip", { signal }),

  /** Every model family the engine can fit, with what it needs and assumes. */
  models: (signal?: AbortSignal) => request<FamilyInfo[]>("/models", { signal }),

  /** The files added to the project to join to its table. */
  files: (pid: string, signal?: AbortSignal) =>
    request<AddedFile[]>(`/projects/${enc(pid)}/files`, { signal }),

  /** Add a file on this machine to join (local mode only). */
  addFile: (pid: string, path: string) =>
    request<AddedFile>(`/projects/${enc(pid)}/files`, { method: "POST", json: { path } }),

  /** Upload a file to join to the project's table. */
  uploadFile: (pid: string, file: File) => {
    const form = new FormData();
    form.append("file", file, file.name);
    return request<AddedFile>(`/projects/${enc(pid)}/files/upload`, { method: "POST", form });
  },

  /** The row counts a join would give, before `join_files` records it. */
  joinPreview: (pid: string, body: JoinPreviewRequest) =>
    request<JoinPreview>(`/projects/${enc(pid)}/join-preview`, { method: "POST", json: body }),

  /** Read a codebook by its path (local mode), or the table's own XPT labels (`labels`). */
  codebook: (pid: string, body: CodebookRequest) =>
    request<CodebookPreview>(`/projects/${enc(pid)}/codebooks`, { method: "POST", json: body }),

  /** Upload a codebook: a variable table, an NHANES codebook page or an XPT file. */
  uploadCodebook: (pid: string, file: File) => {
    const form = new FormData();
    form.append("file", file, file.name);
    return request<CodebookPreview>(`/projects/${enc(pid)}/codebooks/upload`, {
      method: "POST",
      form,
    });
  },
};

/** The export's address, for a plain link; `api.exportBundle` reads refusals, which a link cannot. */
export function exportUrl(pid: string): string {
  return `${API_BASE}/projects/${enc(pid)}/export`;
}

/** The SSE endpoint's URL. The EventSource itself is opened in events.ts. */
export function eventsUrl(pid: string): string {
  return `${API_BASE}/projects/${enc(pid)}/events`;
}
