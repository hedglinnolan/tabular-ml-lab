/**
 * The only module that talks to the network. One typed function per route.
 * A 409 is parsed into a typed `Refusal` and thrown as `RefusalError`; any other
 * non-2xx becomes `ApiError` carrying the status and whatever body came back.
 */
import type {
  ColumnSummary,
  Decision,
  FsListing,
  Health,
  Histogram,
  JobView,
  ProjectSummary,
  ProjectView,
  Refusal,
  StageArtifacts,
  StageName,
  StageResult,
  StageStatus,
  TableWindow,
} from "./schema";

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

interface RequestOptions {
  method?: "GET" | "POST";
  query?: Query;
  json?: unknown;
  form?: FormData;
  signal?: AbortSignal;
}

async function request<T>(path: string, opts: RequestOptions = {}): Promise<T> {
  const init: RequestInit = { method: opts.method ?? "GET", signal: opts.signal ?? null };
  const headers: Record<string, string> = { Accept: "application/json" };
  if (opts.json !== undefined) {
    headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(opts.json);
  } else if (opts.form) {
    init.body = opts.form;
  }
  init.headers = headers;
  const res = await fetch(withQuery(`${API_BASE}${path}`, opts.query), init);
  if (res.ok) return (await readBody(res)) as T;
  const body = await readBody(res);
  if (res.status === 409) {
    const refusal = parseRefusal(body);
    if (refusal) throw new RefusalError(refusal);
  }
  throw new ApiError(res.status, errorMessage(res.status, body), body);
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

  decide: (pid: string, decision: Decision) =>
    request<ProjectView>(`/projects/${enc(pid)}/decisions`, { method: "POST", json: decision }),

  stage: <S extends StageName>(pid: string, stage: S, signal?: AbortSignal) =>
    request<StageResult<StageArtifacts[S]>>(`/projects/${enc(pid)}/stages/${enc(stage)}`, {
      signal,
    }),

  /** Compute a stage again after it failed or was cancelled (and whatever it waits on). */
  runStage: (pid: string, stage: StageName) =>
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
};

/** The SSE endpoint's URL. The EventSource itself is opened in events.ts. */
export function eventsUrl(pid: string): string {
  return `${API_BASE}/projects/${enc(pid)}/events`;
}
