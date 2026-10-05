/**
 * The M3 mock API (dev:mock only): real reference journeys replayed, every shape the real server
 * answered (src/mocks/fixtures/m3-*.json, written by docs/turbotab-next/m3/capture_fixtures.py).
 *
 * A journey is opened as a project: `/p/m3~<journey>` from its first step, or
 * `/p/m3~<journey>~<n>` from snapshot n (the lab at /lab/m3 lists them). Each project replays:
 *
 *   GET  the view        the snapshot's ProjectView (its decisions from the log, stamped now)
 *   GET  a stage         the artifact at the snapshot's key (patched forward from its first)
 *   POST a decision      a decision the capture posted from here moves to the snapshot it led to,
 *                        or answers the refusal the server gave (with its exits); anything else
 *                        is refused, its exits the answers the capture recorded from here
 *   GET  readings, methods, plan, columns, table, files: as captured at the journey's end
 *
 * Every project, the mock server's own included, also gets the endpoints no other mock answers:
 * the readings card, the methods text, the plan, the model shelf, the files to join, the join
 * preview and the codebooks, each the real server's shape from these captures.
 */
import { http, HttpResponse, type HttpHandler } from "msw";
import type { DecisionRecord, ProjectView, StageStatus } from "../api/schema";

// ── the fixtures, loaded on demand ───────────────────────────────────────────

type Json = Record<string, unknown>;

interface Snapshot {
  open: string | null;
  first: string | null;
  after: string | null;
  view?: CompactView;
  patch?: Json;
}

interface CompactView {
  summary: ProjectView["summary"];
  state: ProjectView["state"];
  decisions: string[];
  stages: Record<string, Omit<StageStatus, "updated_at">>;
  interview: ProjectView["interview"];
}

interface CapturedRecord {
  from: number;
  decision: Json;
  status: number;
  to?: number;
  body?: Json;
}

export interface M3Fixture {
  meta: { journey: string; label: string; source: string; trim: string; trimmed: Json; notes: string[] };
  start: number;
  snapshots: Snapshot[];
  artifacts: Record<string, { keys: string[]; base: unknown; patches: Json[] }>;
  records: CapturedRecord[];
  log: Record<string, DecisionRecord>;
  endpoints: Record<string, { status: number; body: unknown }>;
}

const LOADERS = import.meta.glob<M3Fixture>("./fixtures/m3-*.json", { import: "default" });
const SHARED = "./fixtures/m3-shared.json";

/** The journeys captured, by name (`nhanes-inference`, `causal`, …). */
export const M3_JOURNEYS: string[] = Object.keys(LOADERS)
  .filter((path) => path !== SHARED)
  .map((path) => /m3-(.+)\.json$/.exec(path)![1]!)
  .sort();

const loaded = new Map<string, Promise<M3Fixture>>();
export function loadJourney(name: string): Promise<M3Fixture> {
  const path = `./fixtures/m3-${name}.json`;
  const load = LOADERS[path];
  if (!load) return Promise.reject(new Error(`no journey ${name}`));
  if (!loaded.has(name)) loaded.set(name, load());
  return loaded.get(name)!;
}

// ── packing (the capture script's `pack`) ────────────────────────────────────

const isObject = (v: unknown): v is Json => typeof v === "object" && v !== null && !Array.isArray(v);
const isDelete = (v: unknown) => isObject(v) && v.$del === 1 && Object.keys(v).length === 1;

/** Apply one patch: objects merge key by key, anything else is replaced, `{"$del": 1}` removes. */
export function applyPatch<T>(base: T, patch: unknown): T {
  if (!isObject(base) || !isObject(patch)) return patch as T;
  const out: Json = { ...base };
  for (const [k, v] of Object.entries(patch)) {
    if (isDelete(v)) delete out[k];
    else if (isObject(v) && isObject(out[k])) out[k] = applyPatch(out[k], v);
    else out[k] = v;
  }
  return out as T;
}

/** Every snapshot's view, in order. */
export function viewsOf(f: M3Fixture): CompactView[] {
  const out: CompactView[] = [];
  let view = f.snapshots[0]!.view!;
  out.push(view);
  for (const s of f.snapshots.slice(1)) {
    view = s.view ?? applyPatch(view, s.patch ?? {});
    out.push(view);
  }
  return out;
}

/** A stage's artifact at a key, if the capture holds that version. */
export function artifactAt(f: M3Fixture, stage: string, key: string | null): unknown {
  const a = f.artifacts[stage];
  if (!a) return null;
  const i = key === null ? -1 : a.keys.indexOf(key);
  const upTo = i === -1 ? a.keys.length - 1 : i;
  let v = a.base;
  for (const p of a.patches.slice(0, upTo)) v = applyPatch(v, p);
  return v;
}

/** The full ProjectView of a compact one: decisions from the log, every status stamped `at`. */
export function expandView(f: M3Fixture, v: CompactView, pid: string, at: string): ProjectView {
  const stages: ProjectView["stages"] = {};
  for (const [name, s] of Object.entries(v.stages)) stages[name] = { ...s, updated_at: at } as StageStatus;
  return {
    summary: { ...v.summary, id: pid },
    state: v.state,
    decisions: v.decisions.map((id) => f.log[id]!).filter(Boolean),
    stages,
    interview: v.interview,
  };
}

// ── matching a posted decision to the capture ────────────────────────────────

/** A decision's identity, independent of key order and of fields left at their defaults. */
export function looseKey(d: unknown): string {
  const strip = (v: unknown): unknown => {
    if (Array.isArray(v)) return v.map(strip);
    if (!isObject(v)) return typeof v === "number" ? Number(v.toPrecision(10)) : v;
    const out: Json = {};
    for (const k of Object.keys(v).sort()) {
      const x = strip(v[k]);
      const empty =
        x === null ||
        x === false ||
        x === undefined ||
        (Array.isArray(x) && x.length === 0) ||
        (isObject(x) && Object.keys(x).length === 0);
      if (!empty) out[k] = x;
    }
    return out;
  };
  return JSON.stringify(strip(d));
}

const refusal = (code: string, message: string, exits: { label: string; decision: unknown }[] = []) => ({
  error: { code, message, exits },
});

export class Replay {
  readonly views: CompactView[];
  current: number;
  constructor(
    readonly f: M3Fixture,
    readonly pid: string,
    start: number,
  ) {
    this.views = viewsOf(f);
    this.current = Math.max(0, Math.min(start, this.views.length - 1));
  }
  view(): ProjectView {
    return expandView(this.f, this.views[this.current]!, this.pid, new Date().toISOString());
  }
  stage(name: string) {
    const v = this.views[this.current]!;
    const status = v.stages[name];
    if (!status) return null;
    const none = status.status === "idle" || status.status === "blocked";
    return {
      stage: name,
      key: status.key,
      fresh: status.status === "fresh",
      status: status.status,
      artifact: none ? null : artifactAt(this.f, name, status.key),
    };
  }
  /** What the capture answered to `d` from here: a move, or the refusal it gave. */
  decide(d: unknown): { status: number; body: unknown } {
    if (isObject(d) && d.kind === "revert") {
      return {
        status: 409,
        body: refusal(
          "not_captured",
          "The replay holds only the answers its capture recorded, so it cannot take one back; the real server records a revert. Open the journey at an earlier step from /lab/m3 instead.",
        ),
      };
    }
    const key = looseKey(d);
    const same = this.f.records.filter((r) => looseKey(r.decision) === key);
    const hit = same.find((r) => r.from === this.current) ?? same.find((r) => r.from > this.current);
    if (hit?.status === 200 && hit.to !== undefined) {
      this.current = hit.to;
      return { status: 200, body: this.view() };
    }
    if (hit) return { status: hit.status, body: hit.body };
    const here = this.f.records.filter((r) => r.from === this.current && r.status === 200);
    return {
      status: 409,
      body: refusal(
        "not_captured",
        "The replay answers only what its capture posted from this step.",
        here.map((r) => ({ label: `Record what the capture recorded (${String(r.decision.kind)})`, decision: r.decision })),
      ),
    };
  }
}

/** `m3~<journey>` or `m3~<journey>~<n>`. */
export function parseM3(pid: string): { journey: string; start: number } | null {
  const m = /^m3~([a-z0-9-]+)(?:~(\d+))?$/.exec(pid);
  return m ? { journey: m[1]!, start: Number(m[2] ?? 0) } : null;
}

export const m3Pid = (journey: string, step = 0) => (step ? `m3~${journey}~${step}` : `m3~${journey}`);

// ── the handlers ─────────────────────────────────────────────────────────────

const replays = new Map<string, Promise<Replay | null>>();
function replayOf(pid: string): Promise<Replay | null> | null {
  const parsed = parseM3(pid);
  if (!parsed || !M3_JOURNEYS.includes(parsed.journey)) return null;
  if (!replays.has(pid))
    replays.set(
      pid,
      loadJourney(parsed.journey).then(
        (f) => new Replay(f, pid, parsed.start),
        () => null,
      ),
    );
  return replays.get(pid)!;
}

const notFound = (what: string) => HttpResponse.json(refusal("not_found", what), { status: 404 });

/** The endpoints every project gets from a capture: `source` journey, `name` endpoint. */
async function captured(source: string, name: string) {
  const f = await loadJourney(source);
  const e = f.endpoints[name];
  return e ? HttpResponse.json(e.body as never, { status: e.status }) : notFound(`Not captured: ${name}.`);
}

export function m3Handlers(): HttpHandler[] {
  const pidOf = (params: Record<string, unknown>) => String(params.pid);
  /** The replay's own capture for an m3 project; else the reference journey's. */
  const endpoint = (name: string, fallback?: string) =>
    http.get(`/api/projects/:pid/${name}`, async ({ params }) => {
      const r = replayOf(pidOf(params));
      if (r) {
        const replay = await r;
        if (!replay) return notFound("No such journey.");
        const e = replay.f.endpoints[name];
        return e ? HttpResponse.json(e.body as never, { status: e.status }) : notFound(`Not captured: ${name}.`);
      }
      // The mock server's own projects answer their table reads themselves.
      return fallback ? captured(fallback, name) : undefined;
    });
  return [
    http.get("/api/models", async () => {
      const shared = (await LOADERS[SHARED]!()) as unknown as { models: unknown };
      return HttpResponse.json(shared.models as never);
    }),

    // ── the replayed journeys ─────────────────────────────────────────────
    http.get("/api/projects/:pid", async ({ params }) => {
      const r = replayOf(pidOf(params));
      if (!r) return undefined;
      const replay = await r;
      return replay ? HttpResponse.json(replay.view()) : notFound("No such journey.");
    }),
    http.post("/api/projects/:pid/decisions", async ({ params, request }) => {
      const r = replayOf(pidOf(params));
      if (!r) return undefined;
      const replay = await r;
      if (!replay) return notFound("No such journey.");
      const out = replay.decide(await request.clone().json());
      return HttpResponse.json(out.body as never, { status: out.status });
    }),
    http.get("/api/projects/:pid/stages/:stage", async ({ params }) => {
      const r = replayOf(pidOf(params));
      if (!r) return undefined;
      const replay = await r;
      const result = replay?.stage(String(params.stage));
      return result ? HttpResponse.json(result as never) : notFound("No such stage.");
    }),
    http.post("/api/projects/:pid/stages/:stage/run", async ({ params }) => {
      const r = replayOf(pidOf(params));
      if (!r) return undefined;
      const replay = await r;
      const status = replay?.view().stages[String(params.stage)];
      return status ? HttpResponse.json(status) : notFound("No such stage.");
    }),
    http.post("/api/projects/:pid/preview", async ({ params }) => {
      if (!replayOf(pidOf(params))) return undefined;
      return HttpResponse.json({
        kind: "none",
        views: [],
        basis: "The replay holds no previews: its capture recorded answers, not their consequences.",
        note: null,
        caution: null,
      });
    }),
    http.get("/api/projects/:pid/findings/:fid/evidence", ({ params }) => {
      if (!replayOf(pidOf(params))) return undefined;
      return notFound("The replay holds no finding evidence.");
    }),
    http.get("/api/projects/:pid/columns/:name/histogram", ({ params }) => {
      if (!replayOf(pidOf(params))) return undefined;
      return notFound("The replay holds no histograms.");
    }),
    http.get("/api/projects/:pid/jobs/:jid", ({ params }) => {
      if (!replayOf(pidOf(params))) return undefined;
      return notFound("The replay runs no jobs.");
    }),
    // Nothing streams for a replay: every change is answered by the request that made it (the
    // mock server's own event stream subscribes to a project it does not hold, and stays quiet).

    // ── the endpoints no other mock answers, for every project ─────────────
    endpoint("readings", "nhanes-prediction"),
    endpoint("methods", "nhanes-prediction"),
    endpoint("plan", "nhanes-inference"),
    endpoint("columns"),
    endpoint("table"),
    http.get("/api/projects/:pid/files", async ({ params }) => {
      const r = replayOf(pidOf(params));
      if (r) {
        const replay = await r;
        const e = replay?.f.endpoints.files;
        return e ? HttpResponse.json(e.body as never, { status: e.status }) : HttpResponse.json([]);
      }
      return captured("assembly", "files_list");
    }),
    http.post("/api/projects/:pid/files", () => captured("assembly", "files_add")),
    http.post("/api/projects/:pid/files/upload", () => captured("assembly", "files_upload")),
    http.post("/api/projects/:pid/join-preview", () => captured("assembly", "join_preview")),
    http.post("/api/projects/:pid/codebooks", async ({ request }) => {
      const body = (await request.clone().json()) as { labels?: boolean };
      return captured("assembly", body.labels ? "codebook_labels" : "codebook_nhanes");
    }),
    http.post("/api/projects/:pid/codebooks/upload", () => captured("assembly", "codebook_upload")),
  ];
}
