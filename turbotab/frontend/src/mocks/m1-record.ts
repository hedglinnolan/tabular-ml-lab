/**
 * The M1 mock for the Record (M1_CONTRACT §14 "record"): the interview Router on the view,
 * GET /api/teaching, server-authored sentences, the M1 validators' refusals, findings with
 * summaries and levers, and the M1 stages the Record and the banner read (roles, proposals,
 * cohort, split, shelf, design, fit, substitution) with their stale → running → fresh
 * statuses and job chips. It also seeds an NHANES-shaped project (m1-nhanes.ts).
 *
 * Registered in handlers.ts with one line, ahead of the M0 handlers: a resolver that returns
 * nothing falls through to them.
 */
import { http, HttpResponse, type HttpHandler } from "msw";
import type {
  AnyStageName,
  InterviewStep,
  M1StageArtifacts,
  RolesArtifact,
  TeachingEntry,
} from "../api/m1-types";
import { M1_STAGES } from "../api/m1-types";
import type {
  Decision,
  FindingsArtifact,
  JobView,
  ProjectState,
  ProjectView,
  Slot,
  StageResult,
  StageStatus,
  TargetInfoArtifact,
} from "../api/schema";
import { fold, type MockProject, type MockServer } from "./db";
import teachingJson from "./m2-teaching.json";
import shared from "./fixtures/m3-shared.json";
import { nhanesLike } from "./m1-nhanes";
import { route } from "./m1-router";
import {
  cohort,
  design,
  energyBearing,
  fit,
  proposalsArtifact,
  rolesArtifact,
  shelf,
  split,
  substitution,
  type CohortRows,
} from "./m1-stages";
import { sentenceFor, validateM1, voiceFindings } from "./m1-voice";
import { FEATURE_MAJOR_NAME, effectiveGrain, m2Mock, metabolomicsFeatureMajor } from "./m2-record";

type M1Stage = keyof M1StageArtifacts;

interface Def {
  name: M1Stage;
  deps: string[];
  reads: Slot[];
  requires: Slot[];
  ms: number;
  label: string;
}

/** Mirrors turbotab/core/stages/__init__.py (deps, reads, requires) and its job labels. */
const DEFS: Def[] = [
  {
    name: "roles",
    deps: ["ingest", "profile"],
    reads: ["lens", "target"],
    requires: [],
    ms: 320,
    label: "Reading what each column is",
  },
  {
    name: "proposals",
    deps: ["ingest", "profile", "roles"],
    reads: ["lens", "roles", "target"],
    requires: [],
    ms: 260,
    label: "Looking up what the field usually does",
  },
  {
    name: "cohort",
    deps: ["ingest", "target_info"],
    reads: ["target", "roles", "exclusions", "missing"],
    requires: ["target"],
    ms: 280,
    label: "Counting who is in the analysis",
  },
  {
    name: "split",
    deps: ["cohort", "target_info"],
    reads: ["split", "roles", "task"],
    requires: ["split"],
    ms: 300,
    label: "Drawing the held-out rows",
  },
  {
    name: "shelf",
    deps: ["cohort", "target_info"],
    reads: ["purpose", "task", "roles"],
    requires: ["roles"],
    ms: 200,
    label: "Ranking the model families for this table",
  },
  {
    name: "design",
    deps: ["split", "target_info"],
    reads: ["roles", "energy_adjustment", "missing", "models", "purpose"],
    requires: ["models", "roles"],
    ms: 420,
    label: "Building each model's pipeline",
  },
  {
    name: "fit",
    deps: ["design", "split", "target_info"],
    reads: ["models", "purpose", "task"],
    requires: ["models"],
    ms: 2600,
    label: "Fitting the models",
  },
  {
    name: "substitution",
    deps: ["fit", "design"],
    reads: ["substitution"],
    requires: ["substitution"],
    ms: 700,
    label: "Drawing the substitution curves",
  },
];

/** Every stage's dependencies, M0 and M1, for the Router's "still computing" test. */
const DEPS: Record<string, string[]> = {
  ingest: [],
  profile: ["ingest"],
  target_info: ["ingest"],
  findings: ["ingest"],
  ...Object.fromEntries(DEFS.map((d) => [d.name, d.deps])),
};

const STALE_PAUSE_MS = 450;
// The M2 entries as the M1/M2 journeys were written against, then every question the server has
// added since (the follow-up, the grouping, the survey, the estimand, the adjustment set, the
// time-varying and causal lanes), as the real server serves them (m3-shared.json), in its order.
const TEACHING: TeachingEntry[] = (() => {
  const m2 = teachingJson as TeachingEntry[];
  const real = (shared as unknown as { teaching: TeachingEntry[] }).teaching;
  const have = new Map(m2.map((e) => [e.key, e]));
  return real.map((e) => have.get(e.key) ?? e);
})();

interface M1Job {
  view: JobView;
  key: string;
  timers: ReturnType<typeof setTimeout>[];
}

interface M1Project {
  stages: Record<M1Stage, StageStatus>;
  artifacts: Map<string, unknown>;
  cohorts: Map<string, CohortRows>;
  latest: Partial<Record<M1Stage, string>>;
  jobs: Map<string, M1Job>;
  errors: Map<string, string>;
  cancelled: Set<string>;
  staleSeen: Set<string>;
  staleReady: Set<string>;
  reconciling: boolean;
}

function hash(s: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(16).padStart(8, "0");
}

const now = () => new Date().toISOString();
const isM1 = (stage: string): stage is M1Stage => (M1_STAGES as readonly string[]).includes(stage);
const isActive = (v: JobView) => v.state === "queued" || v.state === "running";

/** §12.4: the mostly-blank columns the latest missing-values answer leaves out. */
function dropColumns(p: MockProject): string[] {
  const rec = [...p.records].reverse().find((r) => r.decision.kind === "set_missing");
  return (rec?.decision as { drop_columns?: string[] } | undefined)?.drop_columns ?? [];
}

export function m1RecordHandlers(server: MockServer): HttpHandler[] {
  const projects = new Map<string, M1Project>();
  let jobSeq = 0;

  const m2 = m2Mock(fold);
  server.sentenceFor = (p, d) =>
    m2.sentence(p, d, fold(p.records), d.kind === "open_seal" ? heldOut(p) : null) ??
    sentenceFor(p.source, d, fold(p.records), p.records);
  /** The drawn split's held-out rows: what opening the seal scores. */
  const heldOut = (p: MockProject): number | null => {
    const m = projects.get(p.summary.id);
    const key = m?.latest.split;
    const split = key
      ? (m!.artifacts.get(`split@${key}`) as { n_holdout: number } | undefined)
      : undefined;
    return split?.n_holdout ?? null;
  };

  const blank = (stage: M1Stage): StageStatus => ({
    stage,
    status: "idle",
    key: null,
    fresh: false,
    missing: [],
    error: null,
    job_id: null,
    progress: null,
    updated_at: null,
    cancelled: false,
    held: null,
  });

  function ensure(pid: string): M1Project | null {
    const p = server.get(pid);
    if (!p) return null;
    let m = projects.get(pid);
    if (m) return m;
    m = {
      stages: Object.fromEntries(DEFS.map((d) => [d.name, blank(d.name)])) as Record<
        M1Stage,
        StageStatus
      >,
      artifacts: new Map(),
      cohorts: new Map(),
      latest: {},
      jobs: new Map(),
      errors: new Map(),
      cancelled: new Set(),
      staleSeen: new Set(),
      staleReady: new Set(),
      reconciling: false,
    };
    projects.set(pid, m);
    // M1 stages wait on M0 ones: follow them as they finish.
    server.subscribe(pid, (type, data) => {
      if (type !== "stage") return;
      const stage = (data as StageStatus).stage;
      if (!isM1(stage)) queueMicrotask(() => reconcile(pid));
    });
    reconcile(pid);
    return m;
  }

  function keyOf(
    p: MockProject,
    m: M1Project,
    def: Def,
    keys: Partial<Record<string, string>>,
    state: ProjectState,
  ) {
    const extra = def.name === "cohort" || def.name === "design" ? dropColumns(p) : [];
    return hash(
      JSON.stringify([
        def.name,
        1,
        def.deps.map(
          (d) =>
            keys[d] ??
            (p.stages as Record<string, StageStatus>)[d]?.key ??
            m.stages[d as M1Stage]?.key ??
            null,
        ),
        def.reads.map((s) => state[s]),
        extra,
        p.fingerprint,
      ]),
    );
  }

  function artifactOf<S extends M1Stage>(m: M1Project, stage: S, key: string | undefined) {
    return key
      ? (m.artifacts.get(`${stage}@${key}`) as M1StageArtifacts[S] | undefined)
      : undefined;
  }

  function targetInfoOf(p: MockProject): TargetInfoArtifact | null {
    const st = p.stages.target_info;
    if (st.status !== "fresh" || !st.key) return null;
    return (p.artifacts.get(`target_info@${st.key}`) as TargetInfoArtifact | undefined) ?? null;
  }

  function compute(
    p: MockProject,
    m: M1Project,
    stage: M1Stage,
    keys: Record<string, string>,
    state: ProjectState,
  ): unknown {
    const ds = p.source;
    const ti = targetInfoOf(p);
    const task = state.task ?? ti?.task ?? null;
    switch (stage) {
      case "roles":
        return rolesArtifact(ds, state);
      case "proposals":
        return proposalsArtifact(ds, state);
      case "cohort": {
        const c = cohort(ds, state, p.records, dropColumns(p));
        m.cohorts.set(keys.cohort!, c);
        return c.artifact;
      }
      case "split": {
        const c = m.cohorts.get(keys.cohort!)!;
        const roles =
          (artifactOf(m, "roles", keys.roles) as RolesArtifact | undefined) ??
          rolesArtifact(ds, state);
        // The grain in force: a stated grain (a unique identifier) is the seal's basis too.
        return split(ds, { ...state, grain: effectiveGrain(p, state) }, c, roles, task);
      }
      case "shelf":
        return shelf(state, artifactOf(m, "cohort", keys.cohort)!, task);
      case "design":
        return design(
          ds,
          state,
          artifactOf(m, "cohort", keys.cohort)!,
          artifactOf(m, "split", keys.split)!.n_train,
        );
      case "fit":
        return fit(
          state,
          artifactOf(m, "design", keys.design)!,
          artifactOf(m, "split", keys.split)!,
          task,
        );
      case "substitution":
        return substitution(
          state,
          artifactOf(m, "fit", keys.fit)!,
          artifactOf(m, "design", keys.design)!,
        );
    }
  }

  function reconcile(pid: string): void {
    const p = server.get(pid);
    const m = projects.get(pid);
    if (!p || !m || m.reconciling) return;
    m.reconciling = true;
    try {
      const state = fold(p.records);
      const keys: Record<string, string> = {};
      for (const [name, st] of Object.entries(p.stages)) if (st.key) keys[name] = st.key;
      const statusOf = (name: string): StageStatus | undefined =>
        isM1(name) ? m.stages[name] : (p.stages as Record<string, StageStatus>)[name];
      for (const def of DEFS) {
        const key = keyOf(p, m, def, keys, state);
        keys[def.name] = key;
        const ak = `${def.name}@${key}`;
        for (const job of m.jobs.values()) {
          if (job.view.stage === def.name && job.key !== key && isActive(job.view))
            stopJob(pid, m, job, "Superseded by a newer decision.");
        }
        const missing = def.requires.filter((s) => {
          const v = state[s];
          return v === null || (Array.isArray(v) && v.length === 0);
        });
        const hasOld = m.latest[def.name] !== undefined && m.latest[def.name] !== key;
        const active = [...m.jobs.values()].find(
          (j) => j.view.stage === def.name && j.key === key && isActive(j.view),
        );
        const base = {
          key,
          missing: [] as string[],
          error: null,
          job_id: null,
          progress: null,
          cancelled: false,
          held: null,
        };
        let next: Omit<StageStatus, "stage" | "updated_at">;
        if (missing.length) next = { ...base, status: "blocked", fresh: false, missing };
        else if (m.artifacts.has(ak)) next = { ...base, status: "fresh", fresh: true };
        else if (active)
          next = {
            ...base,
            status: active.view.state === "running" ? "running" : "queued",
            fresh: false,
            job_id: active.view.job_id,
            progress: active.view.progress,
          };
        else if (m.errors.has(ak))
          next = { ...base, status: "error", fresh: false, error: m.errors.get(ak)! };
        else if (m.cancelled.has(ak))
          next = { ...base, status: hasOld ? "stale" : "idle", fresh: false, cancelled: true };
        else if (!def.deps.every((d) => statusOf(d)?.status === "fresh")) {
          const stopped = def.deps.some(
            (d) => statusOf(d)?.cancelled || statusOf(d)?.status === "error",
          );
          next = {
            ...base,
            status: hasOld ? "stale" : stopped ? "idle" : "queued",
            fresh: false,
            cancelled: stopped,
          };
        } else if (hasOld && !m.staleReady.has(ak)) {
          // Announce staleness first, so the banner can veil before the recompute starts.
          if (!m.staleSeen.has(ak)) {
            m.staleSeen.add(ak);
            setTimeout(() => {
              m.staleReady.add(ak);
              reconcile(pid);
            }, STALE_PAUSE_MS);
          }
          next = { ...base, status: "stale", fresh: false };
        } else {
          const job = startJob(pid, p, m, def, key, { ...keys });
          next = { ...base, status: "queued", fresh: false, job_id: job.view.job_id, progress: 0 };
        }
        const prev = m.stages[def.name];
        const changed =
          prev.status !== next.status ||
          prev.key !== next.key ||
          prev.job_id !== next.job_id ||
          prev.error !== next.error ||
          prev.cancelled !== next.cancelled ||
          prev.missing.join() !== next.missing.join();
        if (changed) {
          m.stages[def.name] = { stage: def.name, ...next, updated_at: now() };
          server.emit(pid, "stage", m.stages[def.name]);
        }
      }
    } finally {
      m.reconciling = false;
    }
  }

  function startJob(
    pid: string,
    p: MockProject,
    m: M1Project,
    def: Def,
    key: string,
    keys: Record<string, string>,
  ): M1Job {
    const job: M1Job = {
      key,
      timers: [],
      view: {
        job_id: `m1job${(++jobSeq).toString(36)}`,
        label: def.label,
        stage: def.name,
        state: "queued",
        progress: 0,
        message: null,
        error: null,
      },
    };
    m.jobs.set(job.view.job_id, job);
    server.emit(pid, "job", job.view);
    const update = (patch: Partial<JobView>) => {
      job.view = { ...job.view, ...patch };
      server.emit(pid, "job", job.view);
    };
    job.timers.push(
      setTimeout(() => {
        update({ state: "running" });
        reconcile(pid);
      }, 60),
    );
    const steps = Math.max(2, Math.round(def.ms / 300));
    for (let i = 1; i <= steps; i++) {
      job.timers.push(
        setTimeout(
          () => {
            if (i < steps) {
              update({
                progress: i / steps,
                message: def.name === "fit" ? `fold ${i} of ${steps}` : null,
              });
              return;
            }
            const ak = `${def.name}@${key}`;
            try {
              m.artifacts.set(
                ak,
                compute(p, m, def.name, { ...keys, [def.name]: key }, fold(p.records)),
              );
              m.latest[def.name] = key;
              update({ state: "done", progress: 1 });
            } catch (e) {
              const message = e instanceof Error ? e.message : String(e);
              m.errors.set(ak, message);
              update({ state: "error", error: message });
            }
            reconcile(pid);
          },
          60 + (i * def.ms) / steps,
        ),
      );
    }
    return job;
  }

  function stopJob(pid: string, _m: M1Project, job: M1Job, message: string): void {
    for (const t of job.timers) clearTimeout(t);
    job.view = { ...job.view, state: "cancelled", message };
    server.emit(pid, "job", job.view);
  }

  function view(pid: string): ProjectView | null {
    const v = server.view(pid);
    const p = server.get(pid);
    const m = ensure(pid);
    if (!v || !p || !m) return null;
    const stages = { ...v.stages, ...m.stages, ...m2.statuses(p) };
    // M2 (m2-record.ts): the grain stated by a unique identifier counts as answered for what
    // follows it; then the seal's opening and the findings held for each question.
    const interview: InterviewStep[] = m2.route(
      p,
      route(
        m2.routingState(p, v.state),
        stages,
        targetInfoOf(p),
        v.decisions,
        DEPS,
        energyBearing,
        {
          featureMajor:
            p.source.name === FEATURE_MAJOR_NAME && p.source.columns[0]?.name === "feature_id",
        },
      ),
      stages,
    );
    return { ...v, stages, interview };
  }

  function stageResult(pid: string, stage: M1Stage): StageResult | null {
    const m = ensure(pid);
    if (!m) return null;
    const status = m.stages[stage];
    let key: string | null = null;
    let artifact: unknown = null;
    if (status.key && m.artifacts.has(`${stage}@${status.key}`)) {
      key = status.key;
      artifact = m.artifacts.get(`${stage}@${key}`);
    } else if (m.latest[stage]) {
      key = m.latest[stage]!;
      artifact = m.artifacts.get(`${stage}@${key}`) ?? null;
    }
    return {
      stage,
      key,
      fresh: key !== null && key === status.key,
      status: status.status,
      artifact,
    };
  }

  // M2: a metabolomics table exported features-in-rows (the orientation question fires on it),
  // then the NHANES-shaped table the M1 journey runs on, newest in the Recent list.
  server.createProject(metabolomicsFeatureMajor(), "path", {
    instant: true,
    createdAt: new Date(Date.now() - 2 * 86_400_000).toISOString(),
  });
  server.createProject(nhanesLike(), "upload", { instant: true });

  /** The working table's row count the seal plan counts from: the latest cohort's, else all. */
  const analyzed = (pid: string, p: MockProject): number => {
    const m = projects.get(pid);
    const key = m?.latest.cohort;
    const c = key
      ? (m!.artifacts.get(`cohort@${key}`) as { n_final: number } | undefined)
      : undefined;
    return c?.n_final ?? p.source.nRows;
  };

  return [
    http.get("/api/teaching", () => HttpResponse.json(TEACHING)),

    // A repair's preview (M2 §4): the changed cells and the column's distribution. Anything else
    // falls through to the stage mock's captured previews.
    http.post("/api/projects/:pid/preview", async ({ params, request }) => {
      const p = server.get(String(params.pid));
      if (!p) return undefined;
      const out = m2.preview(p, (await request.clone().json()) as Decision);
      return out ? HttpResponse.json(out) : undefined;
    }),

    http.get("/api/projects/:pid", ({ params }) => {
      const v = view(String(params.pid));
      return v ? HttpResponse.json(v) : undefined;
    }),

    http.post("/api/projects/:pid/decisions", async ({ params, request }) => {
      const pid = String(params.pid);
      const p = server.get(pid);
      if (!p) return undefined;
      const decision = (await request.clone().json()) as Decision;
      const m = ensure(pid);
      const refusal =
        m2.validate(p, decision, { ...p.stages, ...(m?.stages ?? {}) }) ??
        validateM1(p.source, decision, fold(p.records));
      if (refusal) return HttpResponse.json(refusal, { status: 409 });
      // A turned table replaces the one every stage reads before the decision recomputes them.
      m2.beforeDecision(p, decision);
      const out = server.decide(pid, decision);
      if ("error" in out) return HttpResponse.json(out, { status: 409 });
      reconcile(pid);
      m2.reconcile(pid, p, (type, data) => server.emit(pid, type, data));
      return HttpResponse.json(view(pid));
    }),

    http.get("/api/projects/:pid/stages/:stage", ({ params }) => {
      const pid = String(params.pid);
      const stage = String(params.stage) as AnyStageName;
      const p = server.get(pid);
      if (isM1(stage)) {
        const result = stageResult(pid, stage);
        if (result && p && stage === "fit" && result.artifact) {
          // The held-out scores are computed and withheld until the seal is opened (M2 §3).
          const state = fold(p.records);
          const fit = result.artifact as M1StageArtifacts["fit"];
          if (!state.seal_opened) {
            result.artifact = {
              ...fit,
              holdout_sealed: true,
              models: fit.models.map((x) => ({ ...x, holdout: null })),
            };
          }
        }
        return result ? HttpResponse.json(result) : undefined;
      }
      if (stage === "findings") {
        const result = server.stageResult(pid, "findings");
        if (!p || !result?.artifact) return undefined;
        const voiced = voiceFindings(
          p.source,
          result.artifact as FindingsArtifact,
          fold(p.records),
        );
        return HttpResponse.json({ ...result, artifact: m2.findings(p, voiced) });
      }
      if (p && (stage === "oriented" || stage === "structure" || stage === "seal_plan")) {
        const st = m2.statuses(p)[stage];
        if (!st) return undefined;
        const ti = targetInfoOf(p);
        const state = fold(p.records);
        const artifact = m2.artifact(p, stage, {
          task: state.task ?? ti?.task ?? null,
          nAnalyzed: analyzed(pid, p),
        });
        return HttpResponse.json({ stage, key: st.key, fresh: true, status: "fresh", artifact });
      }
      return undefined;
    }),

    http.post("/api/projects/:pid/stages/:stage/run", ({ params }) => {
      const pid = String(params.pid);
      const stage = String(params.stage);
      const m = ensure(pid);
      if (!isM1(stage) || !m) return undefined;
      for (const def of DEFS) {
        const ak = `${def.name}@${m.stages[def.name].key}`;
        m.errors.delete(ak);
        m.cancelled.delete(ak);
      }
      reconcile(pid);
      return HttpResponse.json(m.stages[stage]);
    }),

    http.get("/api/projects/:pid/jobs/:jid", ({ params }) => {
      const job = projects.get(String(params.pid))?.jobs.get(String(params.jid));
      return job ? HttpResponse.json(job.view) : undefined;
    }),

    http.post("/api/projects/:pid/jobs/:jid/cancel", ({ params }) => {
      const pid = String(params.pid);
      const m = projects.get(pid);
      const job = m?.jobs.get(String(params.jid));
      if (!m || !job) return undefined;
      if (isActive(job.view)) {
        stopJob(pid, m, job, "Cancelled.");
        m.cancelled.add(`${job.view.stage}@${job.key}`);
        reconcile(pid);
      }
      return HttpResponse.json(job.view);
    }),
  ];
}
