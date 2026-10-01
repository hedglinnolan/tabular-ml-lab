/**
 * The mock server's state: projects, the decision log, a small stage graph with
 * keys, simulated jobs, and an event bus the SSE handler forwards.
 *
 * It follows BLUEPRINT §3-§5 closely enough that the UI meets the same sequences
 * it will meet on the real server: a decision changes keys downstream, stages with
 * an older artifact go `stale` first, then `queued` -> `running` (with progress) ->
 * `fresh`; a job whose key is no longer current is cancelled.
 */
import type {
  Decision,
  DecisionRecord,
  JobView,
  Lens,
  ProjectState,
  ProjectSummary,
  ProjectView,
  Refusal,
  Slot,
  StageName,
  StageResult,
  StageStatus,
  Task,
} from "../api/schema";
import { LENSES } from "../api/schema";
import type { MockDataset } from "./datasets";
import { findings, lensHints, targetInfo } from "./findings";
import { columnInfo, columnSummary, findColumn, isNumericDtype, nUnique } from "./stats";

export type EventType = "decision" | "stage" | "job" | "resync" | "ping";
type Listener = (type: EventType, data: unknown) => void;

interface StageDef {
  name: StageName;
  version: number;
  deps: StageName[];
  reads: Slot[];
  requires: Slot[];
  ms: number;
  label: string;
}

/** Plain-language job labels: what a job chip says while this stage runs. */
export const STAGE_DEFS: StageDef[] = [
  {
    name: "ingest",
    version: 1,
    deps: [],
    reads: [],
    requires: [],
    ms: 1100,
    label: "Reading the file into columnar storage",
  },
  {
    name: "profile",
    version: 1,
    deps: ["ingest"],
    reads: [],
    requires: [],
    ms: 600,
    label: "Summarizing every column",
  },
  {
    name: "target_info",
    version: 1,
    deps: ["ingest"],
    reads: ["target", "task"],
    requires: ["target"],
    ms: 650,
    label: "Reading the outcome column",
  },
  {
    name: "findings",
    version: 1,
    deps: ["ingest"],
    reads: ["lens", "target"],
    requires: ["lens"],
    ms: 1000,
    label: "Checking the table against the chosen lenses",
  },
];

/** How long a stage shows `stale` before its recompute is queued. */
const STALE_PAUSE_MS = 450;

interface MockJob {
  view: JobView;
  key: string;
  timers: ReturnType<typeof setTimeout>[];
}

export interface MockProject {
  summary: ProjectSummary;
  source: MockDataset;
  fingerprint: string;
  records: DecisionRecord[];
  stages: Record<StageName, StageStatus>;
  artifacts: Map<string, unknown>;
  errors: Map<string, string>;
  latest: Partial<Record<StageName, string>>;
  jobs: Map<string, MockJob>;
  cancelled: Set<string>;
  staleSeen: Set<string>;
  staleReady: Set<string>;
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

export class MockServer {
  private projects = new Map<string, MockProject>();
  /** The server authors each record's sentence (M1_CONTRACT §2); m1-record.ts supplies it. */
  sentenceFor?: (p: MockProject, decision: Decision) => string | null;
  private listeners = new Map<string, Set<Listener>>();
  private seq = 0;

  subscribe(pid: string, fn: Listener): () => void {
    const set = this.listeners.get(pid) ?? new Set<Listener>();
    set.add(fn);
    this.listeners.set(pid, set);
    return () => set.delete(fn);
  }

  /** Public so the M1 mock modules (m1-*.ts) can push their own stage and job events. */
  emit(pid: string, type: EventType, data: unknown): void {
    for (const fn of this.listeners.get(pid) ?? []) fn(type, data);
  }

  // ─── projects ────────────────────────────────────────────────────────────
  createProject(
    source: MockDataset,
    kind: "path" | "upload",
    opts: { instant?: boolean; createdAt?: string; decisions?: Decision[] } = {},
  ): ProjectSummary {
    const id = `p${(++this.seq).toString(36)}${hash(source.name + this.seq).slice(0, 5)}`;
    const blank = (stage: StageName): StageStatus => ({
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
    });
    const p: MockProject = {
      summary: {
        id,
        name: source.name.replace(/\.(csv|tsv|txt|parquet|xlsx)$/i, "").replace(/_/g, " "),
        created_at: opts.createdAt ?? now(),
        source_kind: kind,
        source_name: source.name,
        n_rows: null,
        n_cols: null,
        ingest: null,
      },
      source,
      fingerprint: hash(`${source.name}:${source.nRows}:${source.columns.length}`),
      records: [],
      stages: {
        ingest: blank("ingest"),
        profile: blank("profile"),
        target_info: blank("target_info"),
        findings: blank("findings"),
      },
      artifacts: new Map(),
      errors: new Map(),
      latest: {},
      jobs: new Map(),
      cancelled: new Set(),
      staleSeen: new Set(),
      staleReady: new Set(),
    };
    this.projects.set(id, p);
    for (const d of opts.decisions ?? []) this.append(p, d);
    if (opts.instant) this.computeNow(p);
    this.reconcile(p);
    return p.summary;
  }

  listProjects(): ProjectSummary[] {
    return [...this.projects.values()]
      .map((p) => p.summary)
      .sort((a, b) => b.created_at.localeCompare(a.created_at));
  }

  get(pid: string): MockProject | undefined {
    return this.projects.get(pid);
  }

  view(pid: string): ProjectView | null {
    const p = this.projects.get(pid);
    if (!p) return null;
    return {
      summary: p.summary,
      state: fold(p.records),
      decisions: [...p.records],
      stages: { ...p.stages },
      interview: [], // the Router is the server's; the M1 mock work may mirror it
    };
  }

  // ─── decisions ───────────────────────────────────────────────────────────
  decide(pid: string, decision: Decision): ProjectView | Refusal {
    const p = this.projects.get(pid)!;
    const refusal = this.validate(p, decision);
    if (refusal) return refusal;
    const record = this.append(p, decision);
    this.emit(pid, "decision", record);
    this.reconcile(p);
    return this.view(pid)!;
  }

  private append(p: MockProject, decision: Decision): DecisionRecord {
    const record: DecisionRecord = {
      id: `d${p.records.length + 1}_${hash(p.summary.id + p.records.length).slice(0, 4)}`,
      seq: p.records.length + 1,
      at: now(),
      note: null,
      sentence: this.sentenceFor?.(p, decision) ?? null,
      decision,
    };
    p.records.push(record);
    return record;
  }

  private validate(p: MockProject, d: Decision): Refusal | null {
    const refuse = (code: string, message: string, exits: Refusal["error"]["exits"] = []) => ({
      error: { code, message, exits },
    });
    const state = fold(p.records);
    switch (d.kind) {
      case "set_lens": {
        if (!d.lenses.length)
          return refuse(
            "lens_empty",
            "The lens question needs an answer. An empty selection would be indistinguishable from never having asked.",
          );
        if (new Set(d.lenses).size !== d.lenses.length)
          return refuse("lens_duplicate", "Each lens can be chosen once.");
        const unknown = d.lenses.find((l) => !(LENSES as readonly string[]).includes(l));
        if (unknown)
          return refuse("lens_unknown", `'${unknown}' is not one of ${LENSES.join(", ")}.`);
        return null;
      }
      case "set_target":
        if (!findColumn(p.source, d.column))
          return refuse("unknown_column", `No column named '${d.column}' in this table.`);
        return null;
      case "set_task": {
        if (!state.target)
          return refuse("no_target", "Choose the outcome first; the task describes it.");
        const col = findColumn(p.source, state.target)!;
        const unique = nUnique(col);
        if (d.column !== state.target)
          return refuse(
            "not_the_target",
            `The outcome is '${state.target}', not '${d.column}'; a task answers for the outcome.`,
          );
        const detected = targetInfo(p.source, state.target, null).detected_task;
        const exits = [
          {
            label: `Model it as ${detected}`,
            decision: { kind: "set_task" as const, column: state.target, task: detected },
          },
          { label: "Keep the current answer", decision: null },
        ];
        if (d.task === "binary" && unique !== 2)
          return refuse(
            "task_mismatch",
            `\`${col.name}\` has ${unique.toLocaleString("en-US")} distinct values; a binary task needs exactly two.`,
            exits,
          );
        if (d.task === "regression" && !isNumericDtype(col.dtype))
          return refuse(
            "task_mismatch",
            `\`${col.name}\` holds text labels; a regression needs numbers.`,
            exits,
          );
        if (d.task === "multiclass" && unique > 50)
          return refuse(
            "task_mismatch",
            `\`${col.name}\` has ${unique.toLocaleString("en-US")} distinct values, too many to be classes.`,
            exits,
          );
        return null;
      }
      case "set_purpose":
        return null;
      case "revert": {
        const target = p.records.find((r) => r.id === d.decision_id);
        if (!target) return refuse("unknown_decision", "There is no such decision in this record.");
        if (target.decision.kind === "revert")
          return refuse(
            "revert_of_revert",
            "A revert cannot itself be reverted; record the answer again instead.",
          );
        return null;
      }
      default:
        return null; // M1 kinds: validated by the M1 mock work
    }
  }

  // ─── the stage graph ─────────────────────────────────────────────────────
  private computeNow(p: MockProject): void {
    const state = fold(p.records);
    const keys: Partial<Record<StageName, string>> = {};
    for (const def of STAGE_DEFS) {
      const key = stageKey(p, def, keys, state);
      keys[def.name] = key;
      if (missingSlots(def, state).length) continue;
      if (!def.deps.every((d) => p.artifacts.has(`${d}@${keys[d]}`))) continue;
      p.artifacts.set(`${def.name}@${key}`, this.compute(p, def.name, state));
      p.latest[def.name] = key;
    }
  }

  private reconcile(p: MockProject): void {
    const state = fold(p.records);
    const keys: Partial<Record<StageName, string>> = {};
    for (const def of STAGE_DEFS) {
      const key = stageKey(p, def, keys, state);
      keys[def.name] = key;
      const ak = `${def.name}@${key}`;
      for (const job of p.jobs.values()) {
        if (job.view.stage === def.name && job.key !== key && isActive(job.view)) {
          this.stopJob(p, job, "Superseded by a newer decision.");
        }
      }
      const missing = missingSlots(def, state);
      const hasOld = p.latest[def.name] !== undefined && p.latest[def.name] !== key;
      const active = [...p.jobs.values()].find(
        (j) => j.view.stage === def.name && j.key === key && isActive(j.view),
      );
      let next: Omit<StageStatus, "stage" | "updated_at">;
      const base = {
        key,
        missing: [] as string[],
        error: null,
        job_id: null,
        progress: null,
        cancelled: false,
      };
      if (missing.length) {
        next = { ...base, status: "blocked", fresh: false, missing };
      } else if (p.artifacts.has(ak)) {
        next = { ...base, status: "fresh", fresh: true };
      } else if (active) {
        next = {
          ...base,
          status: active.view.state === "running" ? "running" : "queued",
          fresh: false,
          job_id: active.view.job_id,
          progress: active.view.progress,
        };
      } else if (p.errors.has(ak)) {
        next = { ...base, status: "error", fresh: false, error: p.errors.get(ak)! };
      } else if (p.cancelled.has(ak)) {
        next = { ...base, status: hasOld ? "stale" : "idle", fresh: false, cancelled: true };
      } else if (!def.deps.every((d) => p.stages[d].status === "fresh")) {
        const stopped = def.deps.some((d) => p.stages[d].cancelled);
        next = {
          ...base,
          status: hasOld ? "stale" : stopped ? "idle" : "queued",
          fresh: false,
          cancelled: stopped,
        };
      } else if (hasOld && !p.staleReady.has(ak)) {
        // Announce staleness first, so the panel can veil before the recompute starts.
        if (!p.staleSeen.has(ak)) {
          p.staleSeen.add(ak);
          setTimeout(() => {
            p.staleReady.add(ak);
            this.reconcile(p);
          }, STALE_PAUSE_MS);
        }
        next = { ...base, status: "stale", fresh: false };
      } else {
        const job = this.startJob(p, def, key);
        next = { ...base, status: "queued", fresh: false, job_id: job.view.job_id, progress: 0 };
      }
      const prev = p.stages[def.name];
      const changed =
        prev.status !== next.status ||
        prev.key !== next.key ||
        prev.job_id !== next.job_id ||
        prev.error !== next.error ||
        prev.cancelled !== next.cancelled ||
        prev.missing.join() !== next.missing.join();
      if (changed) {
        p.stages[def.name] = { stage: def.name, ...next, updated_at: now() };
        if (def.name === "ingest") p.summary = { ...p.summary, ingest: p.stages.ingest };
        this.emit(p.summary.id, "stage", p.stages[def.name]);
      }
    }
  }

  private startJob(p: MockProject, def: StageDef, key: string): MockJob {
    const job: MockJob = {
      key,
      timers: [],
      view: {
        job_id: `job${(++this.seq).toString(36)}`,
        label: def.label,
        stage: def.name,
        state: "queued",
        progress: 0,
        message: null,
        error: null,
      },
    };
    p.jobs.set(job.view.job_id, job);
    this.emit(p.summary.id, "job", job.view);
    const steps = Math.max(3, Math.round(def.ms / 140));
    const update = (patch: Partial<JobView>) => {
      job.view = { ...job.view, ...patch };
      this.emit(p.summary.id, "job", job.view);
    };
    job.timers.push(
      setTimeout(() => {
        update({ state: "running" });
        this.reconcile(p);
      }, 80),
    );
    for (let i = 1; i <= steps; i++) {
      job.timers.push(
        setTimeout(
          () => {
            if (i < steps) {
              update({ progress: i / steps });
              return;
            }
            const ak = `${def.name}@${key}`;
            try {
              p.artifacts.set(ak, this.compute(p, def.name, fold(p.records)));
              p.latest[def.name] = key;
              update({ state: "done", progress: 1 });
            } catch (e) {
              const message = e instanceof Error ? e.message : String(e);
              p.errors.set(ak, message);
              update({ state: "error", error: message });
            }
            this.reconcile(p);
          },
          80 + (i * def.ms) / steps,
        ),
      );
    }
    return job;
  }

  private stopJob(p: MockProject, job: MockJob, message: string): void {
    for (const t of job.timers) clearTimeout(t);
    job.view = { ...job.view, state: "cancelled", message };
    this.emit(p.summary.id, "job", job.view);
  }

  cancelJob(pid: string, jid: string): JobView | null {
    const p = this.projects.get(pid);
    const job = p?.jobs.get(jid);
    if (!p || !job) return null;
    if (isActive(job.view)) {
      this.stopJob(p, job, "Cancelled.");
      p.cancelled.add(`${job.view.stage}@${job.key}`);
      this.reconcile(p);
    }
    return job.view;
  }

  /** Compute a stage again after a failure or a cancel, and whatever it waits on. */
  runStage(pid: string, stage: StageName): StageStatus | null {
    const p = this.projects.get(pid);
    if (!p) return null;
    const todo: StageName[] = [stage];
    while (todo.length) {
      const name = todo.pop()!;
      const ak = `${name}@${p.stages[name].key}`;
      p.errors.delete(ak);
      p.cancelled.delete(ak);
      todo.push(...STAGE_DEFS.find((d) => d.name === name)!.deps);
    }
    this.reconcile(p);
    return p.stages[stage];
  }

  job(pid: string, jid: string): JobView | null {
    return this.projects.get(pid)?.jobs.get(jid)?.view ?? null;
  }

  private compute(p: MockProject, stage: StageName, state: ProjectState): unknown {
    const ds = p.source;
    switch (stage) {
      case "ingest":
        p.summary = { ...p.summary, n_rows: ds.nRows, n_cols: ds.columns.length };
        return {
          n_rows: ds.nRows,
          n_cols: ds.columns.length,
          columns: ds.columns.map(columnInfo),
          source_bytes: ds.sourceBytes,
          parquet_bytes: Math.round(ds.sourceBytes * 0.38),
          ingest_seconds: 0.21,
          fingerprint: p.fingerprint,
          warnings: [],
        };
      case "profile":
        return {
          columns: ds.columns.map(columnSummary),
          lens_hints: lensHints(ds),
          basis: `Computed on all ${ds.nRows.toLocaleString("en-US")} rows of the ingested table.`,
        };
      case "target_info":
        return targetInfo(ds, state.target!, state.task);
      case "findings":
        return findings(ds, state.lens ?? [], state.target);
    }
  }

  stageResult(pid: string, stage: StageName): StageResult | null {
    const p = this.projects.get(pid);
    if (!p) return null;
    const status = p.stages[stage];
    const currentKey = status.key;
    let key: string | null = null;
    let artifact: unknown = null;
    if (currentKey && p.artifacts.has(`${stage}@${currentKey}`)) {
      key = currentKey;
      artifact = p.artifacts.get(`${stage}@${currentKey}`);
    } else if (p.latest[stage]) {
      key = p.latest[stage]!;
      artifact = p.artifacts.get(`${stage}@${key}`) ?? null;
    }
    return {
      stage,
      key,
      fresh: key !== null && key === currentKey,
      status: status.status,
      artifact,
    };
  }

  isIngested(pid: string): boolean {
    return this.projects.get(pid)?.stages.ingest.status === "fresh";
  }
}

function isActive(v: JobView): boolean {
  return v.state === "queued" || v.state === "running";
}

function missingSlots(def: StageDef, state: ProjectState): string[] {
  return def.requires.filter((s) => {
    const v = state[s];
    return v === null || (Array.isArray(v) && v.length === 0);
  });
}

function stageKey(
  p: MockProject,
  def: StageDef,
  keys: Partial<Record<StageName, string>>,
  state: ProjectState,
): string {
  return hash(
    JSON.stringify([
      def.name,
      def.version,
      def.deps.map((d) => keys[d]),
      def.reads.map((s) => state[s]),
      p.fingerprint,
    ]),
  );
}

function slotOf(d: Decision): Slot | null {
  switch (d.kind) {
    case "set_lens":
      return "lens";
    case "set_target":
      return "target";
    case "set_task":
      return "task";
    case "set_purpose":
      return "purpose";
    case "set_roles":
      return "roles";
    case "set_energy_adjustment":
      return "energy_adjustment";
    case "set_exclusions":
      return "exclusions";
    case "set_missing":
      return "missing";
    case "set_split":
      return "split";
    case "select_models":
      return "models";
    case "set_substitution":
      return "substitution";
    case "set_orientation":
      return "orientation";
    case "set_event":
      return "event";
    case "set_grain":
      return "grain";
    case "set_repeat_kind":
      return "repeat_kind";
    case "set_unit":
      return "unit";
    case "set_aggregation":
      return "aggregation";
    case "set_temporal":
      return "temporal";
    case "open_seal":
      return "seal_opened";
    case "apply_repair":
    case "defer_finding":
    case "dismiss_finding":
      return "findings";
    case "revert":
      return null;
  }
}

function valueOf(d: Decision): ProjectState[Slot] {
  switch (d.kind) {
    case "set_lens":
      return d.lenses as Lens[];
    case "set_target":
      return d.column;
    case "set_task":
      return d.task as Task;
    case "set_purpose":
      return d.purpose;
    case "set_roles":
      return d.roles;
    case "set_energy_adjustment": {
      const { kind: _kind, ...value } = d;
      return value;
    }
    case "set_exclusions":
      return d.rules;
    case "set_missing":
      return {
        strategy: d.strategy,
        drop_columns: d.drop_columns ?? [],
        categorical: d.categorical ?? "impute",
        indicators: d.indicators ?? false,
      };
    case "set_split":
      return { holdout: d.holdout, seed: d.seed ?? 0, folds: d.folds ?? 5 };
    case "select_models":
      return d.models;
    case "set_substitution":
      return { donor: d.donor, recipient: d.recipient, step_kcal: d.step_kcal ?? 100, n_boot: d.n_boot ?? 0 };
    // M2 kinds: the mock records them without modeling their effect (M2's mock work).
    case "set_orientation":
      return d.orientation;
    case "set_event":
      return d.level;
    case "set_unit":
      return d.unit;
    case "open_seal":
      return true;
    case "set_grain":
    case "set_repeat_kind":
    case "set_aggregation":
    case "set_temporal": {
      const { kind: _k, ...value } = d;
      return value as ProjectState[Slot];
    }
    case "apply_repair":
    case "defer_finding":
    case "dismiss_finding":
      return null;
    case "revert":
      return null;
  }
}

/**
 * State is a fold of the log: each decision writes one slot; a revert restores its prior
 * value. A task answer names its column, so the task slot holds the answers by column and
 * reads the one for the current target (as the server's fold does).
 */
export function fold(records: DecisionRecord[]): ProjectState {
  type Slots = Omit<ProjectState, "task"> & { task: ReadonlyMap<string, Task> };
  const state: Slots = {
    lens: null,
    target: null,
    task: new Map(),
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
  };
  const before = new Map<string, { slot: Slot; prior: Slots[Slot] }>();
  for (const r of [...records].sort((a, b) => a.seq - b.seq)) {
    const d = r.decision;
    if (d.kind === "revert") {
      const undone = before.get(d.decision_id);
      if (!undone) continue;
      before.set(r.id, { slot: undone.slot, prior: state[undone.slot] });
      (state as Record<Slot, unknown>)[undone.slot] = undone.prior;
      continue;
    }
    const slot = slotOf(d)!;
    before.set(r.id, { slot, prior: state[slot] });
    (state as Record<Slot, unknown>)[slot] =
      d.kind === "set_task" ? new Map(state.task).set(d.column, d.task as Task) : valueOf(d);
  }
  const task = state.target === null ? null : (state.task.get(state.target) ?? null);
  return { ...state, task };
}
