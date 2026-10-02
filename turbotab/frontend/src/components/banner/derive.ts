/**
 * The pipeline banner's content, derived from the stage artifacts (M1_CONTRACT §10). Pure:
 * everything the banner says is read from a computed artifact or a recorded answer, and a
 * segment whose artifact belongs to an earlier answer is veiled rather than restated.
 *
 *   Rows     the participant flow: loaded → each step that removed rows ▸ train | held out
 *   Columns  the column path to the model matrix, with the energy method
 *   Models   the families chosen
 *   Result   the primary metric of the best family, cross-validated
 */
import type { DatasetInfo, ProjectView, StageResult, StageStatus } from "../../api/schema";
import type {
  CohortArtifact,
  DesignArtifact,
  EnergyMethod,
  FitArtifact,
  QuestionKey,
  ShelfArtifact,
  SplitArtifact,
} from "../../api/m1-types";
import { veilFor, type VeilState } from "../../motion/StaleVeil";
import type { BannerSegment } from "../../state/focus";

export interface BannerInput {
  view: Pick<ProjectView, "stages" | "state" | "interview" | "summary">;
  ingest?: StageResult<DatasetInfo> | undefined;
  cohort?: StageResult<CohortArtifact> | undefined;
  split?: StageResult<SplitArtifact> | undefined;
  design?: StageResult<DesignArtifact> | undefined;
  fit?: StageResult<FitArtifact> | undefined;
  shelf?: StageResult<ShelfArtifact> | undefined;
}

export interface FlowCount {
  n: number;
  label: string;
}

interface Base<K extends BannerSegment> {
  key: K;
  label: string;
  veil: VeilState;
  /** Position in the propagate sweep: segments veil in this order. */
  order: number;
  /** Said when there is nothing computed to show yet (never a placeholder number). */
  waiting: string | null;
  /** The whole segment in one sentence, for its accessible name. */
  summary: string;
  /** The stage that failed or was stopped behind this segment: the banner offers to run it again. */
  retry: string | null;
}

export interface RowsSegment extends Base<"rows"> {
  flow: FlowCount[];
  train: number | null;
  holdout: number | null;
  /** A recorded "cross-validation only" answer: nothing is held out. */
  cvOnly: boolean;
}

export interface ColumnsSegment extends Base<"columns"> {
  from: number | null;
  to: number | null;
  /** What `from` counts when there is no `to` yet. */
  unit: "in the file" | "predictors";
  method: string | null;
}

export interface ModelsSegment extends Base<"models"> {
  labels: string[];
}

export interface ResultSegment extends Base<"result"> {
  metric: string | null;
  value: number | null;
  family: string | null;
  basis: string | null;
}

export type Segment = RowsSegment | ColumnsSegment | ModelsSegment | ResultSegment;

export interface BannerModel {
  segments: [RowsSegment, ColumnsSegment, ModelsSegment, ResultSegment];
  /** The segment the open question acts on: the "now" marker (--accent). */
  now: BannerSegment | null;
  /** The open question, by name. */
  nowLabel: string | null;
  /** True while the open question waits on a stage still computing. */
  nowWaiting: boolean;
}

/** The part of the pipeline each question acts on. */
export const QUESTION_SEGMENT: Record<QuestionKey, BannerSegment> = {
  lens: "rows",
  orientation: "columns",
  target: "rows",
  event: "models",
  task: "models",
  purpose: "models",
  grain: "rows",
  repeat_kind: "rows",
  unit: "rows",
  aggregation: "rows",
  temporal: "rows",
  roles: "columns",
  exclusions: "rows",
  missing: "rows",
  split: "rows",
  energy_adjustment: "columns",
  models: "models",
  substitution: "result",
  open_seal: "result",
};

export const QUESTION_NAME: Record<QuestionKey, string> = {
  lens: "the lens",
  orientation: "the table's orientation",
  target: "the outcome",
  event: "the event level",
  task: "the task",
  purpose: "the purpose",
  grain: "repeated units",
  repeat_kind: "repeats or time points",
  unit: "the unit of analysis",
  aggregation: "combining rows",
  temporal: "temporal prediction",
  roles: "column roles",
  exclusions: "exclusions",
  missing: "missing values",
  split: "held-out rows",
  energy_adjustment: "energy adjustment",
  models: "model families",
  substitution: "the substitution",
  open_seal: "opening the seal",
};

const METHOD_SHORT: Record<EnergyMethod, string | null> = {
  none: null,
  standard: "standard",
  residual: "residual",
  density_multivariate: "density + energy",
  density: "density",
  partition: "partition",
};

const FAMILY_SHORT: Record<string, string> = {
  linear: "linear",
  elastic_net: "elastic net",
  boosted_trees: "boosted trees",
};

const fmt = (n: number) => Math.round(n).toLocaleString("en-US");

// ── veils ────────────────────────────────────────────────────────────────────

const VEIL_RANK: Record<VeilState, number> = {
  fresh: 0,
  recomputing: 1,
  stale: 2,
  stopped: 3,
  failed: 4,
};

/** The most serious of several veils: one stale input makes the segment stale. */
export function worstVeil(...veils: VeilState[]): VeilState {
  return veils.reduce<VeilState>((a, b) => (VEIL_RANK[b] > VEIL_RANK[a] ? b : a), "fresh");
}

function veil(status: StageStatus | undefined, result: StageResult<unknown> | undefined) {
  return veilFor(status, result);
}

/** The first of these stages that failed or was stopped, if any: what a retry would run. */
function stopped(stages: Record<string, StageStatus | undefined>, names: string[]): string | null {
  return names.find((n) => stages[n]?.status === "error" || stages[n]?.cancelled) ?? null;
}

/** What to say while a stage has nothing on screen: work under way, or what it waits on. */
function pendingText(
  status: StageStatus | undefined,
  otherwise: string,
  working = "computing…",
): string {
  if (!status) return otherwise;
  if (status.status === "queued" || status.status === "running") return working;
  if (status.status === "error") return "did not finish";
  if (status.cancelled) return "stopped";
  return otherwise;
}

// ── segments ─────────────────────────────────────────────────────────────────

function rowsSegment(input: BannerInput): RowsSegment {
  const { view, ingest, cohort, split } = input;
  const stages = view.stages;
  const c = cohort?.artifact ?? null;
  const s = split?.artifact ?? null;
  const flow: FlowCount[] = [];
  if (c) {
    const [first, ...rest] = c.steps;
    if (first) flow.push({ n: first.n, label: first.label });
    for (const step of rest) if (step.dropped > 0) flow.push({ n: step.n, label: step.label });
    // The final count always closes the flow, even when no step removed a row.
    const last = flow[flow.length - 1];
    if (!last || last.n !== c.n_final) flow.push({ n: c.n_final, label: "in the analysis" });
  } else {
    const n = ingest?.artifact?.n_rows ?? view.summary.n_rows;
    if (n !== null && n !== undefined) flow.push({ n, label: "rows loaded" });
  }
  const cvOnly = s !== null && s.n_holdout === 0;
  const train = s ? s.n_train : null;
  const holdout = s && !cvOnly ? s.n_holdout : null;
  const v = worstVeil(
    c ? veil(stages.cohort, cohort) : "fresh",
    s ? veil(stages.split, split) : "fresh",
  );
  let waiting: string | null = null;
  if (flow.length === 0)
    waiting = pendingText(stages.ingest, "reading the file…", "reading the file…");

  const parts = flow.map((f) => `${fmt(f.n)} ${f.label.toLowerCase()}`);
  let summary = parts.length ? `Rows: ${parts.join(", then ")}` : `Rows: ${waiting ?? ""}`;
  if (train !== null) {
    summary += cvOnly
      ? `; ${fmt(train)} rows, cross-validation only`
      : `; ${fmt(train)} to train, ${fmt(holdout ?? 0)} held out`;
  }
  return {
    key: "rows",
    label: "Rows",
    veil: v,
    order: 0,
    waiting,
    summary: `${summary}.`,
    retry: stopped(stages, ["cohort", "split"]),
    flow,
    train,
    holdout,
    cvOnly,
  };
}

function laneCount(design: DesignArtifact, lane: "raw" | "matrix"): number {
  return design.lineage.nodes
    .filter((n) => n.lane === lane)
    .reduce((sum, n) => sum + (n.count ?? 1), 0);
}

function columnsSegment(input: BannerInput): ColumnsSegment {
  const { view, ingest, cohort, design } = input;
  const stages = view.stages;
  const d = design?.artifact ?? null;
  const c = cohort?.artifact ?? null;
  const ea = view.state.energy_adjustment;
  const method = ea ? METHOD_SHORT[ea.method] : null;
  let from: number | null = null;
  let to: number | null = null;
  let unit: ColumnsSegment["unit"] = "in the file";
  let v: VeilState = "fresh";
  if (d) {
    from = laneCount(d, "raw");
    to = laneCount(d, "matrix") || d.matrix.n_cols;
    unit = "predictors";
    v = worstVeil(veil(stages.design, design), c ? veil(stages.cohort, cohort) : "fresh");
  } else if (c && view.state.roles) {
    from = c.predictors.length;
    unit = "predictors";
    v = veil(stages.cohort, cohort);
  } else {
    const nCols = ingest?.artifact?.n_cols ?? view.summary.n_cols;
    if (nCols !== null && nCols !== undefined) from = nCols;
  }
  const waiting =
    from === null ? pendingText(stages.ingest, "reading the file…", "reading the file…") : null;
  let summary: string;
  if (from === null) summary = `Columns: ${waiting}`;
  else if (to !== null)
    summary = `Columns: ${fmt(from)} predictors become ${fmt(to)} model inputs${method ? `, energy-adjusted by the ${method} method` : ""}`;
  else summary = `Columns: ${fmt(from)} ${unit}`;
  return {
    key: "columns",
    label: "Columns",
    veil: v,
    order: 1,
    waiting,
    summary: `${summary}.`,
    retry: view.state.models?.length ? stopped(stages, ["design"]) : null,
    from,
    to,
    unit,
    method: d ? method : null,
  };
}

function modelsSegment(input: BannerInput): ModelsSegment {
  const { view, shelf, design } = input;
  const chosen = view.state.models ?? [];
  const labelOf = (key: string) =>
    FAMILY_SHORT[key] ??
    shelf?.artifact?.families.find((f) => f.key === key)?.label.toLowerCase() ??
    key.replace(/_/g, " ");
  const labels = chosen.map(labelOf);
  const d = design?.artifact ?? null;
  // The chosen families are the record's; their pipelines are the design's.
  const v = chosen.length && d ? veil(view.stages.design, design) : "fresh";
  const waiting = chosen.length ? null : "not chosen yet";
  return {
    key: "models",
    label: "Models",
    veil: v,
    order: 2,
    waiting,
    retry: null,
    summary: chosen.length
      ? `Models: ${labels.length} ${labels.length === 1 ? "family" : "families"}, ${labels.join(", ")}.`
      : "Models: not chosen yet.",
    labels,
  };
}

const LOWER_IS_BETTER = /rmse|mae|mse|brier|log_?loss|error/i;

/** The best family by its cross-validated primary metric, and the value. */
export function bestModel(
  fit: FitArtifact,
): { family: string; label: string; value: number } | null {
  const metric = fit.primary_metric;
  const lower = LOWER_IS_BETTER.test(metric);
  let best: { family: string; label: string; value: number } | null = null;
  for (const m of fit.models) {
    const value = m.cv[metric]?.mean;
    if (value === null || value === undefined || Number.isNaN(value)) continue;
    if (!best || (lower ? value < best.value : value > best.value)) {
      best = { family: m.family, label: m.label, value };
    }
  }
  return best;
}

function resultSegment(input: BannerInput): ResultSegment {
  const { view, fit } = input;
  const f = fit?.artifact ?? null;
  const best = f ? bestModel(f) : null;
  const metric = f ? (f.metric_labels[f.primary_metric] ?? f.primary_metric) : null;
  const v = f ? veil(view.stages.fit, fit) : "fresh";
  let waiting: string | null = null;
  if (!best) {
    waiting = view.state.models?.length
      ? pendingText(view.stages.fit, "after the fit", "fitting…")
      : "after the fit";
  }
  return {
    key: "result",
    label: "Result",
    veil: v,
    order: 3,
    waiting,
    retry: view.state.models?.length ? stopped(view.stages, ["fit", "substitution"]) : null,
    summary: best
      ? `Result: ${metric} ${formatMetric(best.value)} by cross-validation, best for ${best.label}.`
      : `Result: ${waiting}.`,
    metric,
    value: best?.value ?? null,
    family: best ? (FAMILY_SHORT[best.family] ?? best.label) : null,
    basis: best ? "CV" : null,
  };
}

/** Metrics as a reader expects them: three decimals, a true minus sign. */
export function formatMetric(v: number): string {
  const s = Math.abs(v) >= 100 ? v.toFixed(1) : v.toFixed(3);
  return s.startsWith("-") ? `−${s.slice(1)}` : s;
}

// ── the banner ───────────────────────────────────────────────────────────────

export function deriveBanner(input: BannerInput): BannerModel {
  const steps = input.view.interview;
  const open = steps.find((s) => s.status === "open");
  // While the next question waits on a stage, that is still where the user is.
  const waitingFirst = open
    ? undefined
    : steps.find(
        (s) =>
          s.status === "waiting" &&
          s.waiting_on.length > 0 &&
          s.waiting_on.every((w) => !steps.some((t) => t.key === w)),
      );
  const at = open ?? waitingFirst;
  return {
    segments: [
      rowsSegment(input),
      columnsSegment(input),
      modelsSegment(input),
      resultSegment(input),
    ],
    now: at ? QUESTION_SEGMENT[at.key] : null,
    nowLabel: at ? QUESTION_NAME[at.key] : null,
    nowWaiting: !open && !!waitingFirst,
  };
}
