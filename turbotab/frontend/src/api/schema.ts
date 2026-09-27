/**
 * The M0 JSON contract, hand-written. Names match the server's pydantic models
 * exactly; `npm run gen:api` will later generate these from the OpenAPI document
 * and this file becomes a thin re-export. Times are ISO-8601 strings.
 */

export type Mode = "local" | "server";

export interface Health {
  version: string;
  mode: Mode;
  workers: number;
}

export interface ProjectSummary {
  id: string;
  name: string;
  created_at: string;
  source_kind: "path" | "upload";
  source_name: string;
  n_rows: number | null;
  n_cols: number | null;
}

export const LENSES = ["metabolomics", "genomics", "dietary", "clinical", "survey"] as const;
export type Lens = (typeof LENSES)[number];

export type Task = "regression" | "binary" | "multiclass";
export type Purpose = "prediction" | "inference";

export type Decision =
  | { kind: "set_lens"; lenses: Lens[] }
  | { kind: "set_target"; column: string }
  | { kind: "set_task"; task: Task }
  | { kind: "set_purpose"; purpose: Purpose }
  | { kind: "revert"; decision_id: string };

export type DecisionKind = Decision["kind"];

export interface DecisionRecord {
  id: string;
  seq: number;
  at: string;
  note: string | null;
  decision: Decision;
}

export interface ProjectState {
  lens: Lens[] | null;
  target: string | null;
  task: Task | null;
  purpose: Purpose | null;
}

export type Slot = keyof ProjectState;

export type StageState = "idle" | "queued" | "running" | "fresh" | "stale" | "blocked" | "error";

export interface StageStatus {
  stage: string;
  status: StageState;
  key: string | null;
  fresh: boolean;
  missing: string[];
  error: string | null;
  job_id: string | null;
  progress: number | null;
  updated_at: string | null;
}

export interface ProjectView {
  summary: ProjectSummary;
  state: ProjectState;
  decisions: DecisionRecord[];
  stages: { [stage: string]: StageStatus };
}

export interface StageResult<A = unknown> {
  stage: string;
  key: string | null;
  fresh: boolean;
  status: StageState;
  artifact: A | null;
}

export type Dtype = "numeric" | "integer" | "boolean" | "categorical" | "datetime" | "text";
export const DTYPES: readonly Dtype[] = [
  "numeric",
  "integer",
  "boolean",
  "categorical",
  "datetime",
  "text",
];

/** A cell or sample value as JSON carries it. */
export type Scalar = string | number | boolean | null;

export interface ColumnInfo {
  name: string;
  dtype: Dtype;
  physical_type: string;
  n_missing: number;
  n_unique: number;
  sample: Scalar[];
}

export interface DatasetInfo {
  n_rows: number;
  n_cols: number;
  columns: ColumnInfo[];
  source_bytes: number;
  parquet_bytes: number;
  ingest_seconds: number;
  fingerprint: string;
  warnings: string[];
}

export interface ValueCount {
  value: Scalar;
  count: number;
}

export interface ColumnSummary {
  name: string;
  dtype: Dtype;
  n: number;
  n_missing: number;
  n_unique: number;
  mean: number | null;
  std: number | null;
  min: number | null;
  q25: number | null;
  median: number | null;
  q75: number | null;
  max: number | null;
  top: ValueCount[] | null;
}

export interface LensHint {
  lens: Lens;
  because: string;
}

export interface ProfileArtifact {
  columns: ColumnSummary[];
  lens_hints: LensHint[];
  basis: string;
}

export type Confidence = "high" | "medium" | "low";

export interface Histogram {
  column: string;
  edges: number[];
  counts: number[];
  n_missing: number;
}

export interface TargetInfoArtifact {
  column: string;
  task: Task;
  detected_task: Task;
  confidence: Confidence;
  reason: string;
  histogram: Histogram | null;
  classes: ValueCount[] | null;
}

export type Severity = "info" | "warning" | "critical";

export interface Finding {
  id: string;
  severity: Severity;
  title: string;
  detail: string;
  why_it_matters: string | null;
  affected_columns: string[];
  source: "profile" | "pack" | "structural";
  lens: Lens | null;
  evidence: { status: string; source: string } | null;
}

export interface FindingsArtifact {
  findings: Finding[];
  basis: string;
}

/** Stage name -> artifact type, for the four M0 stages. */
export interface StageArtifacts {
  ingest: DatasetInfo;
  profile: ProfileArtifact;
  target_info: TargetInfoArtifact;
  findings: FindingsArtifact;
}
export type StageName = keyof StageArtifacts;
export const STAGES: readonly StageName[] = ["ingest", "profile", "target_info", "findings"];

export interface TableWindow {
  columns: string[];
  rows: Scalar[][];
  total_rows: number;
  offset: number;
}

export type JobState = "queued" | "running" | "done" | "error" | "cancelled";

export interface JobView {
  job_id: string;
  label: string;
  stage: string | null;
  state: JobState;
  progress: number | null;
  message: string | null;
  error: string | null;
}

export interface RefusalExit {
  label: string;
  decision: Decision | null;
}

export interface Refusal {
  error: {
    code: string;
    message: string;
    exits: RefusalExit[];
  };
}

export interface FsEntry {
  name: string;
  path: string;
  is_dir: boolean;
  size: number | null;
}

export interface FsListing {
  path: string;
  parent: string | null;
  entries: FsEntry[];
}

/** Server-sent events on /api/projects/{pid}/events. */
export type ProjectEvent =
  | { type: "decision"; data: DecisionRecord }
  | { type: "stage"; data: StageStatus }
  | { type: "job"; data: JobView }
  | { type: "resync"; data: Record<string, never> }
  | { type: "ping"; data: unknown };
