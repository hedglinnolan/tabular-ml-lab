/**
 * The M0 JSON contract, as thin aliases over the types `npm run gen:api` generates
 * from the server's OpenAPI document (src/api/generated.ts — never edit that file).
 * Regenerate after any server model change:
 *
 *   venv/bin/python -m turbotab.server.openapi --write && npm run gen:api
 *
 * Only a few things here are not straight re-exports: the runtime lists of the
 * enums (checked against the generated unions below), `StageResult<A>`, which
 * types `artifact` by stage name, and `ProjectEvent`, the SSE stream, which
 * OpenAPI cannot describe. Times are ISO-8601 strings.
 */
import type { components } from "./generated";

type S = components["schemas"];

export type Health = S["Health"];
export type Mode = Health["mode"];

export type ProjectSummary = S["ProjectSummary"];

export type Lens = S["LensHint"]["lens"];
export type Task = S["SetTask"]["task"];
export type Purpose = S["SetPurpose"]["purpose"];

export type Decision = S["DecisionRecord"]["decision"];
export type DecisionKind = Decision["kind"];
export type DecisionRecord = S["DecisionRecord"];

export type ProjectState = S["ProjectState"];
export type Slot = keyof ProjectState;
/** Every slot of ProjectState, in asking order (M1_CONTRACT.md). */
export const SLOTS: readonly Slot[] = [
  "lens", "orientation", "target", "event", "task", "purpose", "grain", "repeat_kind", "unit",
  "aggregation", "temporal", "roles", "exclusions", "missing", "split", "energy_adjustment",
  "models", "substitution", "seal_opened", "findings",
];

export type StageStatus = S["StageStatus"];
export type StageState = StageStatus["status"];

export type ProjectView = S["ProjectView"];

/** A stage's result, with `artifact` typed by the stage it came from. */
export type StageResult<A = unknown> = Omit<S["StageResult"], "artifact"> & {
  artifact: A | null;
};

export type Dtype = S["ColumnInfo"]["dtype"];

/** A cell or sample value as the UI renders it. The contract types these `unknown`. */
export type Scalar = string | number | boolean | null;

export type ColumnInfo = S["ColumnInfo"];
export type DatasetInfo = S["DatasetInfo"];
export type ValueCount = S["ValueCount"];
export type ColumnSummary = S["ColumnSummary"];
export type LensHint = S["LensHint"];
export type ProfileArtifact = S["ProfileArtifact"];
export type Histogram = S["Histogram"];
export type TargetInfoArtifact = S["TargetInfo"];
export type Confidence = TargetInfoArtifact["confidence"];
export type Finding = S["Finding"];
export type Severity = Finding["severity"];
export type FindingsArtifact = S["FindingsArtifact"];

/** Stage name -> artifact type, for the four M0 stages. */
export interface StageArtifacts {
  ingest: DatasetInfo;
  profile: ProfileArtifact;
  target_info: TargetInfoArtifact;
  findings: FindingsArtifact;
}
export type StageName = keyof StageArtifacts;

export type TableWindow = S["TableWindow"];

export type JobView = S["JobView"];
export type JobState = JobView["state"];

export type RefusalExit = S["Exit"];
export type Refusal = S["Refusal"];

export type FsEntry = S["FsEntry"];
export type FsListing = S["FsListing"];

// ── runtime lists of the contract's enums ────────────────────────────────────
// `Exactly<L, U>` fails to compile if a list misses or invents a member of U, so a
// regenerated union that gains a value breaks the build here, not in a picker.
type Exactly<L extends readonly unknown[], U> = [U] extends [L[number]]
  ? [L[number]] extends [U]
    ? L
    : never
  : never;

const lenses = ["metabolomics", "genomics", "dietary", "clinical", "survey"] as const;
export const LENSES: Exactly<typeof lenses, Lens> = lenses;

const dtypes = ["numeric", "integer", "boolean", "categorical", "datetime", "text"] as const;
export const DTYPES: Exactly<typeof dtypes, Dtype> = dtypes;

const stages = ["ingest", "profile", "target_info", "findings"] as const;
export const STAGES: Exactly<typeof stages, StageName> = stages;

/** Server-sent events on /api/projects/{pid}/events. */
export type ProjectEvent =
  | { type: "decision"; data: DecisionRecord }
  | { type: "stage"; data: StageStatus }
  | { type: "job"; data: JobView }
  | { type: "resync"; data: Record<string, never> }
  | { type: "ping"; data: unknown };
