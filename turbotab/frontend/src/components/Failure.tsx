/**
 * Saying that a stage did not finish, truly and with a way forward (DRIVE_RUBRIC §2.1–2.3):
 * which computation stopped, why, in plain words, and two levers — run it again for the current
 * answers, or change the answer that led there. A stage that waits on a failed one names the one
 * that failed, never "Needs 'design', which failed."
 *
 * Changing an answer is the Record's to do: `requestReopen` asks it, from anywhere (the stage, the
 * banner), with a window event the Record listens for — no shared mutable state.
 */
import type { ReactNode } from "react";
import type { QuestionKey } from "../api/m1-types";
import type { DecisionRecord, ProjectView, StageStatus } from "../api/schema";
import { STAGE_LABEL } from "./JobChips";
import { Prose } from "./Prose";
import { StageRetry } from "./StageRetry";
import s from "./Failure.module.css";

/** The decision slots each stage reads (turbotab/core/stages/__init__.py). */
const READS: Record<string, QuestionKey[]> = {
  target_info: ["target", "task"],
  findings: ["lens", "target"],
  roles: ["lens", "target"],
  proposals: ["lens", "roles", "target"],
  cohort: ["target", "roles", "exclusions", "missing"],
  split: ["split", "roles", "task"],
  shelf: ["purpose", "task", "roles"],
  design: ["roles", "energy_adjustment", "missing", "models", "purpose"],
  fit: ["models", "purpose", "task"],
  substitution: ["substitution"],
};

const SUBJECT: Partial<Record<QuestionKey, string>> = {
  lens: "the lens",
  target: "the outcome",
  task: "the task",
  purpose: "the purpose",
  roles: "the column roles",
  exclusions: "the exclusions",
  missing: "the missing values",
  split: "the split",
  energy_adjustment: "the energy adjustment",
  models: "the models",
  substitution: "the substitution",
};

const NEEDS = /^Needs '([a-z_]+)', which failed\.?$/;

export interface Failure {
  /** The stage that failed first (the one a dependent's error names). */
  stage: string;
  status: StageStatus;
  /** Its reason in plain words: no exception class names. */
  message: string;
}

/** An error as a reader should see it: "EnergyAdjustmentNotApplicable: X" says "X". */
export function plainError(error: string | null | undefined): string {
  if (!error) return "the server gave no reason";
  return error
    .replace(/^[A-Z][A-Za-z]+(?:Error|Exception|NotApplicable|Exceeded|Warning)?: /, "")
    .trim();
}

/** The failure behind `stage` (itself, or the upstream stage its error names); null if none. */
export function rootFailure(
  stages: Record<string, StageStatus | undefined>,
  stage: string,
): Failure | null {
  let name = stage;
  for (let i = 0; i < 12; i++) {
    const status = stages[name];
    if (!status || status.status !== "error") return null;
    const m = NEEDS.exec(status.error ?? "");
    if (!m || !stages[m[1]!]) return { stage: name, status, message: plainError(status.error) };
    name = m[1]!;
  }
  return null;
}

/** The answer most likely behind a stage's failure: the newest recorded answer it reads. */
export function causeOf(
  view: Pick<ProjectView, "decisions" | "interview">,
  stage: string,
): QuestionKey | null {
  const reads = READS[stage] ?? [];
  const live = new Set(view.interview.filter((st) => st.status === "answered").map((st) => st.key));
  let best: { key: QuestionKey; seq: number } | null = null;
  for (const r of view.decisions as DecisionRecord[]) {
    const key = slotKey(r);
    if (!key || !reads.includes(key) || !live.has(key)) continue;
    if (!best || r.seq > best.seq) best = { key, seq: r.seq };
  }
  return best?.key ?? null;
}

function slotKey(r: DecisionRecord): QuestionKey | null {
  const k = r.decision.kind;
  if (k === "revert") return null;
  if (k === "select_models") return "models";
  return k.replace(/^set_/, "") as QuestionKey;
}

export const REOPEN_EVENT = "turbotab:reopen";

/** Ask the Record to reopen a question (it scrolls there and takes the keyboard focus). */
export function requestReopen(key: QuestionKey): void {
  window.dispatchEvent(new CustomEvent<QuestionKey>(REOPEN_EVENT, { detail: key }));
}

/**
 * "Building each model's pipeline did not finish: <why>. [Try again] [Change the energy
 * adjustment]". For a stage the user stopped: "You stopped …" and [Recompute].
 */
export function StageFailure({
  pid,
  view,
  stage,
  lead,
  after,
  compact = false,
  changeable = true,
  testId,
}: {
  pid: string;
  view: Pick<ProjectView, "stages" | "decisions" | "interview">;
  stage: string;
  /** What could not be shown because of it, e.g. "No model was fitted". */
  lead?: ReactNode;
  /** What waits on it, said after the reason. */
  after?: ReactNode;
  /** Offer to change the answer behind it (off where its own "change" is beside it). */
  changeable?: boolean;
  compact?: boolean;
  testId?: string;
}) {
  const failure = rootFailure(view.stages, stage);
  const status = view.stages[stage];
  if (!failure && !status?.cancelled) return null;
  const at = failure?.stage ?? stage;
  const cause = causeOf(view, at);
  const work = STAGE_LABEL[at] ?? at.replace(/_/g, " ");
  return (
    <div
      className={compact ? s.compact : s.failure}
      role="alert"
      data-testid={testId ?? `failure-${stage}`}
      data-stage={at}
      data-kind={failure ? "failed" : "stopped"}
    >
      <p className={s.text}>
        {lead ? <>{lead} </> : null}
        {failure ? (
          <>
            <strong className={s.what}>{work} did not finish:</strong>{" "}
            <Prose text={finish(failure.message)} />
          </>
        ) : (
          <>
            <strong className={s.what}>You stopped {lowerFirst(work)}</strong> before it finished.
          </>
        )}
        {after ? <> {after}</> : null}
      </p>
      <div className={s.actions}>
        <StageRetry pid={pid} status={failure ? failure.status : status} />
        {cause && changeable ? (
          <button
            type="button"
            className={s.change}
            onClick={() => requestReopen(cause)}
            data-testid={`failure-change-${cause}`}
          >
            Change {SUBJECT[cause] ?? cause.replace(/_/g, " ")}
          </button>
        ) : null}
      </div>
    </div>
  );
}

function finish(text: string): string {
  const t = text.trim();
  return /[.!?]$/.test(t) ? t : `${t}.`;
}

function lowerFirst(text: string): string {
  return text.charAt(0).toLowerCase() + text.slice(1);
}
