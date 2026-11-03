/**
 * Work is visible (§04 job chip): every running computation shows a chip named in
 * plain language, with its progress and a way to stop it. The chips come from the
 * stage statuses the server pushes; each JobView supplies the label.
 */
import { useCancelJob, useJobs } from "../api/queries";
import type { StageStatus } from "../api/schema";
import styles from "./JobChips.module.css";

/**
 * Each stage's label as the engine's graph declares it (turbotab/core/stages/__init__.py
 * `build_graph`), for every stage: a JobView serves the same words (`label`) while the stage runs,
 * so these say the work before a JobView has arrived and for a stage that is not running, whose
 * status serves no label. JobChips.test.tsx holds them to the engine's (fixtures/stages.json,
 * captured by fixtures/capture_stages.py; capture it again when a stage is added or relabeled).
 */
export const STAGE_LABEL: Record<string, string> = {
  ingest: "Reading the file",
  oriented: "Reading which way round the table is",
  profile: "Summarizing every column",
  findings: "Checking the table against the chosen lenses",
  structure: "Reading how the rows repeat",
  working: "Building the working table",
  target_info: "Reading the outcome column",
  roles: "Reading what each column is",
  proposals: "Looking up what the field usually does",
  cohort: "Counting who is in the analysis",
  split: "Drawing the held-out rows",
  shelf: "Ranking the model families for this table",
  design: "Building each model's pipeline",
  fit: "Fitting the models",
  substitution: "Drawing the substitution curves",
  seal_plan: "Reading what a held-out set can measure",
  sensitivity: "Refitting the model on each analysis's rows",
  calibration: "Correcting intakes for day-to-day error in the recalls",
  secondary: "Fitting the model further adjusted for the declared covariates",
  scales: "Estimating each scale's reliability",
  usual_intake: "Estimating usual-intake distributions",
  effects: "Reporting the effect of what you study across the declared models",
  causal_design: "Reading the causal lane's assumptions and overlap",
  causal: "Estimating the effect in the causal lane",
  time_varying: "Following what you study through time",
  explain: "Explaining each fitted model",
  forms: "Reading which continuous terms take a declared form",
  modification: "Estimating the declared effect modification",
  explore: "Exploring the rows Explore may read",
  evaluation: "Fitting the benchmark and weighing the models",
  mattered: "Refitting the estimate with each recorded choice changed",
};

/** A stage's work in words: its label, or for a stage with none here its name ("time varying"). */
export function stageLabel(stage: string): string {
  return STAGE_LABEL[stage] ?? stage.replace(/_/g, " ");
}

export function JobChips({ pid, stages }: { pid: string; stages: Record<string, StageStatus> }) {
  const active = Object.values(stages).filter(
    (s) => (s.status === "queued" || s.status === "running") && s.job_id,
  );
  const jobs = useJobs(
    pid,
    active.map((s) => s.job_id!),
  );
  const cancel = useCancelJob(pid);
  if (active.length === 0) return null;
  return (
    <ul className={styles.list} aria-label="Work in progress">
      {active.map((s, i) => {
        const job = jobs[i]?.data;
        const label = job?.label ?? stageLabel(s.stage);
        const progress = job?.progress ?? s.progress;
        const pct = progress === null || progress === undefined ? null : Math.round(progress * 100);
        return (
          <li key={s.job_id} className={styles.chip} data-testid={`job-${s.stage}`}>
            <span className={styles.label}>
              {label}
              {s.status === "queued" ? <span className={styles.state}> · queued</span> : null}
            </span>
            <span
              className={styles.bar}
              role="progressbar"
              aria-label={label}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={pct ?? undefined}
            >
              <i style={{ width: `${pct ?? 0}%` }} />
            </span>
            <span className={styles.pct}>{pct === null ? "…" : `${pct}%`}</span>
            <button
              type="button"
              className={styles.cancel}
              onClick={() => cancel.mutate(s.job_id!)}
              aria-label={`Cancel: ${label}`}
              title="Stops this computation. Results already on screen stay, marked stale, with a way to recompute them."
            >
              Cancel
            </button>
          </li>
        );
      })}
    </ul>
  );
}
