/**
 * Work is visible (§04 job chip): every running computation shows a chip named in
 * plain language, with its progress and a way to stop it. The chips come from the
 * stage statuses the server pushes; each JobView supplies the label.
 */
import { useCancelJob, useJobs } from "../api/queries";
import type { StageStatus } from "../api/schema";
import styles from "./JobChips.module.css";

/** Used until the JobView (with the server's own label) has arrived. */
export const STAGE_LABEL: Record<string, string> = {
  ingest: "Reading the file into columnar storage",
  profile: "Summarizing every column",
  target_info: "Reading the outcome column",
  findings: "Checking the table against the chosen lenses",
  roles: "Reading what each column is",
  proposals: "Looking up what the field usually does",
  cohort: "Counting who is in the analysis",
  split: "Drawing the held-out rows",
  shelf: "Ranking the model families for this table",
  design: "Building each model's pipeline",
  fit: "Fitting the models",
  substitution: "Drawing the substitution curves",
};

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
        const label = job?.label ?? STAGE_LABEL[s.stage] ?? s.stage;
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
