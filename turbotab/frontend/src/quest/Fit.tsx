/**
 * Fit at the analysis flowchart's end (FOUNDATION §7; plans/W1_PART2_PLAN.md ruling 3). Before it is
 * pressed: Fit, with how long a held fit is expected to take in the engine's words, and the lock's
 * reason beneath; no Cancel, since nothing runs. While the fit job runs: its progress and a Cancel
 * where Fit was (POST /fit/cancel). Pressing is a job command, not a decision. A fit the engine
 * refused in words (StageStatus.refusal: its own, or the refusal of the stage it waits on) is said
 * beneath Fit with its ways forward in words (StageRefusal's), none of them a press: in the quest
 * shell a choice is recorded on the card.
 */
import { Fragment } from "react";
import { isRefusalError } from "../api/client";
import { useCancelFit, useJobs, usePressFit } from "../api/queries";
import type { QuestLog, StageStatus } from "../api/schema";
import { stageLabel } from "../components/JobChips";
import { finish, refusalOf, waysForward, type StageRefused } from "../components/StageRefusal";
import { Plain, kit as k } from "../kit";
import { fitPosition } from "./log";
import q from "./quest.module.css";

export function FitControl({
  pid,
  log,
  fit,
}: {
  pid: string;
  log: QuestLog;
  fit: StageStatus | undefined;
}) {
  const press = usePressFit(pid);
  const cancel = useCancelFit(pid);
  const job = useJobs(pid, fit?.job_id && (fit.status === "running" || fit.status === "queued") ? [fit.job_id] : [])[0]?.data;
  const pos = fitPosition(log, fit);
  const failure = press.error ?? cancel.error;
  const said = failure ? (isRefusalError(failure) ? failure.refusal.error.message : failure.message) : null;
  const refused = refusalOf(fit);

  if (pos.kind === "press") {
    return (
      <div className={q.fit} data-kit data-testid="fit-position" data-held={pos.held || undefined}>
        <button
          type="button"
          className={k.primary}
          onClick={() => press.mutate()}
          disabled={press.isPending}
          data-testid="fit"
        >
          {pos.estimate ? `Fit · ${pos.estimate}` : "Fit"}
        </button>
        {log.fit?.reason ? (
          <p className={q.fitReason}>
            <Plain text={log.fit.reason} />
          </p>
        ) : null}
        {refused ? <FitRefusal refusal={refused} /> : null}
        {said ? (
          <p className={q.fitReason} role="alert">
            <Plain text={said} />
          </p>
        ) : null}
      </div>
    );
  }
  if (pos.kind === "running") {
    const progress = job?.progress ?? pos.progress;
    const pct = progress === null || progress === undefined ? null : Math.round(progress * 100);
    const label = job?.label ?? stageLabel("fit");
    return (
      <div className={q.fit} data-kit data-testid="fit-position" data-running>
        <span className={q.fitLabel}>
          {label}
          {pos.estimate ? ` · ${pos.estimate}` : ""}
        </span>
        <span
          className={q.fitBar}
          role="progressbar"
          aria-label={label}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={pct ?? undefined}
          data-testid="fit-progress"
        >
          <i style={{ width: `${pct ?? 0}%` }} />
        </span>
        <span className={q.fitPct}>{pct === null ? "…" : `${pct}%`}</span>
        <button
          type="button"
          className={k.linkish}
          onClick={() => cancel.mutate()}
          disabled={cancel.isPending}
          data-testid="fit-cancel"
        >
          Cancel
        </button>
        {said ? (
          <p className={q.fitReason} role="alert">
            <Plain text={said} />
          </p>
        ) : null}
      </div>
    );
  }
  return null;
}

/** The fit's refusal beneath Fit, as Fit's other lines are said there: "<the refusal's words>
 *  <each way forward>." Fit sits in the tapestry, which records nothing (FOUNDATION §1: the card
 *  is the one place a choice is recorded; Stage's `recordHere`), so each way forward is said in
 *  words and none is a press. */
function FitRefusal({ refusal }: { refusal: StageRefused }) {
  return (
    <p
      className={q.fitReason}
      role="alert"
      data-testid="fit-refusal"
      data-stage={refusal.stage}
      data-code={refusal.code}
    >
      <Plain text={finish(refusal.message)} />
      {waysForward(refusal).map((x) => (
        <Fragment key={x.label}>
          {" "}
          <Plain text={finish(x.label)} />
        </Fragment>
      ))}
    </p>
  );
}
