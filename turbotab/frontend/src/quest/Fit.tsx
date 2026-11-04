/**
 * Fit at the analysis flowchart's end (FOUNDATION §7; plans/W1_PART2_PLAN.md ruling 3). Before it is
 * pressed: Fit, with how long a held fit is expected to take in the engine's words, and the lock's
 * reason beneath; no Cancel, since nothing runs. While the fit job runs: its progress and a Cancel
 * where Fit was (POST /fit/cancel). Pressing is a job command, not a decision. A fit the engine
 * refused in words (StageStatus.refusal: its own, or the refusal of the stage it waits on) is said
 * beneath Fit as the shell's stage says a refusal (StageRefusal's `besideCard`), and so is a press
 * of Fit the engine refused (`refusedPress`): no way forward is a press, and one that is a decision
 * is not said either, since in the quest shell a choice is recorded on the card. A press's refusal
 * is said while what it answered stands (`PressedOn`), and not again where it is the fit's own.
 */
import { Fragment, useState } from "react";
import { isRefusalError } from "../api/client";
import { useCancelFit, useJobs, usePressFit, useProjectView } from "../api/queries";
import type { DecisionRecord, QuestLog, Refusal, StageStatus } from "../api/schema";
import { stageLabel } from "../components/JobChips";
import { besideCard, finish, refusalOf, type StageRefused } from "../components/StageRefusal";
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
  const decisions = useProjectView(pid).data?.decisions;
  const [pressedOn, setPressedOn] = useState<PressedOn | null>(null);
  const job = useJobs(pid, fit?.job_id && (fit.status === "running" || fit.status === "queued") ? [fit.job_id] : [])[0]?.data;
  const pos = fitPosition(log, fit);
  const refused = refusalOf(fit);
  const stands = pressedOn !== null && pressedOn.decisions === decisions && pressedOn.refused === refused;
  const pressSaid = press.error
    ? isRefusalError(press.error)
      ? stands
        ? refusedPress(press.error.refusal, refused)
        : null
      : press.error.message
    : null;
  const said =
    pressSaid ??
    (cancel.error ? (isRefusalError(cancel.error) ? cancel.error.refusal.error.message : cancel.error.message) : null);

  if (pos.kind === "press") {
    return (
      <div className={q.fit} data-kit data-testid="fit-position" data-held={pos.held || undefined}>
        <button
          type="button"
          className={k.primary}
          onClick={() => {
            setPressedOn({ decisions, refused });
            press.mutate();
          }}
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

/** What a press of Fit was answered on: the project's decisions and the fit's refusal as they
 *  stood when it was pressed. A refused press's ways past are decisions (the triage's Confirm all
 *  among them, drawn at Models' end on this same screen) or a change in the fit's refusal (a stage
 *  run again), so the press's refusal is said while both stand and no longer once either moves;
 *  Fit is pressed again to ask again. The quest log is not compared: a noticing the triage
 *  records to act on stays open there for its own decision (core/sweep.py `cover_noticings`), so
 *  the log can read the same after the triage is confirmed, and a stage finishing beside the press
 *  moves the log with no decision recorded. The query cache keeps a value the same while a read
 *  brings it unchanged, and the fit's refusal with it while only its progress moves
 *  (api/events.ts `applyJob`). */
interface PressedOn {
  decisions: DecisionRecord[] | undefined;
  refused: StageRefused | null;
}

/** A refused press of Fit, said beneath it as the fit's own refusal is (`FitRefusal`): its words
 *  and each way forward that names no decision, in sentences, none a press. Under Estimate and
 *  Describe the open noticings' triage holds the press until it is confirmed (core/fit_press.py
 *  `triage_refusal`): its way forward that names no decision says where the triage is (the shell
 *  draws it at Models' end), and its "Keep the recommended dispositions", a decision, is the
 *  triage card's Confirm all to record, not said here. A refusal with no way forward is its words
 *  alone. A press refused with the fit's own refusal, its code and words those of the refusal the
 *  fit's status serves (`shown`), is not said again: `FitRefusal` says it. */
function refusedPress(refusal: Refusal, shown: StageRefused | null): string | null {
  if (shown && shown.code === refusal.error.code && shown.message === refusal.error.message) return null;
  const { message, told } = besideCard({ stage: "fit", ...refusal.error });
  return told.length ? [finish(message), ...told.map((x) => finish(x.label))].join(" ") : message;
}

/** The fit's refusal beneath Fit, as Fit's other lines are said there: "<the refusal's words>
 *  <each way forward that names no decision>." Fit sits in the tapestry, which records nothing
 *  (FOUNDATION §1: the card is the one place a choice is recorded; Stage's `recordHere`), so the
 *  refusal is said as the card leaves it (`besideCard`): a decision is the card's to offer, and its
 *  label names a choice, which said as a sentence would read as a fact. */
function FitRefusal({ refusal }: { refusal: StageRefused }) {
  const { message, told } = besideCard(refusal);
  return (
    <p
      className={q.fitReason}
      role="alert"
      data-testid="fit-refusal"
      data-stage={refusal.stage}
      data-code={refusal.code}
    >
      <Plain text={finish(message)} />
      {told.map((x) => (
        <Fragment key={x.label}>
          {" "}
          <Plain text={finish(x.label)} />
        </Fragment>
      ))}
    </p>
  );
}
