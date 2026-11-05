/**
 * Fit at the analysis flowchart's end (FOUNDATION §7; plans/W1_PART2_PLAN.md ruling 3). Before it is
 * pressed: Fit, with how long a held fit is expected to take in the engine's words, and the lock's
 * reason beneath; no Cancel, since nothing runs. While the fit job runs: its progress and a Cancel
 * where Fit was (POST /fit/cancel). Pressing is a job command, not a decision. A fit the engine
 * refused in words (StageStatus.refusal: its own, or the refusal of the stage it waits on) is said
 * beneath Fit as the shell's stage says a refusal (StageRefusal's `useBesideCard`), and so is a
 * press of Fit the engine refused (`refusedPress`): no way forward is a press that records, since
 * in the quest shell a choice is recorded on the card. One that names the card it opens is a press
 * that opens it, a decision's included, which opens the card that records it (the owner's Q3: no
 * dead end); a decision that names no card is not said, and any other way forward is said in
 * words. A press's refusal is said while what it answered stands (`PressedOn`), and not again where
 * it is the fit's own.
 */
import { Fragment, useState } from "react";
import { isRefusalError } from "../api/client";
import { useCancelFit, useJobs, usePressFit, useProjectView } from "../api/queries";
import type { DecisionRecord, QuestLog, Refusal, StageStatus } from "../api/schema";
import { stageLabel } from "../components/JobChips";
import { targetName } from "../components/record/target";
import { finish, refusalOf, useBesideCard, type StageRefused } from "../components/StageRefusal";
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
  const pressRefused = press.error && isRefusalError(press.error) && stands ? refusedPress(press.error.refusal, refused) : null;
  const pressSaid = press.error && !isRefusalError(press.error) ? press.error.message : null;
  const said = pressRefused
    ? null
    : (pressSaid ??
      (cancel.error ? (isRefusalError(cancel.error) ? cancel.error.refusal.error.message : cancel.error.message) : null));

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
        {pressRefused ? <Beside refusal={pressRefused} /> : null}
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
        {pressRefused ? <Beside refusal={pressRefused} /> : null}
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

/** A refused press of Fit, to be said beneath it as the fit's own refusal is (`Beside`), or null
 *  where it is the fit's own: a press refused with the refusal the fit's status serves (`shown`),
 *  its code and words the same, is not said again, since `FitRefusal` says it. Under Estimate and
 *  Describe the open noticings' triage holds the press (core/fit_press.py `triage_refusal`): until
 *  it is confirmed, its way forward to the triage opens the card it is drawn on (Models, at its
 *  end), and its "Keep the recommended dispositions", a decision that names no card, is that
 *  card's Confirm all, not said here; while a noticing kept to act on is not acted on, its way
 *  forward to where it is acted on opens that card, and its "Record as a limitation", a decision,
 *  opens the triage's card, which records it (TriageCard). */
function refusedPress(refusal: Refusal, shown: StageRefused | null): StageRefused | null {
  if (shown && shown.code === refusal.error.code && shown.message === refusal.error.message) return null;
  return { stage: "fit", ...refusal.error };
}

/** The fit's refusal beneath Fit, as Fit's other lines are said there (`Beside`), its words always
 *  ending with their stop. */
function FitRefusal({ refusal }: { refusal: StageRefused }) {
  return <Beside refusal={refusal} testId="fit-refusal" finished />;
}

/** A refusal beneath Fit: "<the refusal's words> <each way forward said in words>", then a press
 *  for each way forward that opens a card. Fit sits in the tapestry, which records nothing
 *  (FOUNDATION §1: the card is the one place a choice is recorded; Stage's `recordHere`), so the
 *  refusal is said as the card leaves it (StageRefusal's `useBesideCard`): no press records; one
 *  that names the card it opens opens it where the shell opens cards, a decision's the card that
 *  records it; a decision that names no card is the card's to offer, and its label names a choice,
 *  which said as a sentence would read as a fact. The words end with their stop wherever something
 *  follows them, and always with `finished`; otherwise they are the refusal's as served. */
function Beside({ refusal, testId, finished = false }: { refusal: StageRefused; testId?: string; finished?: boolean }) {
  const { message, said, to } = useBesideCard(refusal);
  return (
    <p
      className={q.fitReason}
      role="alert"
      data-testid={testId}
      data-stage={refusal.stage}
      data-code={refusal.code}
    >
      <Plain text={finished || said.length || to.length ? finish(message) : message} />
      {said.map((x) => (
        <Fragment key={x.label}>
          {" "}
          <Plain text={finish(x.label)} />
        </Fragment>
      ))}
      {to.map(({ x, target, open }) => (
        <Fragment key={x.label}>
          {" "}
          <button type="button" className={k.linkish} onClick={open} data-testid="fit-to" data-target={targetName(target)}>
            <Plain text={x.label} />
          </button>
        </Fragment>
      ))}
    </p>
  );
}
