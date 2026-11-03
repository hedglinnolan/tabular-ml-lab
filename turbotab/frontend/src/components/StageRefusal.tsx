/**
 * A refusal a stage raised at fit time, as its status serves it (StageStatus.refusal; core/graph.py
 * `Refused`): the stage that refused (the stage itself, or the one it waits on), the engine's
 * words and its ways forward, on the RefusalNote pattern (record/Refusal.tsx). A way forward that
 * is a decision is a press that records it, and the stage computes again for the answers it then
 * reads; one that names no decision is said in words after the refusal's, since no press here can
 * carry it out. Where the card is the one place a choice is recorded (the quest shell's tapestry:
 * FOUNDATION §1, Stage's `recordHere`), the refusal is said as the stage says a preview's refusal
 * there (`besideCard`): no way forward is a press, and one that is a decision is not said either,
 * since the card offers it and its label names a choice ("`age` is an amount"), which said as a
 * sentence would read as a fact. No "Try again" (StageRetry): the refusal is for the current
 * answers, so running them again refuses again. No way forward confirms a column's role with other
 * readings in one press (BLUEPRINT §14 rule 2; besideAsk.ts `confirmsRoleInBlock`).
 */
import { Fragment, useState, type ReactNode } from "react";
import { isRefusalError } from "../api/client";
import { useDecide } from "../api/queries";
import type { Decision, Refusal, RefusalExit, StageStatus } from "../api/schema";
import { Prose } from "./Prose";
import { besideAsk, confirmsRoleInBlock } from "./record/besideAsk";
import { FailureNote, RefusalNote } from "./record/Refusal";
import r from "./record/Refusal.module.css";

/** A stage's refusal in words, as its status serves it. */
export type StageRefused = NonNullable<StageStatus["refusal"]>;

/** The refusal a stage in error serves: its own, or that of the stage it waits on that refused
 *  (`stage` names which); null for an error the engine did not type as a refusal, and for every
 *  other status. */
export function refusalOf(status: StageStatus | undefined): StageRefused | null {
  return status?.status === "error" ? (status.refusal ?? null) : null;
}

/** Its ways forward, each a label and a decision or none, as the engine serves them
 *  (core/graph.py `_way_forward`); one whose decision is in no shape the decision model reads is
 *  not offered, nor one that confirms a role with other readings. */
export function waysForward(refusal: StageRefused): RefusalExit[] {
  const ways: RefusalExit[] = [];
  for (const served of refusal.exits) {
    const { label, decision } = served;
    if (typeof label !== "string") continue;
    const decided = typeof decision === "object" && decision !== null && typeof (decision as { kind?: unknown }).kind === "string";
    if (decision != null && !decided) continue;
    const way = { label, decision: decided ? (decision as RefusalExit["decision"]) : null };
    if (!confirmsRoleInBlock(way)) ways.push(way);
  }
  return ways;
}

/** The refusal as said where the card is the one place a choice is recorded (Stage's `recordHere`
 *  false), as the stage says a preview's refusal there: its words up to the ledger's question about
 *  the readings, which the card asks (record/besideAsk.ts `besideAsk`), and only its ways forward
 *  that name no decision. */
export function besideCard(refusal: StageRefused): { message: string; told: RefusalExit[] } {
  const { code, message, exits } = refusal;
  return {
    message: besideAsk({ error: { code, message, exits: exits as RefusalExit[] } }).error.message,
    told: waysForward(refusal).filter((x) => !x.decision),
  };
}

/** What the server said when it did not record a way forward that was pressed. */
type NotTaken = { refusal: Refusal } | { failure: string };

/** Taking a way forward: its decision recorded as an answer, and what the server said if it
 *  refused it or failed (`notTaken`, until the next press or `dismiss`). */
function useWayForward(pid: string) {
  const decide = useDecide(pid);
  const [notTaken, setNotTaken] = useState<NotTaken | null>(null);
  const take = (decision: Decision) => {
    setNotTaken(null);
    decide.mutate(decision, {
      onError: (e) => setNotTaken(isRefusalError(e) ? { refusal: e.refusal } : { failure: e.message }),
    });
  };
  return { take, pending: decide.isPending, notTaken, dismiss: () => setNotTaken(null) };
}

/** "<lead> <the refusal's words> <after> <each way forward that is no decision>.", then a press
 *  for each way forward that is a decision; with `recordHere` false, the refusal's words as the
 *  card leaves them (`besideCard`) and no press. */
export function StageRefusal({
  pid,
  refusal,
  lead,
  after,
  recordHere = true,
  testId,
}: {
  pid: string;
  refusal: StageRefused;
  /** Said before the refusal's words, e.g. which work did not finish. */
  lead?: ReactNode;
  /** What waits on it, said after the refusal's words. */
  after?: ReactNode;
  /** False where the card is the one place a choice is recorded (Stage's `recordHere`). */
  recordHere?: boolean;
  testId?: string;
}) {
  const way = useWayForward(pid);
  const ways = waysForward(refusal);
  const presses = recordHere ? ways.filter((x) => x.decision) : [];
  const { message, told: said } = recordHere
    ? { message: refusal.message, told: ways.filter((x) => !x.decision) }
    : besideCard(refusal);
  return (
    <div
      className={r.refusal}
      role="alert"
      data-testid={testId ?? `stage-refusal-${refusal.stage}`}
      data-stage={refusal.stage}
      data-code={refusal.code}
      data-kind="refused"
    >
      <p className={r.text}>
        {lead ? <>{lead} </> : null}
        <Prose text={finish(message)} />
        {after ? <> {after}</> : null}
        {said.map((x) => (
          <Fragment key={x.label}>
            {" "}
            <Prose text={finish(x.label)} />
          </Fragment>
        ))}
      </p>
      {presses.length ? (
        <div className={r.exits}>
          {presses.map((x) => (
            <button
              key={x.label}
              type="button"
              className={r.exit}
              disabled={way.pending}
              onClick={() => way.take(x.decision as Decision)}
              data-testid="stage-refusal-exit"
            >
              <Prose text={x.label} />
            </button>
          ))}
        </div>
      ) : null}
      {way.notTaken ? (
        "refusal" in way.notTaken ? (
          <RefusalNote
            refusal={way.notTaken.refusal}
            onExit={(exit) => (exit.decision ? way.take(exit.decision as Decision) : way.dismiss())}
            onDismiss={way.dismiss}
          />
        ) : (
          <FailureNote message={way.notTaken.failure} onDismiss={way.dismiss} />
        )
      ) : null}
    </div>
  );
}

/** A sentence ends with its stop. */
export function finish(text: string): string {
  const t = text.trim();
  return /[.!?]$/.test(t) ? t : `${t}.`;
}
