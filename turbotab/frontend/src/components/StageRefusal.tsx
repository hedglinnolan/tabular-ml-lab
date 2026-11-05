/**
 * A refusal a stage raised at fit time, as its status serves it (StageStatus.refusal; core/graph.py
 * `Refused`): the stage that refused (the stage itself, or the one it waits on), the engine's
 * words and its ways forward, on the RefusalNote pattern (record/Refusal.tsx). A way forward that
 * is a decision is a press that records it, and the stage computes again for the answers it then
 * reads; one that names no decision is said in words after the refusal's, since no press here can
 * carry it out, unless it names the card it opens (`target`, record/target.ts) and the quest shell
 * opens cards: then it is a press that opens that card (G12). Where the card is the one place a
 * choice is recorded (the quest shell's tapestry: FOUNDATION §1, Stage's `recordHere`), the refusal
 * is said as the stage says a preview's refusal there (`besideCard`): no way forward is a press
 * that records, and one that is a decision is not said either, since the card offers it and its
 * label names a choice ("`age` is an amount"), which said as a sentence would read as a fact; a
 * way forward that opens its card records nothing, so it is a press there too, and so is one that
 * is a decision and names the card that records it (`target`): a press that opens that card, which
 * offers the same decision (`useBesideCard`; the owner's Q3: no dead end); one that names none
 * opens the card the quest log places its kind on (`cardOf`), unless that card is the one it is
 * drawn beside, which offers it already. No "Try again" (StageRetry): the refusal is for the
 * current answers, so running them again refuses again. No way forward confirms a column's role
 * with other readings in one press (BLUEPRINT §14 rule 2; besideAsk.ts `confirmsRoleInBlock`).
 */
import { Fragment, useContext, useState, type ReactNode } from "react";
import { isRefusalError } from "../api/client";
import { useDecide, useProjectView, useQuest } from "../api/queries";
import type { Decision, DecisionRecord, QuestLog, Refusal, RefusalExit, StageStatus } from "../api/schema";
import { Prose } from "./Prose";
import { besideAsk, confirmsRoleInBlock } from "./record/besideAsk";
import { FailureNote, RefusalNote } from "./record/Refusal";
import r from "./record/Refusal.module.css";
import { OpenCard, targetName, targetOf, triageTarget, useOpenTarget, type ExitTarget } from "./record/target";

/** A stage's refusal in words, as its status serves it. */
export type StageRefused = NonNullable<StageStatus["refusal"]>;

/** The refusal a stage in error serves: its own, or that of the stage it waits on that refused
 *  (`stage` names which); null for an error the engine did not type as a refusal, and for every
 *  other status. */
export function refusalOf(status: StageStatus | undefined): StageRefused | null {
  return status?.status === "error" ? (status.refusal ?? null) : null;
}

/** Its ways forward, each a label and a decision or none, as the engine serves them
 *  (core/graph.py `_way_forward`), with the card one that names no decision opens where it names
 *  one (`target`, record/target.ts); one whose decision is in no shape the decision model reads is
 *  not offered, nor one that confirms a role with other readings. */
export function waysForward(refusal: StageRefused): RefusalExit[] {
  const ways: RefusalExit[] = [];
  for (const served of refusal.exits) {
    const { label, decision } = served;
    if (typeof label !== "string") continue;
    const decided = typeof decision === "object" && decision !== null && typeof (decision as { kind?: unknown }).kind === "string";
    if (decision != null && !decided) continue;
    const target = targetOf(served);
    const way = { label, decision: decided ? (decision as RefusalExit["decision"]) : null, ...(target ? { target } : {}) };
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

/** A way forward drawn as a press that opens the card it names. */
export interface Opening {
  x: RefusalExit;
  target: ExitTarget;
  open: () => void;
}

/** Where the shell records the decisions it is shown: the quest log, whose `kinds` places each
 *  decision kind in its stage, with the project's records, which name the line each answers; and
 *  the stage whose card the refusal is drawn beside, where known (Fit's: Models). */
export interface Placing {
  log?: Pick<QuestLog, "kinds" | "stages"> | undefined;
  decisions?: readonly DecisionRecord[] | undefined;
  beside?: string | null;
}

/** The decision a way forward carries, as far as the card that records it is read from it. */
type Carried = { kind?: unknown; stage?: unknown; sweep?: unknown; column?: unknown; lines?: unknown };

/** The noticing a decision sets on the triage of the open noticings, its gate's confirmation of
 *  one line (`confirm_sweep`, sweep "noticings": core/fit_press.py's and core/seal.py's "Record as
 *  a limitation"), or null for its Confirm all (no lines) and every other decision. */
function triageLine(d: Carried): string | null {
  if (d.kind !== "confirm_sweep" || d.sweep !== "noticings" || !Array.isArray(d.lines) || d.lines.length !== 1) return null;
  const key = (d.lines[0] as { key?: unknown } | null)?.key;
  return typeof key === "string" ? key : null;
}

/**
 * The card a decision of `d`'s kind is recorded on, by the quest log (G12): the stage it places the
 * kind in (QuestLog.kinds, by which a methods sentence's change link opens it, QuestScreen
 * `openKind`), on the line that records it where one is known: a declaration's own line, keyed by
 * its kind, else the one line of that stage a record of that kind answers (on the same column,
 * where the decision names one); else the stage on its open line. Null where the log places the
 * kind nowhere, as for a sweep's confirmation, which sits in the stage it confirms (core/quest.py
 * `kind_stages`).
 */
export function kindTarget(d: Carried, log: Pick<QuestLog, "kinds" | "stages">, decisions: readonly DecisionRecord[] | undefined): ExitTarget | null {
  if (typeof d.kind !== "string") return null;
  const stage = log.kinds[d.kind];
  const at = stage === undefined ? undefined : log.stages.find((s) => s.key === stage);
  if (stage === undefined || !at) return null;
  if (at.lines.some((l) => l.source === "declaration" && l.key === d.kind)) return { stage, key: d.kind };
  const column = typeof d.column === "string" ? d.column : null;
  const kinds = new Map((decisions ?? []).map((r) => [r.id, r.decision as Carried]));
  const answered = at.lines.filter((l) => {
    const by = l.decision_id ? kinds.get(l.decision_id) : undefined;
    return by !== undefined && by.kind === d.kind && (column === null || by.column === column);
  });
  return { stage, key: answered.length === 1 ? answered[0]!.key : null };
}

/**
 * The card a way forward opens in the shell, or null. One that names no decision opens the card it
 * names (`target`, G12). One that is the triage's confirmation of one noticing and names the
 * triage's stage alone ("Record as a limitation") opens the triage there, on that noticing's line
 * (record/target.ts `triageTarget`), which records it. Any other decision opens the card it names,
 * else the card the quest log places its kind on (`kindTarget`), where that is not the card it is
 * drawn beside (`placing.beside`), which offers it already.
 */
export function cardOf(x: RefusalExit, placing: Placing = {}): ExitTarget | null {
  const target = targetOf(x);
  const d = (x.decision ?? null) as Carried | null;
  if (!d) return target;
  const noticing = triageLine(d);
  if (target && target.key === null && noticing !== null && d.stage === target.stage) return triageTarget(target.stage, noticing);
  if (target) return target;
  const placed = placing.log ? kindTarget(d, placing.log, placing.decisions) : null;
  return placed && placed.stage !== placing.beside ? placed : null;
}

/** The refusal as drawn where the card is the one place a choice is recorded (`recordHere`
 *  false): its words as the card leaves them (`besideCard`); the ways forward said in words after
 *  them (`said`: each that names no decision, where it opens no card here); and, in the order
 *  served, the presses that open a card (`to`), where the shell opens cards (record/target.ts
 *  `OpenCard`). A way forward that names no decision opens the card it names (G12). One that is a
 *  decision is no press that records here, but it is a press that opens the card that records it
 *  (`cardOf`), which offers that same decision: a press that opens a card records nothing
 *  (FOUNDATION §1). One that is a decision and opens no card here (its kind placed nowhere, or on
 *  the card it is drawn beside) is that card's to offer, and is neither a press nor a sentence: its
 *  label names a choice, which said would read as a fact. */
export function useBesideCard(refusal: StageRefused, placing: Placing = {}): { message: string; said: RefusalExit[]; to: Opening[] } {
  const open = useContext(OpenCard);
  const said: RefusalExit[] = [];
  const to: Opening[] = [];
  for (const x of waysForward(refusal)) {
    const target = open ? cardOf(x, placing) : null;
    if (open && target) to.push({ x, target, open: () => open(target) });
    else if (!x.decision) said.push(x);
  }
  return { message: besideCard(refusal).message, said, to };
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
 *  card leaves them (`besideCard`) and no press that records. A way forward that is no decision
 *  but names the card it opens is, where the shell opens cards, a press that opens it instead of a
 *  sentence (G12): it records nothing, so it is a press with `recordHere` false too, as is one that
 *  is a decision, which opens the card that records it (`useBesideCard`: the card it names, else
 *  the one the quest log places its kind on; no caller names the card it is drawn beside here, so
 *  that card's own kinds are presses too, which bring their line into view). */
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
  const opener = useOpenTarget();
  // Where a decision is recorded, read only where the shell opens cards and this card does not record.
  const places = useContext(OpenCard) !== null && !recordHere;
  const log = useQuest(pid, places).data;
  const decisions = useProjectView(pid).data?.decisions;
  const beside = useBesideCard(refusal, { log, decisions });
  const ways = waysForward(refusal);
  const presses = recordHere ? ways.filter((x) => x.decision) : [];
  const opens = ways.filter((x) => !x.decision).map((x) => ({ x, open: opener(x) }));
  const { message, said, to } = recordHere
    ? {
        message: refusal.message,
        said: opens.filter((o) => !o.open).map((o) => o.x),
        to: opens.flatMap(({ x, open }) => (open ? [{ x, open, target: targetOf(x)! }] : [])),
      }
    : beside;
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
      {presses.length || to.length ? (
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
          {to.map(({ x, open, target }) => (
            <button
              key={x.label}
              type="button"
              className={r.ghost}
              onClick={open}
              data-testid="stage-refusal-to"
              data-target={targetName(target)}
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
