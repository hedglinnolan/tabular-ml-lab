/**
 * What the six Predict declaration cards share: the frame, the recording, and the engine's words.
 *
 * A card records as the existing decision hook (`useDecide`) does, with the reason the person gave
 * for the answer (ReasonBox, above the card's record button) as the decision's note, and answers a
 * press the server does not record the way the Record's plain wiring does: the refusal and its
 * exits, or the failure, at the option pressed, and a note the server refuses as too long at the
 * reason box, which names it the decision's note as the refusal does. It states only what the
 * engine serves: a line of
 * the quest log (`reason`, `would_change`, `changes_nothing`), the preview the engine serves for
 * the option pointed at (the caption of its first view, else its note, else its caution), or the
 * refusal the preview meets. A card with nothing served to say about an option names the option and
 * stops; the reason on record is the person's own words, as recorded.
 *
 * A press the card will not send (a number outside what the model takes, a column still to choose)
 * is answered at the option too, in one sentence for each field in the way, so no press is silent.
 *
 * Each card records from its own visible button, as today's cards do (finding 8 of the 2026-10-17
 * walkthrough): a click on an option chooses it, marked "chosen" until it is recorded, and the
 * button under the options records the choice; Enter on an option records it at once, as before.
 */
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useId, useState, type MouseEvent, type ReactNode } from "react";
import { api, isRefusalError } from "../../../api/client";
import {
  keys,
  logDerivedKeys,
  mergeView,
  useColumnSummaries,
  useProjectView,
  useQuest,
  useStageResult,
} from "../../../api/queries";
import type {
  ColumnSummary,
  Decision,
  DecisionRecord,
  ProjectState,
  ProjectView,
  QuestLine,
  QuestLog,
  Refusal,
} from "../../../api/schema";
import { Actions, RecordButton } from "../../../components/record/ask/common";
import { Options, type OptionItem, type OptionTag } from "../../../components/record/Options";
import { FailureNote, RefusalNote } from "../../../components/record/Refusal";
import { Prose } from "../../../components/Prose";
import { decisionKey, stateSeq, usePreview, type Answer } from "../../../components/stage/data";
import { useStageFocus, type StageFocus } from "../../../state/focus";
import { NO_REASON, ReasonBox, noteOf, type Reason } from "../ReasonBox";
import c from "./declaration.module.css";

/** The engine's words for the option pointed at, tagged with the decision they answer. */
export interface Words {
  key: string;
  text: string;
  refused: boolean;
}

export function wordsOf(answer: Answer): Words | null {
  const key = decisionKey(answer.decision);
  if (answer.refusal) return { key, text: answer.refusal.error.message, refused: true };
  const { views, note, caution } = answer.result;
  const text = views[0]?.caption || note || caution?.text || null;
  return text ? { key, text, refused: false } : null;
}

interface PlainAnswer {
  at: string;
  refusal?: Refusal;
  failure?: string;
}

export interface CardWiring {
  /** The project's state, once the view has loaded. */
  state: ProjectState | null;
  /** The table's columns, in the table's order. */
  columns: string[];
  /** Each column's summary (its type and its count of distinct values), by name. */
  summaries: ReadonlyMap<string, ColumnSummary>;
  /** The outcome's task as the engine reads it: the one recorded, else the one the target stage
   *  detected for the outcome (the engine's `state.task or detected_task`); null while neither is. */
  task: ProjectState["task"];
  pending: boolean;
  /** Record a decision; `at` names the option a refusal answers under. */
  record: (decision: Decision, at: string) => void;
  /** The option last pressed that the card would not send, answered until it can be sent. */
  held: string | null;
  hold: (at: string) => void;
  answerAt: { key: string; node: ReactNode } | null;
  words: Words | null;
  /** The reason being written for the next record, sent with it as the decision's note. */
  reason: Reason;
  setReason: (reason: Reason) => void;
  /** The note on the record that answers what the card shows as recorded, or null: none where the
   *  card marks no option (`marked` null); for a keyed kind, the record of the entry its marked
   *  option's decision writes; for a kind that keeps one answer, the kind's record in force, the
   *  one the line names, where the card marks an option or draws its button over several choices
   *  (`marked` left out), and none once a later record of another kind has rewritten its slot. */
  reasonOnRecord: (marked?: Marked) => string | null;
}

/** What a card marks as recorded, for the reason shown beside it: the option it marks, with the
 *  decision that option sends now (null where its offer has problems now), or null where it marks
 *  none (the value on record is not among the values it lists). */
export type Marked = { decision: Decision | null } | null;

/** Where a note the server refuses as too long is answered: at the reason box, which holds the text
 *  to shorten and names it as the refusal does (core/decisions.py `check_note`), not at the option
 *  pressed. */
const AT_REASON = "reason";
const NOTE_TOO_LONG = "note_too_long";

/** Record a decision with the reason given for it (`?note=`, none where none is given), as
 *  `useDecide` records one (queries.ts): the view the server answers with merged into the cache,
 *  and the reads that follow the decision log refetched. `useDecide` takes no note. */
function useDecideWithNote(pid: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ decision, note }: { decision: Decision; note: string | null }) =>
      note === null ? api.decide(pid, decision) : api.decide(pid, decision, { note }),
    onSuccess: (view: ProjectView) => {
      qc.setQueryData<ProjectView>(keys.view(pid), (prev) => mergeView(prev, view));
      for (const key of logDerivedKeys(pid)) void qc.invalidateQueries({ queryKey: key, exact: true });
    },
  });
}

/** The entry a decision of a keyed kind writes, or undefined for a kind that keeps one answer:
 *  core/decisions.py `register_kind`'s `key`, under which "each decision writes one entry (the
 *  latest write per key wins, and a revert restores that key's earlier entry)". Of the kinds a card
 *  here records with a reason, usual intake alone is keyed: one answer per dietary component. */
function entryOf(decision: Decision): string | undefined {
  return decision.kind === "set_usual_intake" ? decision.nutrient : undefined;
}

/** The records in force, in order: every one but those a revert cancels (core/decisions.py
 *  `reverted`: latest first, a revert counts only where nothing later reverts it). */
function live(records: readonly DecisionRecord[]): DecisionRecord[] {
  const ordered = [...records].sort((a, b) => a.seq - b.seq);
  const cancelled = new Set<string>();
  for (const r of [...ordered].reverse()) {
    if (!cancelled.has(r.id) && r.decision.kind === "revert") cancelled.add(r.decision.decision_id);
  }
  return ordered.filter((r) => !cancelled.has(r.id));
}

/** The other kinds whose later live record writes a card's kind's slot whole, so that the answer
 *  the card marks is that record's, not the kind's own (core/decisions.py `register_kind`): the
 *  split holds `set_validation`'s scheme written onto the draw, and `set_split`, a new draw (the
 *  holdout caution's "Cross-validation only" among them, core/row_previews.py), writes the split
 *  with a scheme of its own. */
const REWRITTEN_BY: Partial<Record<Decision["kind"], readonly Decision["kind"][]>> = {
  set_validation: ["set_split"],
};

/** The note on the record that answers what the card shows as recorded, or null. A record answers
 *  only while the quest log names one for the declaration of `kind` (core/quest.py: a declaration's
 *  line is keyed by its kind, and names a record once answered); the log is the one the shell
 *  holds, and the card never fetches it. The record it names is the last live record of the kind
 *  (core/quest.py `last_answer` over `_Log.live`), read here from the view, as the card reads what
 *  it marks: so a new answer's own note is beside it from the decision's response on, not the note
 *  of the record it replaced, which the log names until the shell refetches it. None answers where
 *  the card marks no option (`marked` null). For a kind that keeps one answer, that record is the
 *  answer to the option marked, or to the several choices a card draws one button over (`marked`
 *  left out), while it is the last live record to write its slot: none answers once a later record
 *  of another kind has written the slot whole (`REWRITTEN_BY`), since what the card marks is then
 *  that record's answer, though the log goes on naming the kind's own record. For a keyed kind it
 *  is the last entry written, whatever entry the card shows, so the record that answers is the last
 *  live one of the entry the marked option's decision writes, and there is none where that decision
 *  is not known. */
function noteOnRecord(
  view: ProjectView | undefined,
  log: QuestLog | undefined,
  kind: Decision["kind"],
  marked: Marked | undefined,
): string | null {
  const line = log?.stages.flatMap((s) => s.lines).find((l) => l.source === "declaration" && l.key === kind);
  if (!view || !line?.decision_id || marked === null) return null;
  const records = live(view.decisions);
  const kept = [...records].reverse().filter((r) => r.decision.kind === kind);
  const last = kept[0];
  if (!last) return null;
  if (entryOf(last.decision) === undefined) {
    const others = REWRITTEN_BY[kind] ?? [];
    const rewritten = records.some((r) => r.seq > last.seq && others.includes(r.decision.kind));
    return rewritten ? null : (last.note ?? null);
  }
  const entry = marked?.decision ? entryOf(marked.decision) : undefined;
  if (entry === undefined) return null;
  return kept.find((r) => entryOf(r.decision) === entry)?.note ?? null;
}

/** The wiring of one card: its project's state, its recording, and the preview of the option of
 *  its own kind that is pointed at (the Stage previews the same option from the same cache). */
export function useCard(pid: string, kind: Decision["kind"]): CardWiring {
  const view = useProjectView(pid).data;
  const log = useQuest(pid, false).data;
  const summaries = useColumnSummaries(pid, view?.stages.ingest?.status === "fresh").data;
  const targetStatus = view?.stages.target_info?.status;
  const target = useStageResult(pid, "target_info", targetStatus !== undefined && targetStatus !== "idle" && targetStatus !== "blocked").data?.artifact;
  const decide = useDecideWithNote(pid);
  const [plain, setPlain] = useState<PlainAnswer | null>(null);
  const [held, setHeld] = useState<string | null>(null);
  const [reason, setReason] = useState<Reason>(NO_REASON);
  const { focus } = useStageFocus();
  const pointed =
    focus.kind === "option" && focus.decision.kind === kind
      ? { decision: focus.decision, label: focus.label }
      : null;
  const preview = usePreview(pid, pointed, view ? stateSeq(view) : 0);

  // Every way the card records (its button, Enter on an option, a refusal's exit) sends the reason
  // written for it; once recorded, the box is collapsed and empty again, unless the person has
  // written on meanwhile. A refused press keeps it.
  const record = (decision: Decision, at: string) => {
    setPlain(null);
    setHeld(null);
    const note = noteOf(reason);
    decide.mutate({ decision, note }, {
      onSuccess: () => setReason((now) => (noteOf(now) === note ? NO_REASON : now)),
      onError: (err) =>
        setPlain(
          isRefusalError(err)
            ? { at: err.refusal.error.code === NOTE_TOO_LONG ? AT_REASON : at, refusal: err.refusal }
            : { at, failure: err instanceof Error ? err.message : String(err) },
        ),
    });
  };
  const hold = (at: string) => {
    setPlain(null);
    setHeld(at);
  };
  const answerAt: CardWiring["answerAt"] = plain
    ? {
        key: plain.at,
        node: plain.refusal ? (
          <RefusalNote
            refusal={plain.refusal}
            onDismiss={() => setPlain(null)}
            onExit={(exit) => (exit.decision ? record(exit.decision, plain.at) : setPlain(null))}
          />
        ) : (
          <FailureNote
            message={plain.failure ?? "no reason given"}
            onDismiss={() => setPlain(null)}
          />
        ),
      }
    : null;

  return {
    state: view?.state ?? null,
    columns: (summaries ?? []).map((s) => s.name),
    summaries: new Map((summaries ?? []).map((s) => [s.name, s])),
    task: view?.state.task ?? (target && target.column === view?.state.target ? target.detected_task : null),
    pending: decide.isPending,
    record,
    held,
    hold,
    answerAt,
    words: preview.answer ? wordsOf(preview.answer) : null,
    reason,
    setReason,
    reasonOnRecord: (marked) => noteOnRecord(view, log, kind, marked),
  };
}

const capital = (s: string) => `${s.charAt(0).toUpperCase()}${s.slice(1)}`;

/** The card: its label, the engine's name for the line and what the engine says of it. */
export function Frame({ line, children }: { line: QuestLine; children: ReactNode }) {
  const id = useId();
  return (
    <section
      className={c.card}
      aria-labelledby={id}
      data-testid={`declaration-${line.key}`}
      data-status={line.status}
    >
      <div className={c.kicker}>{line.label}</div>
      <h2 id={id} className={c.title}>
        {capital(line.name)}
      </h2>
      {line.reason ? (
        <p className={c.said} data-testid="line-reason">
          <Prose text={line.reason} />
        </p>
      ) : null}
      {line.would_change ? (
        <p className={c.said} data-testid="line-would-change">
          <Prose text={line.would_change} />
        </p>
      ) : null}
      {line.changes_nothing ? (
        <p className={c.said} data-testid="line-changes-nothing">
          <Prose text={line.changes_nothing} />
        </p>
      ) : null}
      {line.status === "waiting" ? (
        <p className={c.waiting} data-testid="waiting">
          Waiting for: {line.waiting_for.map((w) => w.name).join(", ")}
        </p>
      ) : (
        children
      )}
    </section>
  );
}

/** The engine's words under the option they answer, or nothing. */
function wordsNote(words: Words | null, decision: Decision | null): ReactNode {
  if (!words || !decision || words.key !== decisionKey(decision)) return undefined;
  return (
    <span className={c.words} data-testid="words" data-refused={words.refused || undefined}>
      <Prose text={words.text} />
    </span>
  );
}

/** The quiet label on the option a default line holds until it is confirmed. */
export const SET_FOR_YOU = "set for you";

/** What a card can send for an option: the decision, or what stands in the way (one sentence for
 *  each field, naming what it takes). */
export type Offer = { decision: Decision } | { problems: string[] };

/** The sentence a press the card will not send is answered with, under the option pressed. */
function BlockedNote({ problems }: { problems: string[] }) {
  return (
    <p className={c.blocked} role="alert" data-testid="blocked">
      <span className={c.blockedLabel}>Not recorded</span> {problems.join("; ")}
    </p>
  );
}

/** The quiet label on an option chosen and not yet recorded (FOUNDATION §3: a choice holds in the
 *  card until the action records it). */
export const CHOSEN = "chosen";

/** The card's record button (today's cards' RecordButton), what a press records named on it, the
 *  reason for the answer above it (ReasonBox: the one on record, and the box for the next, collapsed
 *  until asked for, which answers a note the server refuses), and the answer to its press under it
 *  where that press is not answered at an option. */
export function RecordChosen({
  label,
  card,
  onClick,
  answer = null,
  recorded,
}: {
  /** "Record: …", what a press sends; null while nothing is chosen. */
  label: string | null;
  card: CardWiring;
  onClick: () => void;
  answer?: ReactNode;
  /** The option the card marks as recorded (`Marked`), null where it marks none; left out by a card
   *  that draws this button over several choices (LeversCard), whose record in force is its one
   *  answer. */
  recorded?: Marked;
}) {
  return (
    <>
      <ReasonBox
        reason={card.reason}
        onChange={card.setReason}
        onRecord={card.reasonOnRecord(recorded)}
        answer={card.answerAt?.key === AT_REASON ? card.answerAt.node : null}
      />
      <Actions>
        <RecordButton
          disabled={label === null || card.pending}
          onClick={onClick}
          title="Records the choice above with the fields as they stand."
          testId="card-record"
        >
          {label ?? "Choose an option above"}
        </RecordButton>
      </Actions>
      {answer}
    </>
  );
}

/**
 * One choice of a card, as the option list the Record uses: pointing at an option previews what
 * it would record. A click chooses it (the stage holds its preview) and the card's button records
 * the whole decision; Enter on it records at once. `offerOf` gives, for a value, the decision, or
 * what stands in the way of sending it; a press that cannot be sent says so under the option,
 * until the fields are put right.
 *
 * The choice starts at the value on record, else the one set for the person, each only while the
 * card offers it (a use on record that no longer fits the outcome is not listed, so nothing is
 * chosen and the button asks for a choice). A card that records several choices with one button
 * (LeversCard) holds them itself (`chosen`, `onChoose`) and draws that button itself
 * (`button={false}`).
 */
export function Choice<V extends string>({
  card,
  field,
  legend,
  values,
  names,
  offerOf,
  recorded,
  setForYou = null,
  hint = true,
  testId,
  chosen: held,
  onChoose,
  button = true,
}: {
  card: CardWiring;
  field: string;
  legend: string;
  values: readonly V[];
  names: Record<V, string>;
  offerOf: (value: V) => Offer;
  /** The value on record, when the line is answered. */
  recorded: V | null;
  /** The value the engine set for the person, while the line is a default not yet confirmed: its
   *  option carries the quiet label "set for you" (FOUNDATION §3, the Confirm sweep). */
  setForYou?: V | null;
  hint?: boolean;
  testId: string;
  /** The value chosen, where the card holds it. */
  chosen?: V | null;
  onChoose?: (value: V) => void;
  /** Draw the card's record button under the options. */
  button?: boolean;
}) {
  const { setFocus } = useStageFocus();
  const offered = (v: V | null | undefined): v is V =>
    v !== null && v !== undefined && values.includes(v);
  const [own, setOwn] = useState<V | null>(
    offered(recorded) ? recorded : offered(setForYou) ? setForYou : null,
  );
  // A choice the list does not offer is no choice: the button never records an unlisted option.
  const holding = held !== undefined ? held : own;
  const chosen = offered(holding) ? holding : null;
  const decisions = new Map<string, Decision | null>();
  const problems = new Map<string, string[]>();
  for (const v of values) {
    const offer = offerOf(v);
    decisions.set(`${field}:${v}`, "decision" in offer ? offer.decision : null);
    if ("problems" in offer) problems.set(`${field}:${v}`, offer.problems);
  }
  // At most one quiet label per option (FOUNDATION §2): set for you, else chosen; the recorded one
  // carries the list's own.
  const tagOf = (v: V): OptionTag[] | undefined =>
    setForYou !== null && v === setForYou
      ? [{ text: SET_FOR_YOU, tone: "usual" }]
      : v === chosen && v !== recorded
        ? [{ text: CHOSEN, tone: "usual" }]
        : undefined;
  const items: OptionItem[] = values.map((v) => {
    const decision = decisions.get(`${field}:${v}`) ?? null;
    const tags = tagOf(v);
    return {
      key: `${field}:${v}`,
      label: names[v],
      line: "",
      decision,
      ...(tags ? { tags } : {}),
      // A refusal answering at the option says it already.
      note: card.answerAt?.key === `${field}:${v}` ? undefined : wordsNote(card.words, decision),
    };
  });
  const send = (key: string) => {
    const decision = decisions.get(key);
    if (decision) card.record(decision, key);
    else if (decision === null) card.hold(key);
  };
  const choose = (v: V) => {
    (onChoose ?? setOwn)(v);
    const decision = decisions.get(`${field}:${v}`);
    if (decision) setFocus({ kind: "option", decision, label: names[v] } satisfies StageFocus);
  };
  // A click on an option chooses it, ahead of the list's own click, which would record it; the
  // card's button records. Enter is the list's own, and records at once.
  const onClickCapture = (e: MouseEvent<HTMLDivElement>) => {
    const target = e.target as HTMLElement;
    if (target.closest("input, select, textarea, button, a")) return;
    const key = target.closest<HTMLElement>('[role="option"]')?.dataset.key;
    const v = values.find((x) => `${field}:${x}` === key);
    if (v === undefined) return;
    e.stopPropagation();
    choose(v);
  };
  const heldProblems = card.held === null ? undefined : problems.get(card.held);
  const answerAt =
    card.answerAt ??
    (card.held !== null && heldProblems
      ? { key: card.held, node: <BlockedNote problems={heldProblems} /> }
      : null);
  return (
    <fieldset className={c.group}>
      <legend className={c.legend}>{legend}</legend>
      <div onClickCapture={onClickCapture}>
        <Options
          items={items}
          mode="single"
          onRecord={(item) => send(item.key)}
          recordedKey={recorded === null ? null : `${field}:${recorded}`}
          pending={card.pending}
          answerAt={answerAt}
          label={`${legend} — preview with the arrow keys, Enter to record`}
          testId={testId}
          hint={hint}
        />
      </div>
      {button ? (
        <RecordChosen
          label={chosen === null ? null : `Record: ${names[chosen]}`}
          card={card}
          onClick={() => chosen !== null && send(`${field}:${chosen}`)}
          recorded={offered(recorded) ? { decision: decisions.get(`${field}:${recorded}`) ?? null } : null}
        />
      ) : null}
    </fieldset>
  );
}
