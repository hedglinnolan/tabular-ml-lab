/**
 * What the six Predict declaration cards share: the frame, the recording, and the engine's words.
 *
 * A card records through the existing decision hook (`useDecide`) and answers a press the server
 * does not record the way the Record's plain wiring does: the refusal and its exits, or the
 * failure, at the option pressed. It states only what the engine serves: a line of the quest log
 * (`reason`, `would_change`, `changes_nothing`), the preview the engine serves for the option
 * pointed at (the caption of its first view, else its note, else its caution), or the refusal the
 * preview meets. A card with nothing served to say about an option names the option and stops.
 *
 * A press the card will not send (a number outside what the model takes, a column still to choose)
 * is answered at the option too, in one sentence for each field in the way, so no press is silent.
 *
 * Each card records from its own visible button, as today's cards do (finding 8 of the 2026-10-17
 * walkthrough): a click on an option chooses it, marked "chosen" until it is recorded, and the
 * button under the options records the choice; Enter on an option records it at once, as before.
 */
import { useId, useState, type MouseEvent, type ReactNode } from "react";
import { isRefusalError } from "../../../api/client";
import { useColumnSummaries, useDecide, useProjectView, useStageResult } from "../../../api/queries";
import type { ColumnSummary, Decision, ProjectState, QuestLine, Refusal } from "../../../api/schema";
import { Actions, RecordButton } from "../../../components/record/ask/common";
import { Options, type OptionItem, type OptionTag } from "../../../components/record/Options";
import { FailureNote, RefusalNote } from "../../../components/record/Refusal";
import { Prose } from "../../../components/Prose";
import { decisionKey, stateSeq, usePreview, type Answer } from "../../../components/stage/data";
import { useStageFocus, type StageFocus } from "../../../state/focus";
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
}

/** The wiring of one card: its project's state, its recording, and the preview of the option of
 *  its own kind that is pointed at (the Stage previews the same option from the same cache). */
export function useCard(pid: string, kind: Decision["kind"]): CardWiring {
  const view = useProjectView(pid).data;
  const summaries = useColumnSummaries(pid, view?.stages.ingest?.status === "fresh").data;
  const targetStatus = view?.stages.target_info?.status;
  const target = useStageResult(pid, "target_info", targetStatus !== undefined && targetStatus !== "idle" && targetStatus !== "blocked").data?.artifact;
  const decide = useDecide(pid);
  const [plain, setPlain] = useState<PlainAnswer | null>(null);
  const [held, setHeld] = useState<string | null>(null);
  const { focus } = useStageFocus();
  const pointed =
    focus.kind === "option" && focus.decision.kind === kind
      ? { decision: focus.decision, label: focus.label }
      : null;
  const preview = usePreview(pid, pointed, view ? stateSeq(view) : 0);

  const record = (decision: Decision, at: string) => {
    setPlain(null);
    setHeld(null);
    decide.mutate(decision, {
      onError: (err) =>
        setPlain(
          isRefusalError(err)
            ? { at, refusal: err.refusal }
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

/** The card's record button (today's cards' RecordButton), what a press records named on it, and
 *  the answer to its press under it where that press is not answered at an option. */
export function RecordChosen({
  label,
  card,
  onClick,
  answer = null,
}: {
  /** "Record: …", what a press sends; null while nothing is chosen. */
  label: string | null;
  card: CardWiring;
  onClick: () => void;
  answer?: ReactNode;
}) {
  return (
    <>
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
        />
      ) : null}
    </fieldset>
  );
}
