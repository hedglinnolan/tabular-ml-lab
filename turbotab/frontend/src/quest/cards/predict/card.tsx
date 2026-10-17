/**
 * What the six Predict declaration cards share: the frame, the recording, and the engine's words.
 *
 * A card records through the existing decision hook (`useDecide`) and answers a press the server
 * does not record the way the Record's plain wiring does: the refusal and its exits, or the
 * failure, at the option pressed. It states only what the engine serves: a line of the quest log
 * (`reason`, `would_change`, `changes_nothing`), the preview the engine serves for the option
 * pointed at (the caption of its first view, else its note, else its caution), or the refusal the
 * preview meets. A card with nothing served to say about an option names the option and stops.
 */
import { useId, useState, type ReactNode } from "react";
import { isRefusalError } from "../../../api/client";
import { useColumnSummaries, useDecide, useProjectView } from "../../../api/queries";
import type { Decision, ProjectState, QuestLine, Refusal } from "../../../api/schema";
import { Options, type OptionItem } from "../../../components/record/Options";
import { FailureNote, RefusalNote } from "../../../components/record/Refusal";
import { Prose } from "../../../components/Prose";
import { decisionKey, stateSeq, usePreview, type Answer } from "../../../components/stage/data";
import { useStageFocus } from "../../../state/focus";
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
  pending: boolean;
  /** Record a decision; `at` names the option a refusal answers under. */
  record: (decision: Decision, at: string) => void;
  answerAt: { key: string; node: ReactNode } | null;
  words: Words | null;
}

/** The wiring of one card: its project's state, its recording, and the preview of the option of
 *  its own kind that is pointed at (the Stage previews the same option from the same cache). */
export function useCard(pid: string, kind: Decision["kind"]): CardWiring {
  const view = useProjectView(pid).data;
  const summaries = useColumnSummaries(pid, view?.stages.ingest?.status === "fresh").data;
  const decide = useDecide(pid);
  const [plain, setPlain] = useState<PlainAnswer | null>(null);
  const { focus } = useStageFocus();
  const pointed =
    focus.kind === "option" && focus.decision.kind === kind
      ? { decision: focus.decision, label: focus.label }
      : null;
  const preview = usePreview(pid, pointed, view ? stateSeq(view) : 0);

  const record = (decision: Decision, at: string) => {
    setPlain(null);
    decide.mutate(decision, {
      onError: (err) =>
        setPlain(
          isRefusalError(err)
            ? { at, refusal: err.refusal }
            : { at, failure: err instanceof Error ? err.message : String(err) },
        ),
    });
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
    pending: decide.isPending,
    record,
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

/**
 * One choice of a card, as the option list the Record uses: pointing at an option previews what
 * it would record, and pressing it records the whole decision. `decisionOf` gives the decision
 * for a value, or null where the card cannot send it yet (the press then does nothing).
 */
export function Choice<V extends string>({
  card,
  field,
  legend,
  values,
  names,
  decisionOf,
  recorded,
  hint = true,
  testId,
  blocked,
}: {
  card: CardWiring;
  field: string;
  legend: string;
  values: readonly V[];
  names: Record<V, string>;
  decisionOf: (value: V) => Decision | null;
  /** The value on record, when the line is answered. */
  recorded: V | null;
  hint?: boolean;
  testId: string;
  /** A press the card could not send: what it still needs. */
  blocked?: (value: V) => void;
}) {
  const items: OptionItem[] = values.map((v) => {
    const decision = decisionOf(v);
    return {
      key: `${field}:${v}`,
      label: names[v],
      line: "",
      decision,
      // A refusal answering at the option says it already.
      note: card.answerAt?.key === `${field}:${v}` ? undefined : wordsNote(card.words, decision),
    };
  });
  const press = (item: OptionItem) => {
    const value = values.find((v) => `${field}:${v}` === item.key);
    if (value === undefined) return;
    const decision = decisionOf(value);
    if (decision) card.record(decision, item.key);
    else blocked?.(value);
  };
  return (
    <fieldset className={c.group}>
      <legend className={c.legend}>{legend}</legend>
      <Options
        items={items}
        mode="single"
        onRecord={press}
        recordedKey={recorded === null ? null : `${field}:${recorded}`}
        pending={card.pending}
        answerAt={card.answerAt}
        label={`${legend} — preview with the arrow keys, Enter to record`}
        testId={testId}
        hint={hint}
      />
    </fieldset>
  );
}
