/**
 * The dispatcher's fallback (ruling 6): a line no other renderer draws, a declaration without its
 * own card included, is the generic question (record/generic/GenericAsk.tsx) in the line's own
 * words: its name, why it was set and what another answer would change, where the quest log
 * serves them. It offers no option it cannot record: the engine defines each declaration's payload
 * (core/decisions.py), and a card that posts one is W1-F's (src/quest/cards/predict) or later.
 */
import type { QuestionKey } from "../../api/m1-types";
import type { QuestLine } from "../../api/schema";
import { GenericAsk } from "../../components/record/generic/GenericAsk";
import type { GenericQuestion } from "../../components/record/generic/compose";
import { said } from "../words";

const NOOP = () => {};

/** The generic question's words for a line: its reasons as data, and what this version lacks. */
export function lineQuestion(line: QuestLine): GenericQuestion {
  const data = [line.reason, line.would_change, line.changes_nothing].filter(
    (x): x is string => !!x,
  );
  return {
    grammar: "choice",
    data,
    options: [],
    fields: [],
    note:
      line.status === "answered" || line.status === "set_for_you"
        ? `This version has no card to change ${line.name} here yet.`
        : `This version has no card to answer ${line.name} here yet.`,
  };
}

export function LineGeneric({ line }: { line: QuestLine }) {
  return (
    <GenericAsk
      // The Question shell keys its test ids and grammar by a Router key; a line that is not one
      // reads as a choice, named by its own name (fallbackTitle).
      q={lineQuestion(line)}
      fallbackTitle={said(line.name)}
      entry={undefined}
      pending={false}
      record={NOOP}
      answerAt={null}
      shell={{
        qkey: line.key as QuestionKey,
        now: line.status === "open",
        arriving: false,
        onArrived: NOOP,
        onOpenDrawer: NOOP,
        reopened: false,
      }}
    />
  );
}
