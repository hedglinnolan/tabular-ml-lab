/**
 * The dispatcher's fallback (ruling 6): a line no other renderer draws, a declaration without its
 * own card included, is a card in the line's own words: its name, why it was set and what another
 * answer would change, where the quest log serves them. It offers no option it cannot record: the
 * engine defines each declaration's payload (core/decisions.py), and a card that posts one is
 * W1-F's (src/quest/cards/predict) or later.
 *
 * It is the generic question's content (record/generic/GenericAsk.tsx, a choice with no options)
 * in the dispatcher's own card, because the Question shell GenericAsk wears heads every card with
 * its teaching entry's title or else its Router key, and a line has no teaching entry: the card
 * would be headed by the engine's decision kind ("set sensitivity"), which never reaches the
 * screen (FOUNDATION §2).
 */
import type { QuestLine } from "../../api/schema";
import { Plain } from "../../kit";
import { said } from "../words";
import c from "./cards.module.css";

/** The line's own words: its reasons, and what this version lacks. */
export function lineWords(line: QuestLine): { data: string[]; note: string } {
  return {
    data: [line.reason, line.would_change, line.changes_nothing].filter((x): x is string => !!x),
    note:
      line.status === "answered" || line.status === "set_for_you"
        ? `This version has no card to change ${line.name} here yet.`
        : `This version has no card to answer ${line.name} here yet.`,
  };
}

export function LineGeneric({ line }: { line: QuestLine }) {
  const { data, note } = lineWords(line);
  return (
    <section className={c.lineCard} data-testid={`generic-${line.key}`} aria-label={said(line.name)}>
      <p className={c.kicker}>{said(line.name)}</p>
      {data.length ? (
        <ul className={c.rows} data-testid={`generic-data-${line.key}`}>
          {data.map((text) => (
            <li key={text} className={c.row}>
              <Plain text={text} />
            </li>
          ))}
        </ul>
      ) : null}
      <p className={c.quiet}>
        <Plain text={note} />
      </p>
    </section>
  );
}
