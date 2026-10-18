/**
 * What draws a stage's card column, and each quest line's card in it (pure; StageCard.tsx and
 * CardFor.tsx render them). Every line has one, never a blank (plans/W1_PART2_PLAN.md ruling 6), and
 * the renderer contract (CardFor.test.tsx) holds every (source, key) the engine emits to one here.
 *
 * The column (columnOf):
 *
 *   results      Results once reached (Fit pressed): W1-G's ResultsCards (src/quest/results), the
 *                card column's half of Predict Results, mounted once in place of the stage's lines;
 *                it draws them all (src/quest/contracts.ts)
 *   lines        every other stage, and Results before it is reached: the stage's lines, the open
 *                one through CardFor
 *
 * A line's card (rendererOf):
 *
 *   ask          a Router question: today's card through AskFor (record/AskFor.tsx)
 *   declaration  a declaration with its own card: W1-F's DECLARATION_CARDS (src/quest/cards/predict)
 *   finding      one noticing: the findings artifact filtered to it (FindingCard.tsx)
 *   reading      what the values settled, or the readings a refusal waits on: quiet rows (Readings.tsx)
 *   generic      anything else, a declaration without a card included: a card in the line's own
 *                words, the generic question's content (LineGeneric.tsx). It records nothing, so the
 *                card column lists such a line as "Not available yet" and never opens it (recordable)
 */
import { QUESTION_KEYS } from "../../api/m1-types";
import type { QuestLine, QuestStage } from "../../api/schema";
import type { DeclarationCards } from "../contracts";

export type Column = "results" | "lines";

export function columnOf(stage: Pick<QuestStage, "key" | "reached">): Column {
  return stage.key === "results" && stage.reached ? "results" : "lines";
}

export type Renderer = "ask" | "declaration" | "finding" | "reading" | "generic";

const ROUTER_KEYS: ReadonlySet<string> = new Set(QUESTION_KEYS);

/**
 * Whether the line's card can record an answer: every renderer's but the fallback's, which says in
 * the line's own words that this version has no card for it. A line it cannot answer is listed
 * quietly, "Not available yet", after those it can, and is never the card's open line (FOUNDATION
 * §3), so no "Answer" leads to a card that cannot answer.
 */
export function recordable(line: Pick<QuestLine, "source" | "key">, cards: DeclarationCards): boolean {
  return rendererOf(line, cards) !== "generic";
}

export function rendererOf(line: Pick<QuestLine, "source" | "key">, cards: DeclarationCards): Renderer {
  switch (line.source) {
    case "question":
      if (ROUTER_KEYS.has(line.key)) return "ask";
      // The ledger's ask card on a line of its own: the readings a refusal waits on, each with the
      // refusal's ways forward (core/quest.py, `ask:<question>`).
      if (line.key.startsWith("ask:")) return "reading";
      return "generic";
    case "declaration":
      return cards[line.key] ? "declaration" : "generic";
    case "finding":
      return "finding";
    case "reading":
      return "reading";
    default:
      return "generic";
  }
}
