/**
 * Which renderer draws a quest line's card (pure; CardFor.tsx renders it). Every line has one, never
 * a blank (plans/W1_PART2_PLAN.md ruling 6), and the renderer contract (CardFor.test.tsx) holds
 * every (source, key) the engine emits to a renderer here:
 *
 *   ask          a Router question: today's card through AskFor (record/AskFor.tsx)
 *   results      a question in Results (the opening): W1-G's ResultsCards (src/quest/results)
 *   declaration  a declaration with its own card: W1-F's DECLARATION_CARDS (src/quest/cards/predict)
 *   finding      one noticing: the findings artifact filtered to it (FindingCard.tsx)
 *   reading      what the values settled, or the readings a refusal waits on: quiet rows (Readings.tsx)
 *   generic      anything else, a declaration without a card included: a card in the line's own
 *                words, the generic question's content (LineGeneric.tsx)
 */
import { QUESTION_KEYS } from "../../api/m1-types";
import type { QuestLine } from "../../api/schema";
import type { DeclarationCards } from "../contracts";

export type Renderer = "ask" | "results" | "declaration" | "finding" | "reading" | "generic";

const ROUTER_KEYS: ReadonlySet<string> = new Set(QUESTION_KEYS);

export function rendererOf(
  line: Pick<QuestLine, "source" | "key">,
  stage: string,
  cards: DeclarationCards,
): Renderer {
  switch (line.source) {
    case "question":
      if (stage === "results" && ROUTER_KEYS.has(line.key)) return "results";
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
