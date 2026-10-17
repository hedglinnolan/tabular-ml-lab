/**
 * The interfaces W1 part 2's three frontend packages share (docs/turbotab-next/plans/
 * W1_PART2_PLAN.md §2). The shell (W1-E) renders each quest line's card through them; the Predict
 * declaration cards (W1-F) and Predict Results (W1-G) implement them. W1-E owns this file and may
 * add to it, never break it.
 */
import type { ReactElement } from "react";
import type { QuestLine } from "../api/schema";

/** One quest line's card, in the shell's card column. */
export interface LineCardProps {
  pid: string;
  line: QuestLine;
}
export type LineCard = (props: LineCardProps) => ReactElement | null;

/** The cards for declaration lines (source "declaration"), by the line's key: its decision kind. */
export type DeclarationCards = Partial<Record<string, LineCard>>;

/** Predict Results' two halves: the card column's and the tapestry's. */
export interface ResultsProps {
  pid: string;
}
export type ResultsPart = (props: ResultsProps) => ReactElement | null;

/**
 * The shell's card dispatcher (src/quest/cards/CardFor.tsx), added by W1-E. How the shell mounts
 * the parts above (src/quest/StageCard.tsx, src/quest/Tapestry.tsx; renderers.ts names each choice):
 *
 * - the Results stage, once reached (under Predict, once Fit is pressed), is two halves: its card column is
 *   `ResultsCards`, mounted once in place of the stage's lines and kept mounted after the opening,
 *   and its tapestry is `ResultsTapestry`, inside the kit's canvas (`data-kit`, so the calm tokens
 *   and the view kinds' canvas tokens apply). `ResultsCards` draws every Results line: the
 *   declarations there (`set_updating`, through DECLARATION_CARDS), the opening (`open_seal`) and
 *   the explain offer (`set_explain`); the shell draws none of them beside it. The stage's head (its
 *   name, why it dropped back), For the record, the manuscript's arrival line and the one primary
 *   action at the foot stay the shell's. Before Results is reached its lines wait, as any stage's do;
 * - in every other stage, each line the card column expands goes through CardFor, which picks its
 *   renderer by the line's source;
 * - a declaration line's card there is `DECLARATION_CARDS[line.key]` when W1-F defines one, else the
 *   dispatcher's fallback card in the line's own words (src/quest/cards/LineGeneric.tsx).
 */
export interface CardForProps extends LineCardProps {
  /** Open another stage on the card (a finding's lever, a "Waiting for" link). */
  onOpenStage?: (stage: string) => void;
}
