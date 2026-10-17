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
 * the parts above:
 *
 * - each line the card column expands goes through CardFor, which picks its renderer by the line's
 *   source and stage (src/quest/cards/renderers.ts);
 * - a declaration line's card is `DECLARATION_CARDS[line.key]` when W1-F defines one, else
 *   GenericAsk in the line's own words;
 * - in Results, the question line (the opening, `open_seal`) is `ResultsCards`, mounted once in the
 *   card column where that line is expanded (the open line, or a line clicked open), and the stage's
 *   declarations there (`set_updating`, `set_explain`) are W1-F's cards like any other;
 * - the tapestry of the Results stage is `ResultsTapestry`, inside the kit's canvas (`data-kit`, so
 *   the calm tokens and the view kinds' canvas tokens apply).
 */
export interface CardForProps extends LineCardProps {
  /** The stage the line sits in (a QuestStage key). */
  stage: string;
  /** Open another stage on the card (a finding's lever, a "Waiting for" link). */
  onOpenStage?: (stage: string) => void;
}
