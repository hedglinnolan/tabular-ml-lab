/**
 * The interfaces W1 part 2's three frontend packages share (docs/turbotab-next/plans/
 * W1_PART2_PLAN.md §2). The shell (W1-E) renders each quest line's card through them; the Predict
 * declaration cards (W1-F) and Predict Results (W1-G) implement them. W1-E owns this file and may
 * add to it, never break it.
 */
import { createContext, type ReactElement } from "react";
import type { ProjectState, QuestLine } from "../api/schema";

/** One quest line's card, in the shell's card column. */
export interface LineCardProps {
  pid: string;
  line: QuestLine;
}
export type LineCard = (props: LineCardProps) => ReactElement | null;

/** The cards for declaration lines (source "declaration"), by the line's key: its decision kind. */
export type DeclarationCards = Partial<Record<string, LineCard>>;

/**
 * A Confirm line's default in its card's words, from the engine's own value in the project's state,
 * for a line the engine serves no `reason` for (FOUNDATION §3: each choice in the sweep sits with
 * what it is). Null where the state holds nothing to say. By the line's key, as DeclarationCards.
 */
export type DeclarationDefaults = Partial<Record<string, (state: ProjectState) => string | null>>;

/**
 * Where the card's one primary action is drawn (FOUNDATION §1): the foot of the card column, below
 * For the record. The shell provides it for a stage whose primary action a part draws itself (the
 * opening, in Results: the part holds its chain of refusals); a part mounted outside the shell gets
 * null and draws the action in place.
 */
export const FootSlot = createContext<HTMLElement | null>(null);

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
 *   declarations there (`set_updating` and `set_explain`, through the dispatcher, so W1-F's cards are
 *   their only renderers) and the opening (`open_seal`); the shell draws none of them beside it. The
 *   stage's head (its name, why it dropped back), For the record and the manuscript's arrival line
 *   stay the shell's, and so does the foot, except while the opening is the stage's next act: then
 *   `ResultsCards` draws "Open the held-out rows" into the shell's `FootSlot`. Before Results is
 *   reached its lines wait, as any stage's do;
 * - in every other stage, each line the card column expands goes through CardFor, which picks its
 *   renderer by the line's source;
 * - a declaration line's card there is `DECLARATION_CARDS[line.key]` when W1-F defines one, else the
 *   dispatcher's fallback card in the line's own words (src/quest/cards/LineGeneric.tsx).
 */
export interface CardForProps extends LineCardProps {
  /** Open a stage on the card (a finding's lever, a "Waiting for" link), with the line the lever or
   *  link names expanded and brought into view; without one, the stage's open line. */
  onOpenStage?: (stage: string, line?: string) => void;
}
