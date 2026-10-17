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
