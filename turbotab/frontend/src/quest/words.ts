import type { FamiliesLabel } from "../components/record/ask/ChoiceQuestions";
import { plain } from "../kit/parts";

/** The engine's running-text name ("the lens question") as a line's own heading, its data values
 *  in plain text (the calm budget boxes no column names: backticks dropped). */
export function said(name: string): string {
  const text = plain(name);
  return text ? text.charAt(0).toUpperCase() + text.slice(1) : text;
}

/** The families card's words in the shell: choosing the families records the choice, and Fit is
 *  pressed on the analysis flowchart, the stage's one Fit (FOUNDATION §7). */
export const chooseFamilies: FamiliesLabel = (one, n) => (one ? `Choose the ${one}` : `Choose these ${n} families`);
