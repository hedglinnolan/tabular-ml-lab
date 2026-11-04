/**
 * Predict Results v0 (plans/W1_PART2_PLAN.md §2 W1-G), the shell's Results stage in two halves
 * (`ResultsPart`, src/quest/contracts.ts):
 *
 *   ResultsCards     the card column: the Results declarations (the shrinkage question and the
 *                    explanations, through the dispatcher: W1-F's cards), the triage of the open
 *                    noticings, and the opening of the held-out rows with its chain of refusals
 *                    (naming the final model through their exits), whose press it draws at the
 *                    card's foot (the shell's FootSlot) (cards.tsx)
 *   ResultsTapestry  the exhibits: the cross-validated comparison with BBC-CV, the held-out scores
 *                    beside the baseline and its verdict once opened, calibration, SHAP importance,
 *                    the ALE curves and the interpretable model's cost (tapestry.tsx); under any
 *                    goal but prediction, Estimate Results v0 in their place: the plan-lock line,
 *                    Table 2 with its forest, and the primary model's diagnostics (estimate.tsx)
 *
 * Both read the project view the shell already holds (TanStack Query) and render nothing until it
 * is there.
 */
import { useProjectView } from "../../api/queries";
import type { ResultsPart } from "../contracts";
import { Cards } from "./cards";
import { EstimateTapestry } from "./estimate";
import { Tapestry } from "./tapestry";

export const ResultsCards: ResultsPart = ({ pid }) => {
  const view = useProjectView(pid).data;
  return view ? <Cards pid={pid} view={view} /> : null;
};

export const ResultsTapestry: ResultsPart = ({ pid }) => {
  const view = useProjectView(pid).data;
  if (!view) return null;
  return view.state.purpose === "prediction" ? <Tapestry pid={pid} view={view} /> : <EstimateTapestry pid={pid} view={view} />;
};
