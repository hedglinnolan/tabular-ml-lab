/**
 * `set_sensitivity`: the analyses beside the primary, each the same model on the rows its own rules
 * keep (core/decisions.py `SetSensitivity`). The engine asks it as a counted Decide once an exclusion
 * rule is recorded (core/quest.py DECLARATIONS), so without a card a Predict project with any
 * exclusion could not reach Fit. Two answers: the every-row analysis (the same model on every row,
 * as Banna et al. 2017 ask), or none. What each does to this table is the engine's preview of it
 * (the rows the analysis keeps, against the primary's), said under the option pointed at.
 */
import { Pending } from "../../../components/record/blocks";
import type { LineCard } from "../../contracts";
import { Choice, Frame, useCard } from "./card";
import { SENSITIVITIES, sensitivity, sensitivityOf, type SensitivityChoice } from "./payloads";

const NAMES: Record<SensitivityChoice, string> = {
  every_row: "Also on every row, with no exclusion",
  none: "No sensitivity analysis",
};

export const SensitivityCard: LineCard = ({ pid, line }) => {
  const card = useCard(pid, "set_sensitivity");
  const recorded = line.status === "answered" ? sensitivityOf(card.state?.sensitivity) : null;
  return (
    <Frame line={line}>
      {card.state ? (
        <Choice
          key={line.decision_id ?? "open"}
          card={card}
          field="analyses"
          legend="Beside the primary analysis"
          values={SENSITIVITIES}
          names={NAMES}
          offerOf={(choice) => ({ decision: sensitivity(choice) })}
          recorded={recorded}
          testId="options-sensitivity"
        />
      ) : (
        <Pending>Loading the project…</Pending>
      )}
    </Frame>
  );
};
