/**
 * `set_updating`: whether the regression's coefficients are shrunk by the calibration slope, as
 * model updating, or left as fitted. The two values the model knows. What shrinking does to this
 * model is the engine's to say: the evaluation stage serves it once the fit is scored, and the
 * preview of the option says what it can before then.
 */
import { Pending } from "../../../components/record/blocks";
import type { LineCard } from "../../contracts";
import { Choice, Frame, useCard } from "./card";
import { UPDATING_METHODS, updating, type Updating } from "./payloads";

const NAMES: Record<Updating["method"], string> = {
  none: "No updating",
  shrinkage: "Shrinkage",
};

export const UpdatingCard: LineCard = ({ pid, line }) => {
  const card = useCard(pid, "set_updating");
  const on = line.status === "answered" ? (card.state?.updating ?? null) : null;
  return (
    <Frame line={line}>
      {card.state ? (
        <Choice
          key={line.decision_id ?? "open"}
          card={card}
          field="method"
          legend="Method"
          values={UPDATING_METHODS}
          names={NAMES}
          offerOf={(method) => ({ decision: updating(method) })}
          recorded={on?.method ?? null}
          testId="options-updating"
        />
      ) : (
        <Pending>Loading the project…</Pending>
      )}
    </Frame>
  );
};
