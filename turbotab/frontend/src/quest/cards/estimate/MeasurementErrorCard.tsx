/**
 * `set_measurement_error`: whether the intakes the recalls measure are corrected for day-to-day
 * error by regression calibration, declared as a secondary analysis beside the uncorrected estimate
 * (core/decisions.py `SetMeasurementError`; core/stages/calibration.py). A counted Decide in Models
 * under Estimate where each person's row is the mean of their recalls and the linear family is
 * chosen (core/quest.py). The two answers the model knows, and the bootstrap's refits for the
 * calibrated coefficient's interval, which regression calibration alone reads.
 *
 * The card does not ask which intakes are reported (none named reports every exposure the recalls
 * measure), the interval (the engine refuses all but the bootstrap over the whole chain) or the
 * adjustment set (the server records the one in force). A press the engine refuses (regression
 * calibration under prediction) is answered in its own words, with its exits; what the calibration
 * stage then cannot run on this table (rows not a mean of recalls, a design the bootstrap cannot
 * draw) is said by that stage's result, which this card does not fetch.
 */
import { useState } from "react";
import type { ProjectState } from "../../../api/schema";
import { Pending } from "../../../components/record/blocks";
import type { LineCard } from "../../contracts";
import { Choice, Frame, useCard, type CardWiring, type Offer } from "../predict/card";
import c from "../predict/declaration.module.css";
import { NumberField } from "../predict/fields";
import { parseTyped, trouble } from "../predict/payloads";
import {
  BOUNDS,
  CALIBRATION_METHODS,
  DEFAULTS,
  measurementError,
  type MeasurementError,
} from "./payloads";

const NAMES: Record<MeasurementError["method"], string> = {
  none: "No calibration",
  regression_calibration: "Regression calibration",
};
const REFITS = "Bootstrap refits for the interval";

function Body({
  card,
  state,
  recorded,
}: {
  card: CardWiring;
  state: ProjectState;
  recorded: boolean;
}) {
  const on = recorded ? state.measurement_error : null;
  const bounds = BOUNDS.set_measurement_error.n_boot;
  const [refits, setRefits] = useState(String(on?.n_boot ?? DEFAULTS.set_measurement_error.n_boot));
  const typed = parseTyped(refits, bounds, true, true);
  const offerOf = (method: MeasurementError["method"]): Offer =>
    method === "regression_calibration" && typed === "invalid"
      ? { problems: trouble(REFITS, typed, bounds, true) }
      : { decision: measurementError(method, typed === "invalid" ? DEFAULTS.set_measurement_error.n_boot : typed) };
  return (
    <>
      <div className={c.fields}>
        <NumberField
          name="n_boot"
          label={REFITS}
          text={refits}
          onChange={setRefits}
          bounds={bounds}
          integer
          required
        />
      </div>
      <Choice
        card={card}
        field="method"
        legend="Correction"
        values={CALIBRATION_METHODS}
        names={NAMES}
        offerOf={offerOf}
        recorded={on?.method ?? null}
        testId="options-calibration"
      />
    </>
  );
}

export const MeasurementErrorCard: LineCard = ({ pid, line }) => {
  const card = useCard(pid, "set_measurement_error");
  return (
    <Frame line={line}>
      {card.state ? (
        <Body
          key={line.decision_id ?? "open"}
          card={card}
          state={card.state}
          recorded={line.status === "answered"}
        />
      ) : (
        <Pending>Loading the project…</Pending>
      )}
    </Frame>
  );
};
