/**
 * `set_levers`: the rules repeated inside each training fold. Three choices (the predictors' form,
 * the variance filter, the imbalance correction), each option changing one and sending the other
 * two back as they stand, as the engine's own offers do (stages/explore.py). How many predictors
 * the "most variable" filter keeps is sent with that filter alone. A correction the outcome cannot
 * use, or a filter with no count, the engine refuses in its own words, with its exits.
 */
import { useState } from "react";
import type { ProjectState } from "../../../api/schema";
import { Pending } from "../../../components/record/blocks";
import type { LineCard } from "../../contracts";
import { Choice, Frame, useCard, type CardWiring } from "./card";
import c from "./declaration.module.css";
import { NumberField } from "./fields";
import {
  BOUNDS,
  DEFAULTS,
  FORMS,
  IMBALANCES,
  VARIANCE_FILTERS,
  levers,
  parseTyped,
  type LeverRules,
  type Levers,
} from "./payloads";

const FORM_NAMES: Record<Levers["forms"], string> = {
  none: "None",
  rule: "Spline by rule",
  inner_cv: "Linear or spline, by inner cross-validation",
};
const FILTER_NAMES: Record<Levers["variance_filter"], string> = {
  none: "None",
  near_zero: "Near-zero variance",
  top: "Most variable",
};
const IMBALANCE_NAMES: Record<Levers["imbalance"], string> = {
  none: "None",
  weights: "Class weights",
  undersample: "Undersampling",
  oversample: "Oversampling",
};

function Body({
  card,
  state,
  recorded,
}: {
  card: CardWiring;
  state: ProjectState;
  recorded: boolean;
}) {
  const on = recorded ? state.levers : null;
  const rules: LeverRules = on ?? DEFAULTS.set_levers;
  const [keep, setKeep] = useState(rules.keep === null ? "" : String(rules.keep));
  const typed = parseTyped(keep, BOUNDS.set_levers.keep, true);

  /** The levers with one of the three changed, the other two as they stand. */
  const change = (patch: Partial<LeverRules>): Levers | null => {
    const next = { ...rules, ...patch };
    if (next.variance_filter === "top" && typed === "invalid") return null;
    return levers({ ...next, keep: typed === "invalid" ? null : typed });
  };

  return (
    <>
      <div className={c.fields}>
        <NumberField
          name="keep"
          label="Predictors to keep (most variable)"
          text={keep}
          onChange={setKeep}
          bounds={BOUNDS.set_levers.keep}
          integer
        />
      </div>
      <Choice
        card={card}
        field="forms"
        legend="Predictor forms"
        values={FORMS}
        names={FORM_NAMES}
        decisionOf={(v) => change({ forms: v })}
        recorded={on?.forms ?? null}
        testId="options-forms"
        hint={false}
      />
      <Choice
        card={card}
        field="variance_filter"
        legend="Variance filter"
        values={VARIANCE_FILTERS}
        names={FILTER_NAMES}
        decisionOf={(v) => change({ variance_filter: v })}
        recorded={on?.variance_filter ?? null}
        testId="options-variance_filter"
        hint={false}
      />
      <Choice
        card={card}
        field="imbalance"
        legend="Imbalance correction"
        values={IMBALANCES}
        names={IMBALANCE_NAMES}
        decisionOf={(v) => change({ imbalance: v })}
        recorded={on?.imbalance ?? null}
        testId="options-imbalance"
      />
    </>
  );
}

export const LeversCard: LineCard = ({ pid, line }) => {
  const card = useCard(pid, "set_levers");
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
