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
import { Choice, Frame, useCard, type CardWiring, type Offer } from "./card";
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
  trouble,
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
const KEEP = "Predictors to keep (most variable)";
const IMBALANCE_NAMES: Record<Levers["imbalance"], string> = {
  none: "None",
  weights: "Class weights",
  undersample: "Undersampling",
  oversample: "Oversampling",
};

/**
 * The rules in force, in the card's words, from the engine's own value: the levers recorded, else
 * the model's defaults (no lever recorded is none of them). The Confirm line states them, since the
 * engine serves no reason for this default (FOUNDATION §3, the Confirm sweep).
 */
export function leversDefault(state: ProjectState): string {
  const r: LeverRules = state.levers ?? DEFAULTS.set_levers;
  const lower = (s: string) => s.charAt(0).toLowerCase() + s.slice(1);
  const filter = `${lower(FILTER_NAMES[r.variance_filter])}${r.variance_filter === "top" && r.keep !== null ? ` (${r.keep} kept)` : ""}`;
  return `Predictor forms: ${lower(FORM_NAMES[r.forms])}; variance filter: ${filter}; imbalance correction: ${lower(IMBALANCE_NAMES[r.imbalance])}.`;
}

function Body({
  card,
  state,
  recorded,
  setForYou,
}: {
  card: CardWiring;
  state: ProjectState;
  recorded: boolean;
  setForYou: boolean;
}) {
  const on = recorded ? state.levers : null;
  const rules: LeverRules = on ?? DEFAULTS.set_levers;
  // A default not yet confirmed: the rules in force, each marked on its own list.
  const set = setForYou ? (state.levers ?? DEFAULTS.set_levers) : null;
  const [keep, setKeep] = useState(rules.keep === null ? "" : String(rules.keep));
  const typed = parseTyped(keep, BOUNDS.set_levers.keep, true);

  /** The levers with one of the three changed, the other two as they stand. */
  const change = (patch: Partial<LeverRules>): Offer => {
    const next = { ...rules, ...patch };
    // The count is read by the "most variable" filter alone.
    if (next.variance_filter === "top" && typed === "invalid") {
      return { problems: trouble(KEEP, typed, BOUNDS.set_levers.keep, true) };
    }
    return { decision: levers({ ...next, keep: typed === "invalid" ? null : typed }) };
  };

  return (
    <>
      <div className={c.fields}>
        <NumberField
          name="keep"
          label={KEEP}
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
        offerOf={(v) => change({ forms: v })}
        recorded={on?.forms ?? null}
        setForYou={set?.forms ?? null}
        testId="options-forms"
        hint={false}
      />
      <Choice
        card={card}
        field="variance_filter"
        legend="Variance filter"
        values={VARIANCE_FILTERS}
        names={FILTER_NAMES}
        offerOf={(v) => change({ variance_filter: v })}
        recorded={on?.variance_filter ?? null}
        setForYou={set?.variance_filter ?? null}
        testId="options-variance_filter"
        hint={false}
      />
      <Choice
        card={card}
        field="imbalance"
        legend="Imbalance correction"
        values={IMBALANCES}
        names={IMBALANCE_NAMES}
        offerOf={(v) => change({ imbalance: v })}
        recorded={on?.imbalance ?? null}
        setForYou={set?.imbalance ?? null}
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
          setForYou={line.status === "set_for_you"}
        />
      ) : (
        <Pending>Loading the project…</Pending>
      )}
    </Frame>
  );
};
