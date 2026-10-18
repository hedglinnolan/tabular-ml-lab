/**
 * `set_selection`: how predictors are selected. Each method runs inside the resampling (the card
 * sends `where: "in_fold"`; the engine refuses selection outside it as a false performance
 * number, so the other value is not offered). A method's own settings (how many screening keeps;
 * the threshold of stability selection and of the univariable screen; stability selection's q)
 * are sent with that method alone, and the answer to "were the predictors chosen beforehand" with
 * any. What the engine refuses (a method for an outcome it cannot read, stepwise with too few
 * rows) it refuses in its own words, at the option.
 */
import { useState } from "react";
import type { ProjectState } from "../../../api/schema";
import { Pending } from "../../../components/record/blocks";
import type { LineCard } from "../../contracts";
import { Choice, Frame, useCard, type CardWiring, type Offer } from "./card";
import c from "./declaration.module.css";
import { NumberField, SelectField } from "./fields";
import {
  BOUNDS,
  METHODS,
  PRE_SELECTED,
  SETTINGS_OF,
  parseTyped,
  selection,
  trouble,
  type Selection,
  type Setting,
} from "./payloads";

const NAMES: Record<Selection["method"], string> = {
  none: "No selection",
  elastic_net: "Elastic net",
  stability: "Stability selection",
  screening: "Screening",
  stepwise: "Stepwise",
  univariable: "Univariable screen",
  vip: "VIP from PLS",
};
const PRE_NAMES: Record<NonNullable<Selection["pre_selected"]>, string> = {
  no: "No",
  yes: "Yes",
  other_data: "Yes, from other data",
  unknown: "Unknown",
};
const SETTING_LABELS: Record<Setting, string> = {
  keep: "Predictors to keep (screening)",
  threshold: "Threshold (stability, univariable)",
  q: "Predictors per resample, q (stability)",
};
const text = (n: number | null | undefined) => (n === null || n === undefined ? "" : String(n));

function Body({
  card,
  state,
  recorded,
}: {
  card: CardWiring;
  state: ProjectState;
  recorded: boolean;
}) {
  const on = recorded ? state.selection : null;
  const [keep, setKeep] = useState(text(on?.keep));
  const [threshold, setThreshold] = useState(text(on?.threshold));
  const [q, setQ] = useState(text(on?.q));
  const [pre, setPre] = useState<string>(on?.pre_selected ?? "");
  const bounds = BOUNDS.set_selection;
  const typed = {
    keep: parseTyped(keep, bounds.keep, true),
    threshold: parseTyped(threshold, bounds.threshold, false),
    q: parseTyped(q, bounds.q, true),
  };

  const offerOf = (method: Selection["method"]): Offer => {
    // Only what the method reads has to be a number the model accepts.
    const problems = SETTINGS_OF[method].flatMap((k) =>
      trouble(SETTING_LABELS[k], typed[k], bounds[k], k !== "threshold"),
    );
    if (problems.length > 0) return { problems };
    const number = (t: (typeof typed)[keyof typeof typed]) => (t === "invalid" ? null : t);
    return {
      decision: selection(method, {
        keep: number(typed.keep),
        threshold: number(typed.threshold),
        q: number(typed.q),
        pre_selected: pre === "" ? null : (pre as NonNullable<Selection["pre_selected"]>),
      }),
    };
  };

  return (
    <>
      <div className={c.fields}>
        <NumberField
          name="keep"
          label={SETTING_LABELS.keep}
          text={keep}
          onChange={setKeep}
          bounds={bounds.keep}
          integer
        />
        <NumberField
          name="threshold"
          label={SETTING_LABELS.threshold}
          text={threshold}
          onChange={setThreshold}
          bounds={bounds.threshold}
          integer={false}
        />
        <NumberField
          name="q"
          label={SETTING_LABELS.q}
          text={q}
          onChange={setQ}
          bounds={bounds.q}
          integer
        />
        <SelectField
          name="pre_selected"
          label="Predictors chosen beforehand"
          value={pre}
          options={[
            { value: "", label: "Not stated" },
            ...PRE_SELECTED.map((p) => ({ value: p, label: PRE_NAMES[p] })),
          ]}
          onChange={setPre}
        />
      </div>
      <Choice
        card={card}
        field="method"
        legend="Method"
        values={METHODS}
        names={NAMES}
        offerOf={offerOf}
        recorded={on?.method ?? null}
        testId="options-method"
      />
    </>
  );
}

export const SelectionCard: LineCard = ({ pid, line }) => {
  const card = useCard(pid, "set_selection");
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
