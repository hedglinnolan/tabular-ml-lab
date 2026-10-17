/**
 * `set_intended_use`: how the model will be used. Decision support or risk estimation, with the
 * subgroups whose performance is reported and the approach to fairness. The thresholds belong to
 * decision support alone, so a risk estimate never carries them. The engine refuses decision
 * support for an outcome that is not yes/no, in its own words, at the option.
 */
import { useState } from "react";
import type { ProjectState } from "../../../api/schema";
import { Pending } from "../../../components/record/blocks";
import type { LineCard } from "../../contracts";
import { Choice, Frame, useCard, type CardWiring } from "./card";
import c from "./declaration.module.css";
import { ColumnChecks, FieldGroup, NumberField, SelectField } from "./fields";
import {
  BOUNDS,
  DEFAULTS,
  FAIRNESS,
  USES,
  intendedUse,
  parseTyped,
  type IntendedUse,
} from "./payloads";

const USE_NAMES: Record<IntendedUse["use"], string> = {
  risk_estimation: "Risk estimation",
  decision_support: "Decision support",
};
const FAIRNESS_NAMES: Record<IntendedUse["fairness"], string> = {
  subgroup_performance: "Subgroup performance",
  none: "None",
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
  const on = recorded ? state.intended_use : null;
  const [fairness, setFairness] = useState<IntendedUse["fairness"]>(
    on?.fairness ?? DEFAULTS.set_intended_use.fairness,
  );
  const [subgroups, setSubgroups] = useState<ReadonlySet<string>>(new Set(on?.subgroups ?? []));
  const [low, setLow] = useState(text(on?.threshold_low));
  const [high, setHigh] = useState(text(on?.threshold_high));
  const [decision, setDecision] = useState(text(on?.threshold));

  // Any column of the table but the outcome and the identifier can group the performance.
  const groupable = card.columns.filter(
    (col) => col !== state.target && state.roles?.[col] !== "identifier",
  );
  const bounds = BOUNDS.set_intended_use;
  const typed = {
    threshold_low: parseTyped(low, bounds.threshold_low, false),
    threshold_high: parseTyped(high, bounds.threshold_high, false),
    threshold: parseTyped(decision, bounds.threshold, false),
  };
  const toggle = (column: string) =>
    setSubgroups((cur) => {
      const next = new Set(cur);
      if (!next.delete(column)) next.add(column);
      return next;
    });

  const decisionOf = (use: IntendedUse["use"]) => {
    if (use === "decision_support" && Object.values(typed).includes("invalid")) return null;
    const number = (t: typeof typed.threshold) => (t === "invalid" ? null : t);
    return intendedUse(use, {
      threshold_low: number(typed.threshold_low),
      threshold_high: number(typed.threshold_high),
      threshold: number(typed.threshold),
      // the table's order, so the same choice is the same decision
      subgroups: groupable.filter((col) => subgroups.has(col)),
      fairness,
    });
  };

  return (
    <>
      <div className={c.fields}>
        <SelectField
          name="fairness"
          label="Fairness approach"
          value={fairness}
          options={FAIRNESS.map((f) => ({ value: f, label: FAIRNESS_NAMES[f] }))}
          onChange={(v) => setFairness(v as IntendedUse["fairness"])}
        />
      </div>
      <ColumnChecks
        group="subgroups"
        legend="Subgroups to report performance for"
        columns={groupable}
        chosen={subgroups}
        onToggle={toggle}
      />
      <FieldGroup legend="Thresholds, for decision support">
        <NumberField
          name="threshold_low"
          label="Lowest"
          text={low}
          onChange={setLow}
          bounds={bounds.threshold_low}
          integer={false}
        />
        <NumberField
          name="threshold_high"
          label="Highest"
          text={high}
          onChange={setHigh}
          bounds={bounds.threshold_high}
          integer={false}
        />
        <NumberField
          name="threshold"
          label="Decision threshold"
          text={decision}
          onChange={setDecision}
          bounds={bounds.threshold}
          integer={false}
        />
      </FieldGroup>
      <Choice
        card={card}
        field="use"
        legend="Use"
        values={USES}
        names={USE_NAMES}
        decisionOf={decisionOf}
        recorded={on?.use ?? null}
        testId="options-use"
      />
    </>
  );
}

export const IntendedUseCard: LineCard = ({ pid, line }) => {
  const card = useCard(pid, "set_intended_use");
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
