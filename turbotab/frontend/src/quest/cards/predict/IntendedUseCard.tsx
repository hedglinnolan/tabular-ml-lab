/**
 * `set_intended_use`: how the model will be used. Decision support or risk estimation, with the
 * subgroups whose performance is reported and the approach to fairness. The thresholds belong to
 * decision support alone, so a risk estimate never carries them. The engine refuses decision
 * support for an outcome whose task is known and not yes/no (models/decision_curve.py
 * `_intended_use_fits`: the task recorded, else the one the target stage detected), so for such an
 * outcome the card offers neither it nor its thresholds; while no task is known, the engine's
 * refusal answers at the option, in its own words.
 *
 * The subgroups lead with the columns the engine groups by their levels (a categorical, yes/no or
 * text column, one of two values, one declared categorical or confirmed as codes: models/
 * decision_curve.py `grouping_of`), then the amounts it groups by thirds, each in the table's
 * order; the outcome, the identifier and the flags on other columns are not offered. A column's
 * role is the one recorded, else, while the roles are not recorded (the engine opens this card
 * once the goal is chosen, before the roles in Your data), the one the roles stage proposes.
 */
import { useState } from "react";
import { useProjectView, useStageResult } from "../../../api/queries";
import type { ColumnSummary, ProjectState } from "../../../api/schema";
import { takesLevel } from "../../../components/record/ask/ChoiceQuestions";
import { Pending } from "../../../components/record/blocks";
import type { LineCard } from "../../contracts";
import { Choice, Frame, useCard, type CardWiring, type Offer } from "./card";
import c from "./declaration.module.css";
import { ColumnChecks, FieldGroup, NumberField, SelectField } from "./fields";
import {
  BOUNDS,
  DEFAULTS,
  FAIRNESS,
  USES,
  intendedUse,
  parseTyped,
  trouble,
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
// A threshold's field label, and the name the sentence at the option calls it by.
const THRESHOLDS = {
  threshold_low: { label: "Lowest", name: "Lowest threshold" },
  threshold_high: { label: "Highest", name: "Highest threshold" },
  threshold: { label: "Decision threshold", name: "Decision threshold" },
} as const;
const text = (n: number | null | undefined) => (n === null || n === undefined ? "" : String(n));

/** Roles whose columns group no performance: the unit's identifier, and a flag on another column. */
const NOT_SUBGROUPS = new Set(["identifier", "flag"]);

/** Whether the engine groups a column by its levels rather than its thirds (`grouping_of`): its
 *  type or its two values say so, or the person declared it categorical or confirmed it codes. */
function byLevels(
  column: string,
  summary: ColumnSummary | undefined,
  state: ProjectState,
): boolean {
  return (
    takesLevel(summary) ||
    (state.categorical ?? []).includes(column) ||
    state.reading_confirmations?.[`code_or_count:${column}`] === "code"
  );
}

/** The columns that can group the performance, those grouped by their levels first, each in the
 *  table's order. A column's role is the one recorded, else the one `proposed` (the roles stage's). */
export function subgroupColumns(
  columns: readonly string[],
  summaries: ReadonlyMap<string, ColumnSummary>,
  state: ProjectState,
  proposed: ReadonlyMap<string, string> = new Map(),
): string[] {
  const roleOf = (col: string) => state.roles?.[col] ?? proposed.get(col) ?? "";
  const offered = columns.filter((col) => col !== state.target && !NOT_SUBGROUPS.has(roleOf(col)));
  const levels = offered.filter((col) => byLevels(col, summaries.get(col), state));
  return [...levels, ...offered.filter((col) => !levels.includes(col))];
}

/** Decision support is the engine's for a yes/no outcome, or while the task is not known yet. */
export function offersDecisionSupport(task: ProjectState["task"] | undefined): boolean {
  return task === null || task === undefined || task === "binary";
}

function Body({
  card,
  state,
  recorded,
  proposed,
}: {
  card: CardWiring;
  state: ProjectState;
  recorded: boolean;
  proposed: ReadonlyMap<string, string>;
}) {
  const on = recorded ? state.intended_use : null;
  const [fairness, setFairness] = useState<IntendedUse["fairness"]>(
    on?.fairness ?? DEFAULTS.set_intended_use.fairness,
  );
  const [subgroups, setSubgroups] = useState<ReadonlySet<string>>(new Set(on?.subgroups ?? []));
  const [low, setLow] = useState(text(on?.threshold_low));
  const [high, setHigh] = useState(text(on?.threshold_high));
  const [decision, setDecision] = useState(text(on?.threshold));

  const groupable = subgroupColumns(card.columns, card.summaries, state, proposed);
  const support = offersDecisionSupport(card.task);
  const uses = support ? USES : USES.filter((u) => u !== "decision_support");
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

  const offerOf = (use: IntendedUse["use"]): Offer => {
    // The thresholds are read by decision support alone, so only there do they have to be numbers.
    const problems =
      use === "decision_support"
        ? (Object.keys(THRESHOLDS) as (keyof typeof THRESHOLDS)[]).flatMap((k) =>
            trouble(THRESHOLDS[k].name, typed[k], bounds[k], false),
          )
        : [];
    if (problems.length > 0) return { problems };
    const number = (t: typeof typed.threshold) => (t === "invalid" ? null : t);
    return {
      decision: intendedUse(use, {
        threshold_low: number(typed.threshold_low),
        threshold_high: number(typed.threshold_high),
        threshold: number(typed.threshold),
        // the table's order, so the same choice is the same decision
        subgroups: card.columns.filter((col) => groupable.includes(col) && subgroups.has(col)),
        fairness,
      }),
    };
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
      {support ? (
        <FieldGroup legend="Thresholds, for decision support">
          <NumberField
            name="threshold_low"
            label={THRESHOLDS.threshold_low.label}
            text={low}
            onChange={setLow}
            bounds={bounds.threshold_low}
            integer={false}
          />
          <NumberField
            name="threshold_high"
            label={THRESHOLDS.threshold_high.label}
            text={high}
            onChange={setHigh}
            bounds={bounds.threshold_high}
            integer={false}
          />
          <NumberField
            name="threshold"
            label={THRESHOLDS.threshold.label}
            text={decision}
            onChange={setDecision}
            bounds={bounds.threshold}
            integer={false}
          />
        </FieldGroup>
      ) : null}
      <Choice
        card={card}
        field="use"
        legend="Use"
        values={uses}
        names={USE_NAMES}
        offerOf={offerOf}
        recorded={on?.use ?? null}
        testId="options-use"
      />
    </>
  );
}

/** Each column's role as the roles stage proposes it, once that stage has started (state/stages.ts
 *  `useStage`'s rule: a blocked stage's older result, if any, stays cached). */
function useProposedRoles(pid: string): ReadonlyMap<string, string> {
  const status = useProjectView(pid).data?.stages.roles?.status;
  const started = status !== undefined && status !== "idle" && status !== "blocked";
  const roles = useStageResult(pid, "roles", started).data?.artifact;
  return new Map((roles?.columns ?? []).map((c) => [c.column, c.proposed]));
}

export const IntendedUseCard: LineCard = ({ pid, line }) => {
  const card = useCard(pid, "set_intended_use");
  const proposed = useProposedRoles(pid);
  return (
    <Frame line={line}>
      {card.state ? (
        <Body
          key={line.decision_id ?? "open"}
          card={card}
          state={card.state}
          recorded={line.status === "answered"}
          proposed={proposed}
        />
      ) : (
        <Pending>Loading the project…</Pending>
      )}
    </Frame>
  );
};
