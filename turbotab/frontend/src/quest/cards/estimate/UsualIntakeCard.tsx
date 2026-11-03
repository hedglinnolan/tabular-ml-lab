/**
 * `set_usual_intake`: one dietary component's usual-intake distribution by the NCI method, or none
 * (core/decisions.py `SetUsualIntake`; core/stages/usual_intake.py), its own estimand beside any
 * association. A counted Decide in Models under the dietary lens and any goal but prediction, where
 * the usual-intake stage offers the distribution, or once an answer is recorded (core/quest.py
 * `_usual_intake_applies`).
 *
 * The usual-intake stage serves its offer (core/stages/usual_intake.py `build_offer`): what the
 * distribution estimates, why it is not offered where it is not, and each dietary component the
 * recalls measure, with the engine's reason for the model it ranks first. A long table's component
 * is its column; a wide table's is a group of recall-day columns, named by its first day, its days
 * as proposed from their names, which the component's name on the card lists, so the press that
 * records them names them. Before the plan is locked the stage is served with its distributions
 * withheld and its offer kept (core/estimand.py `withhold`). A component already answered that the
 * offer no longer lists stays choosable, with its days on record.
 *
 * The card asks which component and which model. Leaving usual intake out records the component
 * with no model, as the engine's own exit does (core/usual_intake.py `leave_out`). The card does not
 * ask the order or weekend columns, the population (whole, or consumers only with the column that
 * marks them), a cut-off and the EAR's conditions, or the bootstrap's replicates: it sends the
 * model's defaults for them. What an answer cannot run on this table (the two-part model with no
 * zero day, the survey question unanswered) is said by the stage's analysis, with its exits, which
 * this card does not show: it reads the offer alone. A press the engine refuses (under prediction,
 * a column the table lacks) is answered in its own words, with its exits.
 *
 * Every record recomputes the stage, and meanwhile the server serves its older result, so the card
 * is veiled and inert until the fresh result arrives (StaleVeil), with the retry of a recompute that
 * failed or was stopped (components/StageRetry.tsx), as the Model 1 card is.
 */
import { useState } from "react";
import { useProjectView, useStageResult } from "../../../api/queries";
import type { UsualIntakeArtifact } from "../../../api/m3-types";
import type { Decision, ProjectState, ProjectView, QuestLine } from "../../../api/schema";
import { StageFailure, rootFailure } from "../../../components/Failure";
import { Pending } from "../../../components/record/blocks";
import { Prose } from "../../../components/Prose";
import { StageRetry } from "../../../components/StageRetry";
import { StaleVeil, veilFor } from "../../../motion/StaleVeil";
import type { LineCard } from "../../contracts";
import { Choice, Frame, useCard, type CardWiring, type Offer } from "../predict/card";
import c from "../predict/declaration.module.css";
import { SelectField } from "../predict/fields";

export type UsualIntake = Extract<Decision, { kind: "set_usual_intake" }>;
type UsualIntakeOffer = UsualIntakeArtifact["offer"];

type Exactly<L extends readonly unknown[], U> = [U] extends [L[number]]
  ? [L[number]] extends [U]
    ? L
    : never
  : never;

// The contract's order (core/usual_intake.py `_contract`), and leaving it out last, as the engine
// lists its exits.
const models = ["amount_only", "two_part", "none"] as const;
export const USUAL_INTAKE_MODELS: Exactly<typeof models, UsualIntake["model"]> = models;

/** The contract's labels for the two models (core/usual_intake.py `_contract`), and the engine's
 *  own exit's for none (core/usual_intake.py `leave_out`). */
const NAMES: Record<UsualIntake["model"], string> = {
  amount_only: "Amount-only model",
  two_part: "Two-part model (episodically consumed foods)",
  none: "Leave usual intake out",
};
const COMPONENT = "Dietary component";

/** The model's defaults for the fields the card does not ask (core/decisions.py `SetUsualIntake`). */
export const DEFAULTS = {
  order_column: null,
  weekend: [] as string[],
  weekend_coding: "indicator",
  population: "whole",
  consumer_column: null,
  cutoff: null,
  cutoff_kind: null,
  n_boot: 200,
  ear_for_all: false,
  ear_symmetric: false,
} as const satisfies Omit<UsualIntake, "kind" | "nutrient" | "model" | "days">;

/** A component the card offers: the name an answer is keyed by, and a wide table's recall days. */
export interface Component {
  nutrient: string;
  days: readonly string[];
  /** The engine's reason for the model it ranks first, where the offer gives one. */
  why: string | null;
}

/**
 * The decision for `component` and `model`. None sends no day columns, as the engine's own "Leave
 * usual intake out" does; every field the card does not ask is the model's default.
 */
export function usualIntake(component: Pick<Component, "nutrient" | "days">, model: UsualIntake["model"]): UsualIntake {
  return {
    kind: "set_usual_intake",
    nutrient: component.nutrient,
    model,
    days: model === "none" ? [] : [...component.days],
    ...DEFAULTS,
    weekend: [...DEFAULTS.weekend],
  };
}

/** The offer's components in its order, then each one answered that it no longer lists. */
function componentsOf(offer: UsualIntakeOffer, answers: ProjectState["usual_intake"]): Component[] {
  const out: Component[] = offer.candidates.map((cand) => ({
    nutrient: cand.column,
    days: cand.days,
    why: cand.why ?? null,
  }));
  for (const [nutrient, spec] of Object.entries(answers ?? {})) {
    if (!out.some((comp) => comp.nutrient === nutrient)) out.push({ nutrient, days: spec.days, why: null });
  }
  return out;
}

const nameOf = (comp: Component) => (comp.days.length ? comp.days.join(", ") : comp.nutrient);

function Body({
  card,
  offer,
  answered,
  initial,
}: {
  card: CardWiring;
  offer: UsualIntakeOffer;
  answered: boolean;
  /** The component of the line's own record, once the line is answered. */
  initial: string;
}) {
  const answers = card.state?.usual_intake ?? null;
  const components = componentsOf(offer, answers);
  const [chosen, setChosen] = useState(components.some((comp) => comp.nutrient === initial) ? initial : "");
  const component = components.find((comp) => comp.nutrient === chosen) ?? null;
  const recorded = answered && component ? (answers?.[component.nutrient]?.model ?? null) : null;
  const offerOf = (model: UsualIntake["model"]): Offer =>
    component ? { decision: usualIntake(component, model) } : { problems: [`${COMPONENT}: choose a component`] };
  return (
    <>
      <p className={c.said} data-testid="usual-intake-estimand">
        <Prose text={offer.estimand} />
      </p>
      {!offer.offered && offer.reason ? (
        <p className={c.said} data-testid="usual-intake-unoffered">
          <Prose text={offer.reason} />
        </p>
      ) : null}
      <div className={c.fields}>
        <SelectField
          name="nutrient"
          label={COMPONENT}
          value={chosen}
          invalid={card.held !== null && component === null}
          options={[
            { value: "", label: "Choose a component" },
            ...components.map((comp) => ({ value: comp.nutrient, label: nameOf(comp) })),
          ]}
          onChange={setChosen}
        />
      </div>
      {component?.why ? (
        <p className={c.said} data-testid="usual-intake-why">
          <Prose text={component.why} />
        </p>
      ) : null}
      {/* Each component starts at its own record, so another one chosen starts the options again. */}
      <Choice
        key={chosen}
        card={card}
        field="model"
        legend="Model"
        values={USUAL_INTAKE_MODELS}
        names={NAMES}
        offerOf={offerOf}
        recorded={recorded}
        testId="options-usual-intake"
      />
    </>
  );
}

/** The usual-intake stage's result, once the stage has started (state/stages.ts `useStage`). */
function useUsualIntake(pid: string, view: ProjectView | undefined) {
  const status = view?.stages.usual_intake?.status;
  return useStageResult(pid, "usual_intake", status !== undefined && status !== "idle" && status !== "blocked");
}

/** The component the line's own record answers, or "" where the line is not answered. */
function recordedComponent(view: ProjectView | undefined, line: QuestLine): string {
  if (line.status !== "answered" || !line.decision_id) return "";
  const d = view?.decisions.find((r) => r.id === line.decision_id)?.decision;
  return d?.kind === "set_usual_intake" ? d.nutrient : "";
}

/** The line's name as its card's heading says it (predict/card.tsx `Frame`). */
const capital = (s: string) => `${s.charAt(0).toUpperCase()}${s.slice(1)}`;

export const UsualIntakeCard: LineCard = ({ pid, line }) => {
  const card = useCard(pid, "set_usual_intake");
  const view = useProjectView(pid).data;
  const stage = useUsualIntake(pid, view);
  const offer = stage.data?.artifact?.offer ?? null;
  const initial = recordedComponent(view, line);
  // A stage that failed or was stopped does not restart by itself (components/StageRetry.tsx).
  const status = view?.stages.usual_intake;
  const failed = !!view && rootFailure(view.stages, "usual_intake") !== null;
  const stopped = !failed && !!status?.cancelled;
  return (
    <Frame line={line}>
      {!card.state || (stage.isLoading && !offer) ? (
        <Pending>Loading the usual-intake question…</Pending>
      ) : offer ? (
        <StaleVeil
          state={veilFor(status, stage.data)}
          testId="usual-intake-veil"
          action={<StageRetry pid={pid} status={status} />}
        >
          <Body
            key={`${line.decision_id ?? "open"} ${initial}`}
            card={card}
            offer={offer}
            answered={line.status === "answered"}
            initial={initial}
          />
        </StaleVeil>
      ) : failed && view ? (
        // Nothing in the quest shell answers a request to reopen an answer (the Record does), so
        // the failure offers its retry alone.
        <StageFailure
          pid={pid}
          view={view}
          stage="usual_intake"
          changeable={false}
          after={<>{capital(line.name)} waits on it.</>}
          testId="usual-intake-failure"
        />
      ) : stopped ? (
        <p className={c.waiting} role="alert" data-testid="usual-intake-stopped">
          <strong>You stopped the recompute</strong> before it finished. {capital(line.name)} waits on it.{" "}
          <StageRetry pid={pid} status={status} />
        </p>
      ) : (
        <p className={c.waiting} data-testid="usual-intake-unserved">
          The dietary components the recalls measure are not computed yet.
        </p>
      )}
    </Frame>
  );
};
