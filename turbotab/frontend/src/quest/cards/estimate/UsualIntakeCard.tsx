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
 * as proposed from their names (or as an answer on record names them), which the component's name
 * on the card lists, so the press that records them names them. Before the plan is locked the stage is served with its distributions
 * withheld and its offer kept (core/estimand.py `withhold`). A component already answered that the
 * offer no longer lists stays choosable, with its days on record; the engine keys a wide answer by
 * any label (core/decisions.py `SetUsualIntake`), so one whose label is not its first day is named
 * by that label before its days, and never reads as an offered component of the same days.
 *
 * The card asks which component and which model. Leaving usual intake out records the component
 * with no model, as the engine's own exit does (core/usual_intake.py `leave_out`). The card does not
 * ask the order or weekend columns, the population (whole, or consumers only with the column that
 * marks them), a cut-off and the EAR's conditions, or the bootstrap's replicates. For a component
 * with an answer on record it keeps them as the record holds them, with its days, and changes the
 * model alone, as the engine changes one field of a record (core/usual_intake.py `answer`); for one
 * with none it sends the model's defaults and the offer's days.
 *
 * What an answer on record cannot run on this table (the two-part model with no zero day, the
 * weekend column's values, the survey question unanswered) the stage serves as a refused analysis,
 * kept before the lock since it asks for an answer (core/estimand.py `withhold`). The card says each
 * in the engine's words (the analysis's methods sentence, which names the component and the
 * refusal), then each way forward that names no decision in words (or, where it names another
 * card and the shell opens cards, as a press that opens it: G12), and offers each that is a
 * decision as a press that records it as served. It shows no analysis that applies, and so no
 * number. A press the engine refuses (under prediction, a column the table lacks) is answered in its
 * own words, with its exits.
 *
 * Every record recomputes the stage, and meanwhile the server serves its older result, so the card
 * is veiled and inert until the fresh result arrives (StaleVeil), with the retry of a recompute that
 * failed or was stopped (components/StageRetry.tsx), as the Model 1 card is.
 */
import { Fragment, useState } from "react";
import { useProjectView, useStageResult } from "../../../api/queries";
import type { UsualIntakeArtifact } from "../../../api/m3-types";
import type { Decision, ProjectState, ProjectView, QuestLine } from "../../../api/schema";
import { StageFailure, rootFailure } from "../../../components/Failure";
import { Pending } from "../../../components/record/blocks";
import { targetName, targetOf, useOpenTarget, type ExitTarget } from "../../../components/record/target";
import { Prose } from "../../../components/Prose";
import { StageRetry } from "../../../components/StageRetry";
import r from "../../../components/record/Refusal.module.css";
import { StaleVeil, veilFor } from "../../../motion/StaleVeil";
import type { LineCard } from "../../contracts";
import { Choice, Frame, useCard, type CardWiring, type Offer } from "../predict/card";
import c from "../predict/declaration.module.css";
import { SelectField } from "../predict/fields";

export type UsualIntake = Extract<Decision, { kind: "set_usual_intake" }>;
type UsualIntakeOffer = UsualIntakeArtifact["offer"];
type Analysis = UsualIntakeArtifact["analyses"][number];
/** An answer as the state holds it, every field present (core/decisions.py `UsualIntakeSpec`). */
type Spec = NonNullable<ProjectState["usual_intake"]>[string];
/** The fields the card does not ask. */
type Unasked = Required<Omit<UsualIntake, "kind" | "nutrient" | "model" | "days">>;

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
} as const satisfies Unasked;

/** The fields of a record the card does not ask, as the record holds them. */
function unasked(spec: Spec): Unasked {
  return {
    order_column: spec.order_column,
    weekend: [...spec.weekend],
    weekend_coding: spec.weekend_coding,
    population: spec.population,
    consumer_column: spec.consumer_column,
    cutoff: spec.cutoff,
    cutoff_kind: spec.cutoff_kind,
    n_boot: spec.n_boot,
    ear_for_all: spec.ear_for_all,
    ear_symmetric: spec.ear_symmetric,
  };
}

/** A component the card offers: the name an answer is keyed by, and a wide table's recall days (the
 *  record's, where an answer on record names them, else the offer's). */
export interface Component {
  nutrient: string;
  days: readonly string[];
  /** The engine's reason for the model it ranks first, where the offer gives one. */
  why: string | null;
}

/**
 * The decision for `component` and `model`. None sends no day columns and the model's defaults, as
 * the engine's own "Leave usual intake out" does. Any other model keeps every field `record` holds
 * that the card does not ask, as the engine's `answer` does, else sends the model's defaults.
 */
export function usualIntake(
  component: Pick<Component, "nutrient" | "days">,
  model: UsualIntake["model"],
  record: Spec | null = null,
): UsualIntake {
  const kept = model !== "none" && record ? unasked(record) : { ...DEFAULTS, weekend: [...DEFAULTS.weekend] };
  return {
    kind: "set_usual_intake",
    nutrient: component.nutrient,
    model,
    days: model === "none" ? [] : [...component.days],
    ...kept,
  };
}

/** The days an answer on record names: none where it leaves usual intake out (its days are the
 *  offer's to propose again). */
const daysOn = (spec: Spec | undefined): readonly string[] | null => (spec && spec.model !== "none" ? spec.days : null);

/** The offer's components in its order, then each one answered that it no longer lists. */
function componentsOf(offer: UsualIntakeOffer, answers: ProjectState["usual_intake"]): Component[] {
  const out: Component[] = offer.candidates.map((cand) => ({
    nutrient: cand.column,
    days: daysOn(answers?.[cand.column]) ?? cand.days,
    why: cand.why ?? null,
  }));
  for (const [nutrient, spec] of Object.entries(answers ?? {})) {
    if (!out.some((comp) => comp.nutrient === nutrient)) out.push({ nutrient, days: spec.days, why: null });
  }
  return out;
}

/** A component's name: a long table's column; a wide table's days, after its own label where that
 *  is not its first day (the offer names each of its components by its first day). */
function nameOf(comp: Component): string {
  if (!comp.days.length) return comp.nutrient;
  const days = comp.days.join(", ");
  return comp.nutrient === comp.days[0] ? days : `${comp.nutrient} (${days})`;
}

/** A way forward from a refused analysis, as the stage serves it: a label, a decision or none (a
 *  question to answer), and the card one that names no decision opens, where it names one (G12;
 *  record/target.ts). */
interface Way {
  label: string;
  decision: Decision | null;
  target: ExitTarget | null;
}

function waysOf(analysis: Analysis): Way[] {
  const ways: Way[] = [];
  for (const served of analysis.exits) {
    const { label, decision } = served;
    if (typeof label !== "string") continue;
    const decided = typeof decision === "object" && decision !== null && typeof (decision as { kind?: unknown }).kind === "string";
    ways.push({ label, decision: decided ? (decision as Decision) : null, target: targetOf(served) });
  }
  return ways;
}

/** A sentence ends with its stop. */
const finish = (text: string) => (/[.!?]$/.test(text.trim()) ? text.trim() : `${text.trim()}.`);

/** An analysis the stage refused: the engine's sentence of it, each way forward that names no
 *  decision in words after it, and a press for each that is a decision. One that names no decision
 *  but the card it opens (the survey question's, core/stages/usual_intake.py) is, where the shell
 *  opens cards, a press that opens it instead of a sentence; one that names this card's own line
 *  (`own`: naming the component's columns, `toward("set_usual_intake")`) is said in words, since
 *  this card's fields answer it and a press would open only the card it sits in. A press the
 *  server does not record is answered here, in its words. */
function Refused({ card, analysis, own }: { card: CardWiring; analysis: Analysis; own: string }) {
  const opener = useOpenTarget();
  const ways = waysOf(analysis).map((x) => ({ ...x, open: x.target?.key === own ? null : opener(x) }));
  const at = (i: number) => `refused:${analysis.nutrient}:${i}`;
  const answer = card.answerAt && ways.some((_, i) => card.answerAt!.key === at(i)) ? card.answerAt.node : null;
  return (
    <div className={r.refusal} role="alert" data-testid="usual-intake-refused" data-nutrient={analysis.nutrient}>
      <p className={r.text}>
        <Prose text={finish(analysis.methods || analysis.refused || "")} />
        {ways
          .filter((x) => !x.decision && !x.open)
          .map((x) => (
            <Fragment key={x.label}>
              {" "}
              <Prose text={finish(x.label)} />
            </Fragment>
          ))}
      </p>
      {ways.some((x) => x.decision || x.open) ? (
        <div className={r.exits}>
          {ways.map((x, i) =>
            x.decision ? (
              <button
                key={x.label}
                type="button"
                className={r.exit}
                disabled={card.pending}
                onClick={() => card.record(x.decision!, at(i))}
                data-testid="usual-intake-refused-exit"
              >
                <Prose text={x.label} />
              </button>
            ) : x.open && x.target ? (
              <button
                key={x.label}
                type="button"
                className={r.ghost}
                onClick={x.open}
                data-testid="usual-intake-refused-to"
                data-target={targetName(x.target)}
              >
                <Prose text={x.label} />
              </button>
            ) : null,
          )}
        </div>
      ) : null}
      {answer}
    </div>
  );
}

function Body({
  card,
  offer,
  analyses,
  answered,
  initial,
  own,
}: {
  card: CardWiring;
  offer: UsualIntakeOffer;
  analyses: readonly Analysis[];
  answered: boolean;
  /** The component of the line's own record, once the line is answered. */
  initial: string;
  /** The line's key, the card a way forward names when it names this one. */
  own: string;
}) {
  const answers = card.state?.usual_intake ?? null;
  const components = componentsOf(offer, answers);
  const [chosen, setChosen] = useState(components.some((comp) => comp.nutrient === initial) ? initial : "");
  const component = components.find((comp) => comp.nutrient === chosen) ?? null;
  const record = component ? (answers?.[component.nutrient] ?? null) : null;
  const recorded = answered ? (record?.model ?? null) : null;
  const offerOf = (model: UsualIntake["model"]): Offer =>
    component ? { decision: usualIntake(component, model, record) } : { problems: [`${COMPONENT}: choose a component`] };
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
      {analyses
        .filter((a) => a.refused)
        .map((a) => (
          <Refused key={a.nutrient} card={card} analysis={a} own={own} />
        ))}
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
  const analyses = stage.data?.artifact?.analyses ?? [];
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
            analyses={analyses}
            answered={line.status === "answered"}
            initial={initial}
            own={line.key}
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
