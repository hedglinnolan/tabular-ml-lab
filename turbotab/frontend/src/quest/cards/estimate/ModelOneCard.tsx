/**
 * `set_model_sequence`: the Model 1 question, a counted Decide in Models under Estimate
 * (core/quest.py). The proposals stage serves its declaration (core/estimand.py
 * `model_sequence_card`): the columns Model 1 may hold (the primary's adjusted covariates and total
 * energy), the guess at the field's age, sex and energy among them, the one-tap decision that
 * declares that guess, and the engine's reason. Model 2 is the primary and Model 3 adds the possible
 * mediators, both from the answers already given, so Model 1's columns are all the card asks.
 *
 * The guess is a suggestion beside an option, never a column checked for the person: its own option
 * records the one-tap decision as served, and the other records the columns checked, in the
 * engine's order, for the exposure the served decision names (none checked declares Model 1 with no
 * column, as the model takes it). An empty guess suggests nothing, so only the columns checked are
 * offered. A column outside the primary's adjustment set the engine refuses in its own words, with
 * its exits.
 */
import { useState } from "react";
import { useProjectView, useStageResult } from "../../../api/queries";
import type { ProjectView } from "../../../api/schema";
import { Pending } from "../../../components/record/blocks";
import { Prose } from "../../../components/Prose";
import type { LineCard } from "../../contracts";
import { Choice, Frame, useCard, type CardWiring, type Offer } from "../predict/card";
import c from "../predict/declaration.module.css";
import { ColumnChecks } from "../predict/fields";
import { modelSequence, servedSequence, type ModelOneServed, type ModelSequence } from "./payloads";

type Answer = "guess" | "checked";

const listed = (columns: readonly string[]) => columns.join(", ");
const sameColumns = (a: readonly string[], b: readonly string[]) =>
  a.length === b.length && a.every((col) => b.includes(col));

function Body({
  card,
  served,
  guess,
  answered,
}: {
  card: CardWiring;
  served: ModelOneServed;
  /** The served one-tap decision, which declares the guess. */
  guess: ModelSequence;
  answered: boolean;
}) {
  // The columns declared, as the engine holds them current, once the line is answered.
  const declared = answered ? served.declared : null;
  const [checked, setChecked] = useState<ReadonlySet<string>>(new Set(declared ?? []));
  const columns = served.allowed.filter((col) => checked.has(col));
  const values: readonly Answer[] = guess.model_1.length ? ["guess", "checked"] : ["checked"];
  const names: Record<Answer, string> = {
    guess: `${listed(guess.model_1)} (suggested)`,
    checked: columns.length ? `${listed(columns)} (checked above)` : "No column (none checked above)",
  };
  const recorded: Answer | null =
    declared === null
      ? null
      : values.includes("guess") && sameColumns(declared, guess.model_1)
        ? "guess"
        : "checked";
  const toggle = (column: string) =>
    setChecked((cur) => {
      const next = new Set(cur);
      if (!next.delete(column)) next.add(column);
      return next;
    });
  const offerOf = (answer: Answer): Offer => ({
    decision: answer === "guess" ? guess : modelSequence(guess.exposure, columns),
  });
  return (
    <>
      <p className={c.said} data-testid="model-one-reason">
        <Prose text={served.reason} />
      </p>
      <ColumnChecks
        group="model_1"
        legend="Model 1 adjusts for"
        columns={served.allowed}
        chosen={checked}
        onToggle={toggle}
      />
      <Choice
        card={card}
        field="model_1"
        legend="Model 1's columns"
        values={values}
        names={names}
        offerOf={offerOf}
        recorded={recorded}
        testId="options-model-one"
      />
    </>
  );
}

/** The proposals stage's result, once the stage has started (state/stages.ts `useStage`). */
function useProposals(pid: string, view: ProjectView | undefined) {
  const status = view?.stages.proposals?.status;
  return useStageResult(pid, "proposals", status !== undefined && status !== "idle" && status !== "blocked");
}

export const ModelOneCard: LineCard = ({ pid, line }) => {
  const card = useCard(pid, "set_model_sequence");
  const proposals = useProposals(pid, useProjectView(pid).data);
  const served = proposals.data?.artifact?.model_sequence ?? null;
  const guess = served ? servedSequence(served) : null;
  return (
    <Frame line={line}>
      {!card.state || (proposals.isLoading && !served) ? (
        <Pending>Loading the Model 1 question…</Pending>
      ) : served && guess ? (
        <Body
          key={line.decision_id ?? "open"}
          card={card}
          served={served}
          guess={guess}
          answered={line.status === "answered"}
        />
      ) : (
        <p className={c.waiting} data-testid="model-one-unserved">
          The columns Model 1 may hold are not computed yet.
        </p>
      )}
    </Frame>
  );
};
