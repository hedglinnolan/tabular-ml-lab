/**
 * `set_batch`: how a batch column is handled (core/decisions.py `SetBatch`;
 * core/methods/batch.py). A counted Decide in Models under an assay lens and either goal, where a
 * column other than the outcome is named as a batch, a run or a plate (core/quest.py
 * `_batch_applies`).
 *
 * The card asks which column and which method. The engine serves the card no reading of which
 * column names a batch (its name reading, methods/batch.py `batch_columns`, places the line), so
 * the card lists every column but the outcome, in the table's order, none chosen, as the
 * validation scheme's group column is chosen; the column on record stays choosable. The methods are
 * the four of the batch contract, in its declared order and with its labels (core/methods/batch.py
 * `_register`), then the model's fifth, that the column is not a batch at all, in the words of the
 * engine's own way forward for it. Under inference the card also asks whether ComBat with the
 * outcome protected serves the figures (never the tests), which the engine reads with batch as a
 * covariate alone (methods/batch.py `figure_for`) and refuses under prediction, so it is asked
 * under inference only and sent with the covariate alone, starting at the record's.
 *
 * A press the engine refuses (a column perfectly confounded with the outcome, ComBat with the
 * outcome protected for testing or under prediction, batch as a covariate before its role and
 * codes are settled) is answered in its own words at the option, with its ways forward.
 */
import { useState } from "react";
import type { Decision, ProjectState } from "../../../api/schema";
import { Pending } from "../../../components/record/blocks";
import type { LineCard } from "../../contracts";
import { Choice, Frame, useCard, type CardWiring, type Offer } from "../predict/card";
import c from "../predict/declaration.module.css";
import { CheckField, SelectField } from "../predict/fields";

export type Batch = Extract<Decision, { kind: "set_batch" }>;

type Exactly<L extends readonly unknown[], U> = [U] extends [L[number]]
  ? [L[number]] extends [U]
    ? L
    : never
  : never;

// The batch contract's declared order (core/methods/batch.py `_register`), then the model's fifth.
const methods = ["covariate", "reference_combat", "outcome_combat", "none", "not_a_batch"] as const;
export const BATCH_METHODS: Exactly<typeof methods, Batch["method"]> = methods;

/** The batch contract's labels (core/methods/batch.py `_register`), and for a column that is not a
 *  batch the engine's own way forward's (methods/batch.py `confounding_refusal`). */
const NAMES: Record<Batch["method"], string> = {
  covariate: "As a covariate",
  reference_combat: "Reference ComBat in-fold",
  outcome_combat: "ComBat, outcome protected",
  none: "Leave it",
  not_a_batch: "Not a batch: read it as an ordinary column",
};
const COLUMN = "Batch column";
/** The engine's own way forward's words for the figures-only answer (methods/batch.py
 *  `_batch_answer_fits`: "Batch as a covariate; ComBat for figures only"). */
const FIGURES = "ComBat for figures only, with batch as a covariate";

/** The decision for `column` and `method`: ComBat for the figures is sent with batch as a covariate
 *  alone, the one method the engine reads it with. */
export function batch(column: string, method: Batch["method"], figures: boolean): Batch {
  return { kind: "set_batch", column, method, figures: method === "covariate" && figures };
}

function Body({ card, state, answered }: { card: CardWiring; state: ProjectState; answered: boolean }) {
  const on = answered ? state.batch : null;
  const [chosen, setChosen] = useState(on?.column ?? "");
  const inference = state.purpose === "inference";
  const [figures, setFigures] = useState(inference && !!on?.figures);
  const listed = card.columns.filter((col) => col !== state.target);
  const columns = on && !listed.includes(on.column) ? [...listed, on.column] : listed;
  const column = columns.includes(chosen) ? chosen : null;
  const recorded = on && column === on.column ? on.method : null;
  const offerOf = (method: Batch["method"]): Offer =>
    column !== null
      ? { decision: batch(column, method, inference && figures) }
      : { problems: [`${COLUMN}: choose a column`] };
  return (
    <>
      <div className={c.fields}>
        <SelectField
          name="column"
          label={COLUMN}
          value={column ?? ""}
          invalid={card.held !== null && column === null}
          options={[{ value: "", label: "Choose a column" }, ...columns.map((col) => ({ value: col, label: col }))]}
          onChange={setChosen}
        />
        {inference ? <CheckField name="figures" label={FIGURES} checked={figures} onChange={setFigures} /> : null}
      </div>
      {/* Each column starts at its own record, so another one chosen starts the options again. */}
      <Choice
        key={column ?? ""}
        card={card}
        field="method"
        legend="Method"
        values={BATCH_METHODS}
        names={NAMES}
        offerOf={offerOf}
        recorded={recorded}
        testId="options-batch"
      />
    </>
  );
}

export const BatchCard: LineCard = ({ pid, line }) => {
  const card = useCard(pid, "set_batch");
  return (
    <Frame line={line}>
      {card.state ? (
        <Body
          key={line.decision_id ?? "open"}
          card={card}
          state={card.state}
          answered={line.status === "answered"}
        />
      ) : (
        <Pending>Loading the project…</Pending>
      )}
    </Frame>
  );
};
