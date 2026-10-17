/**
 * `set_explain`: how the fitted models are described. The curves (accumulated local effects, or
 * partial dependence), the columns whose curves are drawn (none chosen leaves it to the engine,
 * which draws the declared exposure under inference and the top exposures under prediction, as its
 * own sentence for the decision says), and the refits the stability is measured over. An
 * explanation is never reported as an effect, which the engine refuses under both purposes, so the
 * card does not ask it: `as_effect` is always false. One refit has none to compare with, and the
 * engine says so in its own words, with its exits.
 */
import { useState } from "react";
import type { ProjectState } from "../../../api/schema";
import { Pending } from "../../../components/record/blocks";
import type { LineCard } from "../../contracts";
import { Choice, Frame, useCard, type CardWiring, type Offer } from "./card";
import c from "./declaration.module.css";
import { ColumnChecks, NumberField } from "./fields";
import { BOUNDS, CURVES, DEFAULTS, explain, parseTyped, trouble, type Explain } from "./payloads";

const NAMES: Record<Explain["curves"], string> = {
  ale: "Accumulated local effects",
  partial_dependence: "Partial dependence",
};
const PREDICTOR_ROLES = ["exposure", "covariate", "energy"];
const REFITS = "Refits to measure stability";

/** What an empty list of columns draws, as the engine decides it (models/explain.py): the study
 *  factor declared under inference, the predictors that matter most to the model otherwise.
 *  "Exposure" stays a quiet term (FOUNDATION §2). */
const WHEN_NONE_CHOSEN = {
  inference: "the declared study factor",
  prediction: "the predictors that matter most to the model",
} as const;

function Body({
  card,
  state,
  recorded,
}: {
  card: CardWiring;
  state: ProjectState;
  recorded: boolean;
}) {
  const on = recorded ? state.explain : null;
  const [reseeds, setReseeds] = useState(String(on?.reseeds ?? DEFAULTS.set_explain.reseeds));
  const [chosen, setChosen] = useState<ReadonlySet<string>>(new Set(on?.exposures ?? []));
  const typed = parseTyped(reseeds, BOUNDS.set_explain.reseeds, true, true);

  // The columns the models read: by role, the outcome and the columns left out apart.
  const left = new Set(state.missing?.drop_columns ?? []);
  const known = card.columns.length ? card.columns : Object.keys(state.roles ?? {});
  const predictors = known.filter(
    (col) =>
      col !== state.target && !left.has(col) && PREDICTOR_ROLES.includes(state.roles?.[col] ?? ""),
  );
  const whenNoneChosen =
    WHEN_NONE_CHOSEN[state.purpose === "inference" ? "inference" : "prediction"];
  const toggle = (column: string) =>
    setChosen((cur) => {
      const next = new Set(cur);
      if (!next.delete(column)) next.add(column);
      return next;
    });

  const offerOf = (curves: Explain["curves"]): Offer =>
    typed === "invalid"
      ? { problems: trouble(REFITS, typed, BOUNDS.set_explain.reseeds, true) }
      : {
          decision: explain(curves, {
            exposures: predictors.filter((col) => chosen.has(col)),
            reseeds: typed,
          }),
        };

  return (
    <>
      <div className={c.fields}>
        <NumberField
          name="reseeds"
          label={REFITS}
          text={reseeds}
          onChange={setReseeds}
          bounds={BOUNDS.set_explain.reseeds}
          integer
          required
        />
      </div>
      <ColumnChecks
        group="exposures"
        legend={`Curves for (none chosen: ${whenNoneChosen})`}
        columns={predictors}
        chosen={chosen}
        onToggle={toggle}
      />
      <Choice
        card={card}
        field="curves"
        legend="Curves"
        values={CURVES}
        names={NAMES}
        offerOf={offerOf}
        recorded={on?.curves ?? null}
        testId="options-curves"
      />
    </>
  );
}

export const ExplainCard: LineCard = ({ pid, line }) => {
  const card = useCard(pid, "set_explain");
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
