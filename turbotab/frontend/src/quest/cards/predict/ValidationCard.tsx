/**
 * `set_validation`: how the training rows validate a model. This is the scheme and only the scheme
 * (W1_PART2_PLAN ruling 10): the draw (the holdout and its seed) is the seal's, and `set_validation`
 * writes the scheme onto it and keeps the draw. The fields start at the scheme as it stands in the
 * split, or at the model's defaults where no split is recorded yet. Folds, repeats and resamples
 * are sent with every scheme (the model holds all three); a column whose groups are the folds is
 * sent with internal–external validation alone, which needs one, and that option, pressed without
 * one, says so.
 */
import { useState } from "react";
import type { ProjectState } from "../../../api/schema";
import { Pending } from "../../../components/record/blocks";
import type { LineCard } from "../../contracts";
import { Choice, Frame, useCard, type CardWiring, type Offer } from "./card";
import c from "./declaration.module.css";
import { CheckField, NumberField, SelectField } from "./fields";
import {
  BOUNDS,
  DEFAULTS,
  VALIDATIONS,
  parseTyped,
  trouble,
  validation,
  type Validation,
} from "./payloads";

const NAMES: Record<Validation["validation"], string> = {
  kfold: "k-fold",
  repeated_kfold: "Repeated k-fold",
  bootstrap: "Bootstrap optimism correction",
  internal_external: "Internal–external",
};
const LABELS = {
  folds: "Folds",
  repeats: "Repeats (repeated k-fold)",
  n_boot: "Resamples (bootstrap)",
  cluster: "Group column (internal–external)",
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
  // The scheme as the split holds it, set with the draw or by this card.
  const scheme = state.split ?? DEFAULTS.set_validation;
  const [folds, setFolds] = useState(String(scheme.folds));
  const [repeats, setRepeats] = useState(String(scheme.repeats));
  const [nBoot, setNBoot] = useState(String(scheme.n_boot));
  const [cluster, setCluster] = useState(scheme.cluster ?? "");
  const [nested, setNested] = useState(scheme.nested_cv);
  const bounds = BOUNDS.set_validation;
  // The model holds all three numbers whatever the scheme, so each has to be a number it accepts.
  const typed = {
    folds: parseTyped(folds, bounds.folds, true, true),
    repeats: parseTyped(repeats, bounds.repeats, true, true),
    n_boot: parseTyped(nBoot, bounds.n_boot, true, true),
  };

  const offerOf = (kind: Validation["validation"]): Offer => {
    const { folds: f, repeats: r, n_boot: b } = typed;
    const problems = [
      ...trouble(LABELS.folds, f, bounds.folds, true),
      ...trouble(LABELS.repeats, r, bounds.repeats, true),
      ...trouble(LABELS.n_boot, b, bounds.n_boot, true),
      ...(kind === "internal_external" && cluster === ""
        ? [`${LABELS.cluster}: choose a column`]
        : []),
    ];
    if (f === "invalid" || r === "invalid" || b === "invalid") return { problems };
    const decision = validation(kind, {
      folds: f,
      repeats: r,
      n_boot: b,
      cluster: cluster === "" ? null : cluster,
      nested_cv: nested,
    });
    return decision ? { decision } : { problems };
  };

  const groups = card.columns.filter((col) => col !== state.target);
  return (
    <>
      <div className={c.fields}>
        <NumberField
          name="folds"
          label={LABELS.folds}
          text={folds}
          onChange={setFolds}
          bounds={bounds.folds}
          integer
          required
        />
        <NumberField
          name="repeats"
          label={LABELS.repeats}
          text={repeats}
          onChange={setRepeats}
          bounds={bounds.repeats}
          integer
          required
        />
        <NumberField
          name="n_boot"
          label={LABELS.n_boot}
          text={nBoot}
          onChange={setNBoot}
          bounds={bounds.n_boot}
          integer
          required
        />
        <SelectField
          name="cluster"
          label={LABELS.cluster}
          value={cluster}
          invalid={card.held === "validation:internal_external" && cluster === ""}
          options={[
            { value: "", label: "Choose a column" },
            ...groups.map((g) => ({ value: g, label: g })),
          ]}
          onChange={setCluster}
        />
        <CheckField
          name="nested_cv"
          label="Nested cross-validation interval"
          checked={nested}
          onChange={setNested}
        />
      </div>
      <Choice
        card={card}
        field="validation"
        legend="Scheme"
        values={VALIDATIONS}
        names={NAMES}
        offerOf={offerOf}
        // `set_validation`'s own record only: the scheme rides in the split's record until changed.
        recorded={recorded ? (state.split?.validation ?? null) : null}
        testId="options-validation"
      />
    </>
  );
}

export const ValidationCard: LineCard = ({ pid, line }) => {
  const card = useCard(pid, "set_validation");
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
