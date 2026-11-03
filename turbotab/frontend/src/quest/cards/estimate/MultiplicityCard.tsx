/**
 * `set_multiplicity`: how the tests of a family of study factors are accounted for
 * (core/decisions.py `SetMultiplicity`), a Confirm line in Models under Estimate once the estimand
 * declares a family (core/quest.py). The three methods the model knows: Benjamini–Hochberg
 * q-values, unadjusted p-values with the number of tests stated, or none. The engine blocks no
 * control at all, and unadjusted p-values beyond a few tests or for an omics family, until the
 * answer is recorded as a limitation; it says so in its own words, with the attested answer as its
 * way forward, so the card never sends the attestation itself. What another method would change is
 * the line's own sentence, and what each does to this family is the engine's preview of it.
 *
 * The engine serves no reason for this default, and the state holds no method until one is
 * answered, so no option is marked as set for the person.
 */
import { Pending } from "../../../components/record/blocks";
import type { LineCard } from "../../contracts";
import { Choice, Frame, useCard } from "../predict/card";
import { MULTIPLICITY_METHODS, multiplicity, type Multiplicity } from "./payloads";

/** The estimand card's own labels for the same methods (core/estimand.py `MULTIPLICITY_WORDS`). */
const NAMES: Record<Multiplicity["method"], string> = {
  bh: "Benjamini–Hochberg q-values across the family",
  stated_count: "Unadjusted p-values, with the number of tests stated",
  none: "No multiplicity control",
};

export const MultiplicityCard: LineCard = ({ pid, line }) => {
  const card = useCard(pid, "set_multiplicity");
  const recorded = line.status === "answered" ? (card.state?.multiplicity?.method ?? null) : null;
  return (
    <Frame line={line}>
      {card.state ? (
        <Choice
          key={line.decision_id ?? line.status}
          card={card}
          field="method"
          legend="Method"
          values={MULTIPLICITY_METHODS}
          names={NAMES}
          offerOf={(method) => ({ decision: multiplicity(method) })}
          recorded={recorded}
          testId="options-multiplicity"
        />
      ) : (
        <Pending>Loading the project…</Pending>
      )}
    </Frame>
  );
};
