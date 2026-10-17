/**
 * The seal's card sends the draw and keeps the validation scheme (W1_PART2_PLAN ruling 10).
 *
 * `set_split` replaces the whole split slot, so a card that sends a hard-coded 5-fold k-fold
 * scheme with the draw puts back the scheme the researcher had changed with `set_validation`
 * (the engine's fold, checked in turbotab/core/tests/test_w1_declaration_cards.py). The card now
 * sends the recorded scheme with the new holdout, and the model's own defaults only where no
 * split is recorded yet. The expected schemes are written out here, not read from the card.
 */
import { fireEvent, render, screen } from "@testing-library/react";
import type { SealPlan } from "../../../api/m2-types";
import type { Decision, ProjectState } from "../../../api/schema";
import { SealAsk } from "./SealQuestions";

type Split = NonNullable<ProjectState["split"]>;

function renderSeal(current: ProjectState["split"], plan: SealPlan | null = null) {
  const record = vi.fn<(d: Decision, at?: string) => void>();
  render(
    <SealAsk
      entry={undefined}
      pending={false}
      record={record}
      answerAt={null}
      shell={{
        qkey: "split",
        now: true,
        arriving: false,
        onArrived: () => {},
        onOpenDrawer: () => {},
        reopened: false,
      }}
      plan={plan}
      roles={undefined}
      current={current}
    />,
  );
  return record;
}

const pressed = (record: ReturnType<typeof renderSeal>, option: string): Decision => {
  fireEvent.click(screen.getByTestId(`option-${option}`));
  expect(record).toHaveBeenCalledTimes(1);
  return record.mock.calls[0]![0];
};

// A scheme the researcher set with `set_validation`: nothing in it is a default.
const CHANGED: Split = {
  holdout: 0.2,
  seed: 0,
  folds: 10,
  validation: "repeated_kfold",
  repeats: 20,
  n_boot: 700,
  cluster: null,
  nested_cv: true,
};
const SCHEME_OF_CHANGED = {
  folds: 10,
  validation: "repeated_kfold",
  repeats: 20,
  n_boot: 700,
  cluster: null,
  nested_cv: true,
};

describe("re-answering the seal", () => {
  it.each([
    ["0.1", 0.1],
    ["0.3", 0.3],
    ["0", 0],
  ])("keeps a validation scheme set before when the holdout chosen is %s", (option, holdout) => {
    const record = renderSeal(CHANGED);
    expect(pressed(record, option!)).toEqual({
      kind: "set_split",
      holdout,
      seed: 0,
      ...SCHEME_OF_CHANGED,
    });
  });

  it("keeps an internal-external scheme and the column whose groups are the folds", () => {
    const record = renderSeal({
      ...CHANGED,
      validation: "internal_external",
      cluster: "cycle_begin_year",
      nested_cv: false,
    });
    expect(pressed(record, "0.2")).toMatchObject({
      kind: "set_split",
      holdout: 0.2,
      validation: "internal_external",
      cluster: "cycle_begin_year",
      folds: 10,
      repeats: 20,
      n_boot: 700,
    });
  });

  it("sends the model's defaults as the scheme where no split is recorded yet", () => {
    const record = renderSeal(null);
    expect(pressed(record, "0.1")).toEqual({
      kind: "set_split",
      holdout: 0.1,
      seed: 0,
      folds: 5,
      validation: "kfold",
      repeats: 10,
      n_boot: 500,
      cluster: null,
      nested_cv: false,
    });
  });

  it("keeps the scheme through the options the engine serves, too", () => {
    const plan = {
      basis: null,
      chronology: null,
      refusal: null,
      precision_note: null,
      cv_first: false,
      reason: "",
      options: [
        { holdout: 0.2, label: "Hold out 20%", measures: "" },
        { holdout: 0, label: "Cross-validation only", measures: "" },
      ],
    } as unknown as SealPlan;
    const record = renderSeal(CHANGED, plan);
    expect(pressed(record, "0")).toEqual({
      kind: "set_split",
      holdout: 0,
      seed: 0,
      ...SCHEME_OF_CHANGED,
    });
  });
});
