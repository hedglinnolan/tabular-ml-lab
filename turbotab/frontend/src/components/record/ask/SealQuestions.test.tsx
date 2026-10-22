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
import type { SplitArtifact } from "../../../api/m1-types";
import { SealAsk, basisChronology, cutDecision, cutShown } from "./SealQuestions";

type Split = NonNullable<ProjectState["split"]>;

function renderSeal(
  current: ProjectState["split"],
  plan: SealPlan | null = null,
  timeColumns: { name: string; numeric: boolean }[] = [],
) {
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
      timeColumns={timeColumns}
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
  cut_point: null,
  fold_order: null,
};
const SCHEME_OF_CHANGED = {
  folds: 10,
  validation: "repeated_kfold",
  repeats: 20,
  n_boot: 700,
  cluster: null,
  nested_cv: true,
};
// A share holds no cut-point, and its folds' order is the draw's (decisions.py SplitSpec).
const BY_SHARE = { cut_point: null, fold_order: null };

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
      ...BY_SHARE,
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
      ...BY_SHARE,
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
      ...BY_SHARE,
    });
  });
});

// SWARM_PLAN §8: a temporal seal by cut-point, its folds' order its own answer.
describe("a cut-point", () => {
  const TIME = [
    { name: "cycle", numeric: true },
    { name: "visit", numeric: false },
  ];

  it("records nothing until its value and its folds' order are given", () => {
    const record = renderSeal(CHANGED, null, TIME);
    fireEvent.change(screen.getByLabelText("The first value held out"), {
      target: { value: "2017" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Seal from here" }));
    expect(record).not.toHaveBeenCalled(); // no folds' order chosen yet
    fireEvent.change(screen.getByLabelText("How the training folds are ordered"), {
      target: { value: "random" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Seal from here" }));
    expect(record.mock.calls[0]![0]).toEqual({
      kind: "set_split",
      holdout: 0,
      seed: 0,
      ...SCHEME_OF_CHANGED,
      cut_point: { column: "cycle", at: 2017 },
      fold_order: "random",
    });
  });

  it("sends a numeral on a numeric column as the number, and a date as written", () => {
    expect(cutDecision("cycle", " 2017 ", "time", true, null)?.cut_point).toEqual({
      column: "cycle",
      at: 2017,
    });
    expect(cutDecision("visit", "2020-01-01", "time", false, null)?.cut_point).toEqual({
      column: "visit",
      at: "2020-01-01",
    });
    expect(cutDecision("cycle", "", "time", true, null)).toBeNull();
    expect(cutDecision("cycle", "2017", null, true, null)).toBeNull();
  });

  it("shows a recorded cut-point as the recorded answer, not as cross-validation alone", () => {
    const cut = { column: "cycle", at: 2017 };
    renderSeal({ ...CHANGED, holdout: 0, cut_point: cut, fold_order: "time" }, null, TIME);
    expect(screen.getByTestId("option-cut")).toHaveAttribute("data-recorded", "true");
    expect(screen.getByTestId("option-0")).not.toHaveAttribute("data-recorded", "true");
  });
});

// The package 4 verifier's P7: under a recorded cut-point the seal plan's chronology is a usual
// share's draw by time, which the cut-point replaces, so the basis line states the split's own
// sentence once it is drawn for that cut-point, and the basis alone until then.
describe("the seal's basis line under a cut-point", () => {
  const LATEST = "The latest `6` units by their last `cycle` are held out, from `2021` on.";
  const CUT_SENTENCE =
    "Held out by a cut-point at `2017` on `cycle`: `20` rows held out and `40` train.";
  const plan = {
    basis: {
      state: "one_row_per_unit",
      column: null,
      label: "one row per unit",
      sentence: "Each row is its own unit.",
      exploratory: false,
      source: "grain",
      n_units: null,
    },
    chronology: { drawn: true, time_column: "cycle", cut_point: null, sentence: LATEST },
    refusal: null,
    precision_note: null,
    cv_first: false,
    reason: "",
    options: [],
  } as unknown as SealPlan;
  const drawnAt = (at: string) =>
    ({
      chronology: { drawn: true, time_column: "cycle", cut_point: at, sentence: CUT_SENTENCE },
    }) as unknown as SplitArtifact;
  const cut: Split = {
    ...CHANGED,
    holdout: 0,
    cut_point: { column: "cycle", at: 2017 },
    fold_order: "random",
  };

  it("states the split's own sentence once drawn for this cut-point", () => {
    expect(basisChronology(plan, cut, drawnAt("2017"))?.sentence).toBe(CUT_SENTENCE);
    render(
      <SealAsk
        entry={undefined}
        pending={false}
        record={vi.fn()}
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
        current={cut}
        split={drawnAt("2017")}
      />,
    );
    const line = screen.getByTestId("seal-basis").textContent ?? "";
    expect(line).toContain("Held out by a cut-point at");
    expect(line).not.toContain("The latest");
  });

  it("states the basis alone before the split is drawn, or drawn for another cut-point", () => {
    expect(basisChronology(plan, cut, undefined)).toBeNull();
    expect(basisChronology(plan, cut, drawnAt("2015"))).toBeNull();
  });

  it("keeps the plan's chronology where no cut-point is recorded", () => {
    expect(basisChronology(plan, CHANGED, drawnAt("2017"))?.sentence).toBe(LATEST);
    expect(basisChronology(plan, null, undefined)?.sentence).toBe(LATEST);
  });

  it("shows a cut-point's value as the server does", () => {
    expect([cutShown(2017), cutShown(2016.5), cutShown(" 2020-01-01 ")]).toEqual([
      "2017",
      "2016.5",
      "2020-01-01",
    ]);
  });
});
