/**
 * The orientation card offers the turn check's ways forward (front-end audit G15).
 *
 * When the table cannot be turned round because its first column holds a different whole number on
 * every row (core/stages/working.py `turn_plan`, `label_unclear`), the turn check says so and serves
 * its exits: each a `set_feature_table` answer that says what that column is. The card said the
 * refusal in place of the feature-major option and offered no way to answer it, so a feature-major
 * table with such a first column could not be turned. The exits are now buttons under the refusal,
 * each recording exactly the decision the server served. The turns are the real server's, captured
 * in the m3 fixtures (docs/turbotab-next/m3/capture_fixtures.py); the refusals without a way forward
 * are built in the engine's shape (`refuse(*_label_problem(...))` serves no exits).
 */
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import type { OrientedArtifact, TurnCheck } from "../../../api/m2-types";
import type { Decision } from "../../../api/schema";
import metabolomics from "../../../mocks/fixtures/m3-metabolomics.json";
import prediction from "../../../mocks/fixtures/m3-nhanes-prediction.json";
import { OrientationAsk } from "./SequenceQuestions";

const orientedOf = (capture: unknown) =>
  (capture as { artifacts: { oriented: { base: OrientedArtifact } } }).artifacts.oriented.base;

// NHANES Predict's first column, SEQN, is a different whole number on every row: label_unclear.
const UNCLEAR = orientedOf(prediction);
// The metabolomics table turns: no refusal, no exits.
const TURNS = orientedOf(metabolomics);

const withTurn = (turn: Partial<TurnCheck>): OrientedArtifact => ({
  ...UNCLEAR,
  turn: { ...UNCLEAR.turn, ...turn },
});

function renderOrientation(oriented: OrientedArtifact | null, pending = false) {
  const record = vi.fn<(d: Decision, at?: string) => void>();
  render(
    <OrientationAsk
      entry={undefined}
      pending={pending}
      record={record}
      answerAt={null}
      shell={{
        qkey: "orientation",
        now: true,
        arriving: false,
        onArrived: () => {},
        onOpenDrawer: () => {},
        reopened: false,
      }}
      oriented={oriented}
      current={null}
    />,
  );
  return record;
}

const exitButtons = (option: string) =>
  within(screen.getByTestId(`option-${option}`)).queryAllByTestId("turn-exit");

afterEach(cleanup);

describe("the turn check's ways forward", () => {
  it("are the captured fixture's own: a refusal with two set_feature_table exits", () => {
    // The premise of every test below, read from the served payload.
    expect(UNCLEAR.turn.code).toBe("label_unclear");
    expect(UNCLEAR.turn.refusal).toBeTruthy();
    expect(UNCLEAR.turn.exits.map((x) => x.decision?.kind)).toEqual([
      "set_feature_table",
      "set_feature_table",
    ]);
    expect(TURNS.turn.refusal).toBeNull();
    expect(TURNS.turn.exits).toEqual([]);
  });

  it("are buttons under the refusal, one per exit, in the server's words and order", () => {
    renderOrientation(UNCLEAR);
    const option = screen.getByTestId("option-feature_major");
    // The refusal is still said in place of the option's consequence.
    expect(option.textContent).toContain(UNCLEAR.turn.refusal!.replace(/`/g, ""));
    expect(exitButtons("feature_major").map((b) => b.textContent)).toEqual(
      UNCLEAR.turn.exits.map((x) => x.label.replace(/`/g, "")),
    );
  });

  it("each records exactly the decision the server served, answered at the feature-major option", () => {
    UNCLEAR.turn.exits.forEach((exit, i) => {
      const record = renderOrientation(UNCLEAR);
      fireEvent.click(exitButtons("feature_major")[i]!);
      // One press, one decision: the option's own set_orientation is not recorded with it.
      expect(record).toHaveBeenCalledTimes(1);
      expect(record.mock.calls[0]).toEqual([exit.decision, "feature_major"]);
      cleanup();
    });
  });

  it("wait while a decision is being recorded", () => {
    renderOrientation(UNCLEAR, true);
    const buttons = exitButtons("feature_major");
    expect(buttons).toHaveLength(UNCLEAR.turn.exits.length);
    for (const b of buttons) expect((b as HTMLButtonElement).disabled).toBe(true);
  });

  it("are not offered beside the sample-major option", () => {
    renderOrientation(UNCLEAR);
    expect(exitButtons("sample_major")).toHaveLength(0);
  });

  it("are absent when the table turns", () => {
    renderOrientation(TURNS);
    expect(screen.queryAllByTestId("turn-exit")).toHaveLength(0);
    // The turned shape is the option's data, as before.
    expect(screen.getByTestId("option-feature_major").textContent).toContain(
      `${TURNS.turn.n_samples.toLocaleString("en-US")} rows`,
    );
  });

  it("are absent when the refusal serves none, and the refusal is still said", () => {
    const refusal = "Two rows are both named `x`. Give the rows distinct names first.";
    renderOrientation(withTurn({ code: "duplicate_features", refusal, exits: [] }));
    expect(screen.queryAllByTestId("turn-exit")).toHaveLength(0);
    expect(screen.getByTestId("option-feature_major").textContent).toContain(
      refusal.replace(/`/g, ""),
    );
  });

  it("leave out an exit that records nothing", () => {
    const [named] = UNCLEAR.turn.exits;
    renderOrientation(withTurn({ exits: [named!, { label: "Rename the column", decision: null }] }));
    expect(exitButtons("feature_major").map((b) => b.textContent)).toEqual([
      named!.label.replace(/`/g, ""),
    ]);
  });

  it("are absent before the table's shape is read", () => {
    renderOrientation(null);
    expect(screen.queryAllByTestId("turn-exit")).toHaveLength(0);
  });
});
