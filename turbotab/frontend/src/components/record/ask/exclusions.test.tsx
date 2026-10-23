/**
 * The exclusions card's two ranges of one's own (the final fix round's V1): a range of plausible
 * values and a range of who the study is about, such as an age range, each recording why its rows
 * leave, which the participant flow and its exported figure print ("Excluded: <reason>"). An age
 * range kept from 20 is not a screen of implausible values. The expected rules, and both reasons,
 * are written out here.
 */
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import type { EligibilityRule } from "../../../api/m1-types";
import type { Decision } from "../../../api/schema";
import { ExclusionsAsk } from "./ChoiceQuestions";

const PLAUSIBLE = "implausible values (your own range)";
const POPULATION = "outside the study population (your own range)";

function renderCard(current: EligibilityRule[] | null = null) {
  const record = vi.fn<(d: Decision, at?: string) => void>();
  render(
    <ExclusionsAsk
      entry={undefined}
      pending={false}
      record={record}
      answerAt={null}
      shell={{
        qkey: "exclusions",
        now: true,
        arriving: false,
        onArrived: () => {},
        onOpenDrawer: () => {},
        reopened: false,
      }}
      proposals={undefined}
      current={current}
      numericColumns={["BMXBMI", "RIDAGEYR"]}
    />,
  );
  return record;
}

/** Fill the range form under `option` and submit it: the rows kept from `low` on `column`. */
function keepFrom(option: string, column: string, low: string) {
  const form = within(screen.getByTestId(`option-${option}`));
  fireEvent.change(form.getByRole("combobox"), { target: { value: column } });
  fireEvent.change(form.getByLabelText("Lowest value kept"), { target: { value: low } });
  fireEvent.click(form.getByRole("button", { name: "Exclude the rest" }));
}

const range = (column: string, low: number, reason: string): EligibilityRule => ({
  kind: "range",
  column,
  low,
  high: null,
  by: null,
  reason,
  missing: "exclude",
});

afterEach(cleanup);

describe("a range of one's own", () => {
  it("records who the study is about with the study population's reason", () => {
    const record = renderCard();
    keepFrom("eligibility", "RIDAGEYR", "20");
    expect(record).toHaveBeenCalledTimes(1);
    expect(record.mock.calls[0]).toEqual([
      { kind: "set_exclusions", rules: [range("RIDAGEYR", 20, POPULATION)] },
      "eligibility",
    ]);
  });

  it("keeps the plausible range's own reason", () => {
    const record = renderCard();
    keepFrom("custom", "BMXBMI", "12");
    expect(record.mock.calls[0]).toEqual([
      { kind: "set_exclusions", rules: [range("BMXBMI", 12, PLAUSIBLE)] },
      "custom",
    ]);
  });

  it.each([
    ["eligibility", POPULATION],
    ["custom", PLAUSIBLE],
  ])("reads an answer on record back as the %s option, by its reason", (key, reason) => {
    renderCard([range("RIDAGEYR", 20, reason)]);
    expect(screen.getByTestId(`option-${key}`).getAttribute("aria-selected")).toBe("true");
    const other = key === "custom" ? "eligibility" : "custom";
    expect(screen.getByTestId(`option-${other}`).getAttribute("aria-selected")).toBe("false");
  });
});
