/**
 * The ledger's ask card (BLUEPRINT §14.2) on the real server's cards (src/mocks/fixtures): a
 * confirmation settles exactly the readings its line lists, each with the value the line shows;
 * a family is one line and one block; a changed value is the value confirmed; "every line"
 * lists every line shown and nothing else; a line the consumer answers by its own decision (total
 * energy's unit and days) is offered only as the server's exit. And its words keep the budgets.
 */
import { fireEvent, render, screen, within } from "@testing-library/react";
import type { AskCard as Card } from "../../../api/m3-types";
import type { Decision } from "../../../api/schema";
import { viewsOf, type M3Fixture } from "../../../mocks/m3";
import { AskCard, confirmAll, confirmLine } from "./AskCard";

const FIXTURES = import.meta.glob<M3Fixture>("../../../mocks/fixtures/m3-*.json", {
  eager: true,
  import: "default",
});

/** The ask card on `key` at the first snapshot of `journey` where it is open and asks. */
function cardOf(journey: string, key: string, where?: (c: Card) => boolean): Card {
  const f = Object.entries(FIXTURES).find(([p]) => p.endsWith(`m3-${journey}.json`))![1];
  for (const v of viewsOf(f)) {
    const st = v.interview.find((s) => s.key === key && s.status === "open" && s.ask);
    if (st?.ask && (!where || where(st.ask))) return st.ask;
  }
  throw new Error(`no ask card on ${key} in ${journey}`);
}

function renderCard(card: Card, mastered = true) {
  const record = vi.fn<(d: Decision, at: string) => void>();
  render(<AskCard card={card} pending={false} record={record} answerAt={null} mastered={mastered} />);
  return record;
}

type Confirm = Extract<Decision, { kind: "confirm_readings" }>;
const items = (d: Decision) => (d as Confirm).items.map((i) => `${i.reading}:${i.column}=${i.value}`).sort();

describe("the ask card", () => {
  const survey = cardOf("survey", "models", (c) => c.groups.some((g) => g.columns.length > 1));

  it("shows each line's guess with its evidence, a family as one line", () => {
    renderCard(survey);
    const lines = screen.getAllByTestId("ask-line");
    expect(lines).toHaveLength(survey.groups.length);
    const family = survey.groups.find((g) => g.columns.length > 1)!;
    expect(screen.getByText(new RegExp(`${family.columns.length} columns like`))).toBeInTheDocument();
    for (const g of survey.groups) if (g.evidence) expect(document.body.textContent).toContain(g.evidence.replace(/`/g, ""));
  });

  it("confirms a family as one block that lists exactly its readings, each with its guess", () => {
    const record = renderCard(survey);
    const i = survey.groups.findIndex((g) => g.columns.length > 1);
    const family = survey.groups[i]!;
    fireEvent.click(within(screen.getAllByTestId("ask-line")[i]!).getByTestId("ask-confirm"));
    const [d] = record.mock.calls[0]!;
    expect(d.kind).toBe("confirm_readings");
    expect(items(d)).toEqual(family.columns.map((c) => `code_or_count:${c}=${family.guess}`).sort());
  });

  it("confirms the value the line shows once it was changed", () => {
    const record = renderCard(survey);
    const i = survey.groups.findIndex((g) => g.columns.length === 1);
    const line = within(screen.getAllByTestId("ask-line")[i]!);
    const other = survey.groups[i]!.guess === "code" ? "amount" : "code";
    fireEvent.change(line.getByTestId("ask-change"), { target: { value: other } });
    expect(line.getByText(/changed/)).toBeInTheDocument();
    fireEvent.click(line.getByTestId("ask-confirm"));
    expect(items(record.mock.calls[0]![0])).toEqual([`code_or_count:${survey.groups[i]!.columns[0]}=${other}`]);
  });

  it("offers the block of every line only once a guess was confirmed line by line (§11.4)", () => {
    renderCard(survey, false);
    expect(screen.queryByTestId("ask-confirm-all")).toBeNull();
    expect(screen.getAllByTestId("ask-confirm").length).toBe(survey.groups.length);
  });

  it("confirms every line in one block that lists every reading shown and nothing else", () => {
    const record = renderCard(survey);
    fireEvent.click(screen.getByTestId("ask-confirm-all"));
    const listed = survey.groups.flatMap((g) => g.columns.map((c) => `${g.kind}:${c}=${g.guess}`)).sort();
    expect(items(record.mock.calls[0]![0])).toEqual(listed);
    expect(confirmAll(survey.groups, {})).toEqual(record.mock.calls[0]![0]);
  });

  it("answers total energy's unit and days by the screens' own decision, never as a reading", () => {
    const card = cardOf("dietary-recalls", "exclusions", (c) => c.groups.some((g) => g.kind === "unit"));
    const unit = card.groups.find((g) => g.kind === "unit")!;
    expect(confirmLine(unit, unit.guess!)).toBeNull();
    const record = renderCard(card);
    const exit = within(screen.getByTestId("ask-exits")).getByRole("button");
    fireEvent.click(exit);
    expect(record.mock.calls[0]![0].kind).toBe("set_column_unit");
    // "every line" lists the role line only: the unit line is not a reading it may settle.
    const all = confirmAll(card.groups, {});
    if (all) expect(items(all).every((x) => !x.startsWith("unit:"))).toBe(true);
  });

  it("lists what the values settled, each with the answers that change it", () => {
    const card = cardOf("nhanes-prediction", "models", (c) => c.read_from_data.length > 0);
    renderCard(card);
    const read = screen.getByTestId("read-from-data");
    fireEvent.click(within(read).getByRole("button", { name: /Read from your data/ }));
    expect(within(read).getAllByRole("listitem")).toHaveLength(card.read_from_data.length);
  });

  it("keeps its own words within the question budget", () => {
    // BLUEPRINT §11 rule 4 (turbotab/core/teaching BUDGETS): a question ≤ 14 words.
    renderCard(survey);
    const title = screen.getByRole("heading", { level: 3 }).textContent!;
    expect(title.split(/\s+/).length).toBeLessThanOrEqual(14);
    for (const b of screen.getAllByRole("button"))
      expect(b.textContent!.split(/\s+/).length, b.textContent!).toBeLessThanOrEqual(8);
  });
});
