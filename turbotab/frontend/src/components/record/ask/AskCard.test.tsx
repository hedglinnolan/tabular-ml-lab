/**
 * The ledger's ask card (BLUEPRINT §14.2) on the real server's cards (src/mocks/fixtures): a
 * confirmation settles exactly the readings its line lists, each with the value the line shows;
 * a family is one line and one block, but a column's role, which is a line and a press of its own;
 * a changed value is the value confirmed; "every line" lists every line shown but the roles, and
 * nothing else; a line the consumer answers by its own decision (total
 * energy's unit and days) is offered only as the server's exit. And its words keep the budgets.
 */
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
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
    // "every line" lists no unit line, which is not a reading it may settle (nor the role line,
    // which is confirmed on its own: §14 rule 2).
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

  /**
   * BLUEPRINT §14 rule 2 (ruling R-g of 2026-10-17, which replaces R-b's one per column): a role
   * proposed below high confidence never rides along in a block with other readings. The card's
   * role lines are such roles (the ask card asks only readings not settled): the engine serves each
   * as a line of its own, or, where its roles stage proposed several columns with one role, one
   * confidence and one reason (one judgment of the recognizer: ask.py `with_proposals`), the whole
   * group as one line. A line's press records exactly its columns' roles, as proposed, and no press
   * records a role with anything else. Built as the engine serves the Models ask under Predict:
   * `bp_sys` alone and six imputed flags proposed alike as one group, here beside two code-or-amount
   * lines.
   */
  describe("below-high roles, one line per judgment (ruling R-g)", () => {
    const FLAGS = ["imputed_bp", "imputed_hdl", "imputed_height", "imputed_tg", "imputed_waist", "imputed_weight"];
    const role = (columns: string[], guess: string) => ({ kind: "role", columns, guess, guess_words: guess === "flag" ? "a flag on another column" : "a covariate", evidence: "proposed below high confidence" });
    const code = (column: string) => ({ kind: "code_or_count", columns: [column], guess: "amount", guess_words: "an amount", evidence: "whole numbers from 20 to 230" });
    const CARD = {
      question: "models",
      consumer: "The fit",
      text: "Tell me about these columns.",
      groups: [role(["bp_sys"], "covariate"), role(FLAGS, "flag"), code("hdl"), code("triglycerides")],
      exits: [],
      read_from_data: [],
    } as unknown as Card;
    /** The readings a decision records, as `reading:column`. */
    const read = (d: Decision) => (d.kind === "confirm_readings" ? d.items.map((i) => `${i.reading}:${i.column}`) : []);

    it("gives each judgment one line, whose one press records its columns' roles as proposed", () => {
      const record = renderCard(CARD);
      const lines = screen.getAllByTestId("ask-line").filter((l) => l.getAttribute("data-kind") === "role");
      expect(lines).toHaveLength(2);
      const served = [["bp_sys"], FLAGS];
      for (const [n, line] of lines.entries()) {
        const button = within(line).getByTestId("ask-confirm");
        fireEvent.click(button);
        const value = n === 0 ? "covariate" : "flag";
        expect(record.mock.calls.at(-1)![0]).toEqual({
          kind: "confirm_readings",
          items: served[n]!.map((column) => ({ reading: "role", column, value })),
        });
        // the group's press says what it confirms, and its role is not changed here
        if (n === 1) {
          expect(button).toHaveTextContent(`Confirm these ${FLAGS.length} as a flag on another column`);
          expect(within(line).queryByTestId("ask-change")).toBeNull();
        }
      }
    });

    it("offers no press that records a role with any other reading, or with another line's", () => {
      const record = renderCard(CARD);
      for (const b of screen.getAllByRole("button")) fireEvent.click(b);
      expect(record.mock.calls.length).toBeGreaterThan(0);
      const lines = [["role:bp_sys"], FLAGS.map((c) => `role:${c}`)];
      for (const [d] of record.mock.calls) {
        const items = read(d);
        if (items.some((x) => x.startsWith("role:"))) expect(lines, JSON.stringify(d)).toContainEqual(items);
      }
    });

    it("keeps the block of every other line, listing no role", () => {
      const record = renderCard(CARD);
      fireEvent.click(screen.getByTestId("ask-confirm-all"));
      expect(read(record.mock.calls[0]![0]).sort()).toEqual(["code_or_count:hdl", "code_or_count:triglycerides"]);
      expect(confirmAll(CARD.groups, {})).toEqual(record.mock.calls[0]![0]);
      expect(screen.getByTestId("ask-confirm-all")).toHaveTextContent("(2)");
    });

    it("offers no block where one line besides the roles is left", () => {
      renderCard({ ...CARD, groups: [role(["bp_sys"], "covariate"), role(FLAGS, "flag"), code("hdl")] });
      expect(screen.queryByTestId("ask-confirm-all")).toBeNull();
    });

    /** The W1-V verifier's P3: the heading counts the columns its lines list, as the engine's ask
     *  text does (core/readings.py `ask_text`), so one line of a group of six asks about "these
     *  columns", and a line of one column about "this column". */
    it("heads one line of several columns 'these columns', and a lone column 'this column'", () => {
      const heading = () => [
        screen.getByRole("heading", { level: 3 }).textContent,
        screen.getByTestId("ask-card").querySelector('[class*="askConsumer"]')!.textContent,
      ];
      const one = renderCard({ ...CARD, groups: [role(FLAGS, "flag")] }, false);
      expect(screen.getAllByTestId("ask-line")).toHaveLength(1);
      expect(heading()).toEqual(["Tell me about these columns", "The fit reads them"]);
      cleanup();
      renderCard({ ...CARD, groups: [role(["bp_sys"], "covariate")] }, false);
      expect(heading()).toEqual(["Tell me about this column", "The fit reads it"]);
      cleanup();
      renderCard({ ...CARD, groups: [role(["bp_sys"], "covariate"), code("hdl")] }, false);
      expect(heading()).toEqual(["Tell me about these columns", "The fit reads them"]);
      expect(one).not.toHaveBeenCalled();
    });
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
