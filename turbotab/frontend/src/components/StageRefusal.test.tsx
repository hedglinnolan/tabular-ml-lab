/**
 * A refusal a stage raised at fit time (StageStatus.refusal), shown with its ways forward (the
 * audit's G04), on the statuses the engine serves (fixtures/stages.json, captured from the engine
 * by capture_stages.py): every expected word is the served payload's own.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { RefusalError, api } from "../api/client";
import type { ProjectView, Refusal } from "../api/schema";
import { SERVED } from "./fixtures/stages";
import { OpenCard } from "./record/target";
import { StageRefusal, besideCard, refusalOf, waysForward } from "./StageRefusal";

function wrap(node: ReactNode) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={qc}>{node}</QueryClientProvider>);
}

/** Prose sets a backticked span in mono: its text is the label without the backticks. */
const shown = (label: string) => label.replaceAll("`", "");

afterEach(() => vi.restoreAllMocks());

describe("refusalOf", () => {
  it("is the refusal a refused stage serves: its own, or that of the stage it waits on", () => {
    expect(refusalOf(SERVED.R)).toEqual(SERVED.R.refusal);
    // AR waits on RH, which refused (the toy graph's construction).
    expect(refusalOf(SERVED.AR)?.stage).toBe("RH");
    expect(refusalOf(SERVED.U)?.code).toBe(SERVED.U.refusal!.code);
  });

  it("is null for a crash, a stage waiting on one, and a stage not in error", () => {
    expect(refusalOf(SERVED.C)).toBeNull();
    expect(refusalOf(SERVED.AC)).toBeNull();
    expect(refusalOf({ ...SERVED.R, status: "stale" })).toBeNull();
    expect(refusalOf(undefined)).toBeNull();
  });
});

describe("waysForward", () => {
  it("keeps each way forward as the engine serves it: a decision, or none", () => {
    expect(waysForward(SERVED.R.refusal!)).toEqual(SERVED.R.refusal!.exits);
    expect(waysForward(SERVED.U.refusal!)).toEqual(SERVED.U.refusal!.exits);
  });

  it("offers no way forward that confirms a role with other readings in one press", () => {
    // BLUEPRINT §14 rule 2 (besideAsk.ts confirmsRoleInBlock): a block that confirms a role with
    // a code-or-amount reading is no press; a block of roles that are one proposed role is one.
    const block = {
      label: "Confirm both as shown",
      decision: {
        kind: "confirm_readings",
        items: [
          { reading: "role", column: "a", value: "exposure" },
          { reading: "code_or_count", column: "b", value: "amount" },
        ],
      },
    };
    const group = {
      label: "Confirm both as covariates",
      decision: {
        kind: "confirm_readings",
        items: [
          { reading: "role", column: "a", value: "covariate" },
          { reading: "role", column: "b", value: "covariate" },
        ],
      },
    };
    const refusal = { ...SERVED.R.refusal!, exits: [block, group] };
    expect(waysForward(refusal).map((x) => x.label)).toEqual([group.label]);
  });
});

describe("besideCard", () => {
  it("keeps the refusal's words up to the ledger's question, which the card asks, and its ways that name no decision", () => {
    // By construction, as core/readings.py `ask_text` follows a consumer's words in a refusal that
    // asks: the card lists each reading with its evidence (record/besideAsk.ts), so the words stop
    // where its question begins.
    const unsettled = SERVED.U.refusal!;
    const asked = {
      ...unsettled,
      message: `${unsettled.message} Tell me about this column: \`age\`: an amount? (12 distinct values).`,
      exits: [...unsettled.exits, { label: "Wait for the table", decision: null }],
    };
    expect(besideCard(asked)).toEqual({ message: unsettled.message, told: [{ label: "Wait for the table", decision: null }] });
  });

  it("keeps a refusal that asks no reading whole, with its way forward that names no decision", () => {
    const refusal = SERVED.R.refusal!;
    expect(besideCard(refusal)).toEqual({
      message: refusal.message,
      told: refusal.exits.filter((x) => x.decision === null),
    });
    // A refusal with no confirmation keeps words that read like the ledger's question.
    const like = { ...refusal, message: `${refusal.message} Tell me about this column: \`x\`.` };
    expect(besideCard(like).message).toBe(like.message);
  });
});

describe("StageRefusal", () => {
  const refusal = SERVED.R.refusal!;
  const [take, told] = waysForward(refusal);

  it("says the refusal's words and offers its decision as a press that records it", async () => {
    expect(take!.decision).not.toBeNull();
    expect(told!.decision).toBeNull();
    const decide = vi.spyOn(api, "decide").mockReturnValue(new Promise<ProjectView>(() => {}));
    wrap(<StageRefusal pid="p" refusal={refusal} />);
    const note = screen.getByTestId(`stage-refusal-${refusal.stage}`);
    expect(note).toHaveAttribute("role", "alert");
    expect(note).toHaveAttribute("data-code", refusal.code);
    expect(note).toHaveTextContent(refusal.message);
    const exits = screen.getAllByTestId("stage-refusal-exit");
    expect(exits.map((b) => b.textContent)).toEqual([shown(take!.label)]);
    // The way forward that names no decision is said in words: no press here could carry it out.
    expect(note).toHaveTextContent(told!.label);
    expect(screen.queryByRole("button", { name: told!.label })).toBeNull();
    // No Try again: the refusal is for the current answers, so running them again refuses again.
    expect(screen.queryByRole("button", { name: /try again/i })).toBeNull();
    fireEvent.click(exits[0]!);
    await waitFor(() => expect(decide).toHaveBeenCalledTimes(1));
    expect(decide.mock.calls[0]![1]).toEqual(take!.decision);
    await waitFor(() => expect(exits[0]).toBeDisabled());
  });

  it("says the refusal's words and only its way forward that names no decision, where the card records", () => {
    // The decision is the card's to offer (Stage's recordHere false): its label names a choice, so
    // said here as a sentence it would read as a fact.
    const decide = vi.spyOn(api, "decide");
    wrap(<StageRefusal pid="p" refusal={refusal} recordHere={false} />);
    const note = screen.getByTestId(`stage-refusal-${refusal.stage}`);
    expect(note.textContent).toBe(`${refusal.message} ${told!.label}.`);
    expect(screen.queryAllByRole("button")).toEqual([]);
    expect(decide).not.toHaveBeenCalled();
  });

  it("says no confirmation an unsettled reading waits on as a sentence, where the card records", () => {
    // U's two ways forward are each a confirmation, the card's to ask: "`age` is an amount" and
    // "`age` holds codes" are alternatives, never two facts side by side.
    const unsettled = SERVED.U.refusal!;
    wrap(<StageRefusal pid="p" refusal={unsettled} recordHere={false} />);
    const note = screen.getByTestId(`stage-refusal-${unsettled.stage}`);
    expect(note.textContent).toBe(shown(unsettled.message));
    for (const x of unsettled.exits) expect(note).not.toHaveTextContent(shown(String(x.label)));
    expect(screen.queryAllByRole("button")).toEqual([]);
  });

  it("offers each confirmation an unsettled reading waits on", () => {
    const unsettled = SERVED.U.refusal!;
    wrap(<StageRefusal pid="p" refusal={unsettled} />);
    expect(screen.getAllByTestId("stage-refusal-exit").map((b) => b.textContent)).toEqual(
      unsettled.exits.map((x) => shown(String(x.label))),
    );
  });

  it("says why, with that refusal's own ways, when the server refuses the way forward pressed", async () => {
    const refused: Refusal = { error: { code: "toy_locked", message: "Not recorded for the toy.", exits: [] } };
    vi.spyOn(api, "decide").mockRejectedValue(new RefusalError(refused));
    wrap(<StageRefusal pid="p" refusal={refusal} />);
    fireEvent.click(screen.getByTestId("stage-refusal-exit"));
    expect(await screen.findByTestId("refusal")).toHaveTextContent(refused.error.message);
  });
});

/**
 * G12 (the owner's Q3): a way forward that names no decision names the card it opens (`target`,
 * core/exit_target.py, served as `{stage, key}`), and in the quest shell, which opens cards
 * (`OpenCard`), it is a press that opens that card. R's way forward that names none is given one in
 * the served shape; with no target, or where no shell opens cards, it is said in words as before.
 */
describe("StageRefusal's ways forward that name their card", () => {
  const refusal = SERVED.R.refusal!;
  const told = waysForward(refusal).find((x) => !x.decision)!;
  const target = { stage: "data", key: null };
  const targeted = { ...refusal, exits: refusal.exits.map((x) => (x.decision ? x : { ...x, target })) };

  it("keeps each way forward's target as the engine serves it", () => {
    expect(waysForward(targeted).find((x) => !x.decision)).toEqual({ ...told, target });
    expect(besideCard(targeted).told).toEqual([{ ...told, target }]);
  });

  it("is a press that opens the card it names, where the shell opens cards and the card records", () => {
    const open = vi.fn();
    const decide = vi.spyOn(api, "decide");
    wrap(
      <OpenCard.Provider value={open}>
        <StageRefusal pid="p" refusal={targeted} recordHere={false} />
      </OpenCard.Provider>,
    );
    const note = screen.getByTestId(`stage-refusal-${refusal.stage}`);
    // Its words are the refusal's alone; the way forward is the press, not a sentence beside it.
    expect(note.querySelector("p")!.textContent).toBe(refusal.message);
    const press = screen.getByRole("button", { name: told.label });
    expect(press).toHaveAttribute("data-target", "data");
    fireEvent.click(press);
    expect(open).toHaveBeenCalledWith(target);
    expect(decide).not.toHaveBeenCalled();
    // The decision is still the card's to offer: no press records it here.
    expect(screen.getAllByRole("button")).toEqual([press]);
  });

  it("is a press beside the decisions where the stage records", () => {
    const open = vi.fn();
    wrap(
      <OpenCard.Provider value={open}>
        <StageRefusal pid="p" refusal={targeted} />
      </OpenCard.Provider>,
    );
    expect(screen.getAllByTestId("stage-refusal-exit").map((b) => b.textContent)).toEqual([
      shown(refusal.exits[0]!.label as string),
    ]);
    fireEvent.click(screen.getByTestId("stage-refusal-to"));
    expect(open).toHaveBeenCalledWith(target);
  });

  it("is said in words where no shell opens cards", () => {
    wrap(<StageRefusal pid="p" refusal={targeted} recordHere={false} />);
    const note = screen.getByTestId(`stage-refusal-${refusal.stage}`);
    expect(note.textContent).toBe(`${refusal.message} ${told.label}.`);
    expect(screen.queryAllByRole("button")).toEqual([]);
  });
});

/**
 * b1-decision-exits (the owner's Q3: no dead end; FOUNDATION §1): where the card is the one place a
 * choice is recorded, a way forward that is a decision records nothing here, but where it names the
 * card that records it (`target`, as core/fit_press.py `triage_refusal` names the triage's card for
 * "Record as a limitation"), it is a press that opens that card. R's decision is given a target in
 * the served shape: its own line, where the card that records `set_selection` is.
 */
describe("StageRefusal's decisions that name the card that records them", () => {
  const refusal = SERVED.R.refusal!;
  const [take, told] = waysForward(refusal);
  const target = { stage: "models", key: "set_selection" };
  const named = { ...refusal, exits: refusal.exits.map((x) => (x.decision ? { ...x, target } : x)) };

  it("is a press that opens that card, and records nothing, where the shell opens cards and the card records", () => {
    const open = vi.fn();
    const decide = vi.spyOn(api, "decide");
    wrap(
      <OpenCard.Provider value={open}>
        <StageRefusal pid="p" refusal={named} recordHere={false} />
      </OpenCard.Provider>,
    );
    const note = screen.getByTestId(`stage-refusal-${refusal.stage}`);
    // The words are the refusal's and the way forward with no card; the decision is the press.
    expect(note.querySelector("p")!.textContent).toBe(`${refusal.message} ${told!.label}.`);
    const press = screen.getByRole("button", { name: take!.label });
    expect(press).toHaveAttribute("data-testid", "stage-refusal-to");
    expect(press).toHaveAttribute("data-target", "models:set_selection");
    expect(screen.queryAllByTestId("stage-refusal-exit")).toEqual([]);
    fireEvent.click(press);
    expect(open).toHaveBeenCalledWith(target);
    expect(decide).not.toHaveBeenCalled();
  });

  it("is neither a press nor a sentence where no shell opens cards, as a decision there always was", () => {
    wrap(<StageRefusal pid="p" refusal={named} recordHere={false} />);
    const note = screen.getByTestId(`stage-refusal-${refusal.stage}`);
    expect(note.textContent).toBe(`${refusal.message} ${told!.label}.`);
    expect(screen.queryAllByRole("button")).toEqual([]);
  });

  it("is the press that records it where the stage records, its target aside", async () => {
    const open = vi.fn();
    const decide = vi.spyOn(api, "decide").mockReturnValue(new Promise<ProjectView>(() => {}));
    wrap(
      <OpenCard.Provider value={open}>
        <StageRefusal pid="p" refusal={named} />
      </OpenCard.Provider>,
    );
    fireEvent.click(screen.getByTestId("stage-refusal-exit"));
    await waitFor(() => expect(decide).toHaveBeenCalledTimes(1));
    expect(decide.mock.calls[0]![1]).toEqual(take!.decision);
    expect(open).not.toHaveBeenCalled();
    expect(screen.queryAllByTestId("stage-refusal-to")).toEqual([]);
  });

  it("is neither where it names no card, though the shell opens cards: the card it is drawn beside offers it", () => {
    wrap(
      <OpenCard.Provider value={vi.fn()}>
        <StageRefusal pid="p" refusal={refusal} recordHere={false} />
      </OpenCard.Provider>,
    );
    expect(screen.queryAllByRole("button")).toEqual([]);
    expect(screen.getByTestId(`stage-refusal-${refusal.stage}`)).not.toHaveTextContent(take!.label);
  });
});
