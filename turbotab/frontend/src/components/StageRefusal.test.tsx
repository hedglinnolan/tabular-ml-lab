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
