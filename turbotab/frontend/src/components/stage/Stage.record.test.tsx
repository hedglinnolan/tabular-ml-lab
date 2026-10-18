/**
 * Recording from the stage (findings 4 and 12 of the 2026-10-17 walkthrough). Classic's stage keeps
 * its record button, the touch equivalent of Enter in the Record, and a refusal's ways forward,
 * each of which previews the decision it carries for that button to record. A screen whose card is
 * the one place a choice is recorded (the quest shell: FOUNDATION §1, the card's foot is the one
 * action) passes `recordHere={false}`: the stage draws neither, so the tapestry never offers a
 * second record button that disagrees with the card's, nor the refusal's readings again as a wall
 * of buttons beside the card that asks them. A refusal for the readings the card asks is the card's
 * to say, once (the re-walk's N4): the card asks them where its question is the open step with the
 * ledger's ask card (record/besideAsk.ts `askedBeside`; W1-W round 1, P4). Any other refusal, a
 * declaration card's among them, still says why there is no preview.
 *
 * The refusal is the families press's while four whole-number columns' readings are unconfirmed, as
 * core/decisions.py `_models_read_settled_readings` raises it: the fit's reason, then the ledger's
 * question (core/readings.py `ask_text`: each reading with its guess and its evidence), one block
 * confirmation, then two alternatives per column (`ask_exits`), then the roles question; 9 + 1
 * exits, by construction. In the shell the card asks those readings, so the stage leaves the
 * refusal to the card's answer to the press (record/besideAsk.ts); Classic's stage says it whole.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api, RefusalError } from "../../api/client";
import type { PreviewResult } from "../../api/m1-stage-types";
import type { Decision, ProjectView, Refusal } from "../../api/schema";
import type { StageFocus } from "../../state/focus";
import { Stage } from "./Stage";

const PID = "p-record";
const COLUMNS = ["bp_sys", "bp_di", "hdl", "triglycerides"];
const readings = (pairs: [string, string][]) =>
  ({
    kind: "confirm_readings",
    items: pairs.map(([column, value]) => ({ reading: "code_or_count", column, value })),
  }) as Extract<Decision, { kind: "confirm_readings" }>;
const confirm = (column: string, value: string) => ({
  label: `\`${column}\`: ${value === "amount" ? "an amount" : "codes for categories"}`,
  decision: readings([[column, value]]),
});
const REASON =
  "`bp_sys`, `bp_di`, `hdl` and `triglycerides` hold whole numbers, which may be codes for categories (one indicator per level) or amounts (one slope); the fit waits for the answer for each.";
/** Each column's evidence, by construction. */
const EVIDENCE: Record<string, string> = {
  bp_sys: "whole numbers from 64 to 230",
  bp_di: "whole numbers from 0 to 134",
  hdl: "whole numbers from 6 to 226",
  triglycerides: "whole numbers from 9 to 4233",
};
const ASK_TEXT = `Tell me about these columns: ${COLUMNS.map((c) => `\`${c}\`: an amount? (${EVIDENCE[c]})`).join("; ")}.`;
const REFUSAL: Refusal = {
  error: {
    code: "reading_unsettled",
    message: `${REASON} ${ASK_TEXT}`,
    exits: [
      {
        label: "Confirm every best guess as shown",
        decision: readings(COLUMNS.map((column) => [column, "amount"])),
      },
      ...COLUMNS.flatMap((c) => [confirm(c, "amount"), confirm(c, "code")]),
      { label: "Change the roles (the roles question)", decision: null },
    ],
  },
};
const WITH_DECISION = REFUSAL.error.exits.filter((x) => x.decision).length;
const RIDGE: Decision = { kind: "select_models", models: ["ridge"] };
const FOCUS: StageFocus = {
  kind: "option",
  decision: RIDGE,
  label: "Ridge",
  record: { kind: "select_models", models: ["linear", "ridge"] },
  recordLabel: "Choose these 2 families",
};
const PREVIEW: PreviewResult = {
  kind: "none",
  views: [],
  basis: "No views.",
  note: null,
  caution: null,
};

const VIEW = {
  summary: { id: PID, name: "NHANES", n_rows: 21849, n_cols: 29 },
  state: { purpose: "prediction", target: "glucose", roles: null, models: null },
  decisions: [],
  interview: [],
  stages: {},
} as unknown as ProjectView;

/**
 * The families question open with the ledger's ask card on it, which lists the four readings the
 * refusal waits on (core/ask.py `card`: the fit's consumer, its question "models"): where AskFor
 * draws that card and says a refusal for those readings beside it (`besideAsk`).
 */
const ASKING = {
  ...VIEW,
  interview: [
    {
      key: "models",
      status: "open",
      decision_id: null,
      reason: null,
      waiting_on: [],
      deferred_findings: [],
      followup: null,
      ask: {
        question: "models",
        consumer: "the fit",
        text: ASK_TEXT,
        groups: [{ kind: "code_or_count", columns: COLUMNS, guess: "amount", guess_words: "an amount", evidence: "whole numbers" }],
        exits: REFUSAL.error.exits,
        read_from_data: [],
      },
    },
  ],
} as unknown as ProjectView;

function mount(recordHere?: boolean, { view = VIEW, focus = FOCUS }: { view?: ProjectView; focus?: StageFocus } = {}) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={qc}>
      <Stage
        pid={PID}
        view={view}
        focus={focus}
        onFocus={() => {}}
        {...(recordHere === undefined ? {} : { recordHere })}
      />
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  vi.stubGlobal(
    "fetch",
    vi.fn(() => new Promise(() => {})),
  );
});
afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("a refused preview on the stage", () => {
  beforeEach(() => {
    vi.spyOn(api, "preview").mockRejectedValue(new RefusalError(REFUSAL));
  });

  /** A column's reading as the ledger's question words it, as the refusal shows it (no backticks). */
  const asked = (c: string) => `${c}: an amount? (${EVIDENCE[c]})`;

  it("keeps its record button, each way forward and the ledger's question on Classic's stage", async () => {
    mount();
    await waitFor(() =>
      expect(screen.getByTestId("stage")).toHaveAttribute("data-scene", "refusal"),
    );
    expect(screen.getByTestId("stage-record")).toHaveTextContent("Choose these 2 families");
    expect(screen.getAllByTestId("refusal-exit")).toHaveLength(WITH_DECISION);
    expect(WITH_DECISION).toBe(9);
    const said = screen.getByTestId("refusal");
    expect(said).toHaveTextContent("the fit waits for the answer for each");
    expect(said).toHaveTextContent("Tell me about these columns:");
    for (const c of COLUMNS) expect(said).toHaveTextContent(asked(c));
  });

  it("draws none of it where the card records and asks those readings: the card says it, once (N4)", async () => {
    // The re-walk's N4: the families refusal was said twice, on the card and again here. Where the
    // card asks the readings a refusal waits on (its question the open step with the ask card), the
    // card's answer to the press is the one place it is said; the stage stays on the pipeline.
    mount(false, { view: ASKING });
    await waitFor(() => expect(api.preview).toHaveBeenCalled());
    // The refused preview is answered (Classic's stage draws it by now, above); this one draws none.
    await act(async () => {
      await new Promise((r) => setTimeout(r, 100));
    });
    expect(screen.getByTestId("stage")).toHaveAttribute("data-scene", "now");
    expect(screen.queryByTestId("refusal")).toBeNull();
    expect(document.body).not.toHaveTextContent("the fit waits for the answer for each");
    for (const c of COLUMNS) expect(document.body).not.toHaveTextContent(asked(c));
    expect(screen.queryByTestId("stage-record")).toBeNull();
    expect(screen.queryAllByTestId("refusal-exit")).toHaveLength(0);
  });

  /**
   * W1-W round 1, P4: the stage kept quiet for any refusal with a reading's confirmation among its
   * ways forward, whichever card set the focus. Intended use is a declaration card with no ask card
   * (quest/cards/predict), so a use refused for a subgroup whose code-or-amount reading is unsettled
   * (core/models/decision_curve.py `_subgroups_read_settled_readings`: its reason, then the
   * ledger's question for that one column, one confirmation per reading) was said nowhere while
   * chosen. It is said on the stage, its reason up to the ledger's question, even while the families
   * question is open with an ask card listing that column: that card is not the one beside it.
   */
  it("still says a refusal for readings where the card that set the focus asks none (P4)", async () => {
    const subgroups: Refusal = {
      error: {
        code: "reading_unsettled",
        message:
          "Subgroup performance groups a column by its levels when its numbers are codes and by its thirds when they are an amount, and this one is not settled. " +
          `Tell me about this column: \`bp_sys\`: an amount? (${EVIDENCE.bp_sys})`,
        exits: [confirm("bp_sys", "amount"), confirm("bp_sys", "code")],
      },
    };
    vi.mocked(api.preview).mockRejectedValue(new RefusalError(subgroups));
    const use = { kind: "set_intended_use", use: "risk_estimation", subgroups: ["bp_sys"] } as unknown as Decision;
    mount(false, { view: ASKING, focus: { kind: "option", decision: use, label: "Risk estimation" } });
    await waitFor(() => expect(screen.getByTestId("stage")).toHaveAttribute("data-scene", "refusal"));
    const said = screen.getByTestId("refusal");
    expect(said).toHaveTextContent(
      "Not available: Subgroup performance groups a column by its levels when its numbers are codes and by its thirds when they are an amount, and this one is not settled.",
    );
    expect(said).not.toHaveTextContent("Tell me about");
    expect(screen.queryAllByTestId("refusal-exit")).toHaveLength(0);
  });

  it("says the families refusal where the families question is not the open step with the ask card", async () => {
    // No card beside it says the refusal for those readings (AskFor trims a refusal only beside its
    // own open ask card), so the stage says why there is no preview, up to the ledger's question.
    mount(false, { view: VIEW });
    await waitFor(() => expect(screen.getByTestId("stage")).toHaveAttribute("data-scene", "refusal"));
    const said = screen.getByTestId("refusal");
    expect(said).toHaveTextContent("the fit waits for the answer for each");
    expect(said).not.toHaveTextContent("Tell me about");
    expect(screen.queryAllByTestId("refusal-exit")).toHaveLength(0);
  });

  it("still says a refusal the card asks nothing of, where the card records", async () => {
    const plain: Refusal = {
      error: { code: "models_task", message: "These families cannot fit a numeric outcome.", exits: [{ label: "Choose other families", decision: null }] },
    };
    vi.mocked(api.preview).mockRejectedValue(new RefusalError(plain));
    mount(false);
    await waitFor(() => expect(screen.getByTestId("stage")).toHaveAttribute("data-scene", "refusal"));
    expect(screen.getByTestId("refusal")).toHaveTextContent("These families cannot fit a numeric outcome.");
    expect(screen.queryAllByTestId("refusal-exit")).toHaveLength(0);
  });
});

describe("a preview on the stage", () => {
  beforeEach(() => {
    vi.spyOn(api, "preview").mockResolvedValue(PREVIEW);
  });

  it("offers its record button on Classic's stage", async () => {
    mount(true);
    await waitFor(() => expect(screen.getByTestId("stage-record")).toBeEnabled());
  });

  it("offers none where the card records", async () => {
    mount(false);
    await waitFor(() =>
      expect(screen.getByTestId("stage")).toHaveAttribute("data-scene", "preview"),
    );
    expect(screen.queryByTestId("stage-record")).toBeNull();
  });
});

/**
 * W1-R R3 P2: a preview served with a caution that asks the fit's unconfirmed readings (core/
 * plan_previews.py `asks_first(offer=True)`: the caution's text is the ledger's question, its
 * exits each confirmation) offers those confirmations as ways forward, each previewing a decision
 * for the stage's record button. Where the card records, no such button exists, so they would lead
 * nowhere: the shell draws the caution's words and no way forward; Classic keeps them.
 */
describe("a preview caution's ways forward", () => {
  const CAUTION = `Tell me about these columns: ${COLUMNS.slice(0, 2).map((c) => `\`${c}\`: an amount? (${EVIDENCE[c]})`).join("; ")}.`;
  const CAUTIONED: PreviewResult = {
    ...PREVIEW,
    caution: { text: CAUTION, exits: [confirm("bp_sys", "amount"), confirm("bp_di", "amount")] } as PreviewResult["caution"],
  };
  beforeEach(() => {
    vi.spyOn(api, "preview").mockResolvedValue(CAUTIONED);
  });

  it("are drawn on Classic's stage, beside its record button", async () => {
    mount(true);
    const caution = await screen.findByTestId("stage-caution");
    expect(caution).toHaveTextContent("Tell me about these columns:");
    expect(caution.querySelectorAll("button")).toHaveLength(2);
  });

  it("are not drawn where the card records, and the caution's words stay", async () => {
    mount(false);
    const caution = await screen.findByTestId("stage-caution");
    expect(caution).toHaveTextContent("Tell me about these columns:");
    expect(caution.querySelectorAll("button")).toHaveLength(0);
    expect(screen.queryByTestId("stage-record")).toBeNull();
  });
});
