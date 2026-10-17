/**
 * Recording from the stage (findings 4 and 12 of the 2026-10-17 walkthrough). Classic's stage keeps
 * its record button, the touch equivalent of Enter in the Record, and a refusal's ways forward,
 * each of which previews the decision it carries for that button to record. A screen whose card is
 * the one place a choice is recorded (the quest shell: FOUNDATION §1, the card's foot is the one
 * action) passes `recordHere={false}`: the stage draws neither, so the tapestry never offers a
 * second record button that disagrees with the card's, nor the refusal's readings again as a wall
 * of buttons beside the card that asks them. The refusal still says why there is no preview.
 *
 * The refusal is the families press's while four whole-number columns' readings are unconfirmed, as
 * core/decisions.py `_models_read_settled_readings` raises it: the fit's reason, then the ledger's
 * question (core/readings.py `ask_text`: each reading with its guess and its evidence), one block
 * confirmation, then two alternatives per column (`ask_exits`), then the roles question; 9 + 1
 * exits, by construction. In the shell the card asks those readings, so the stage keeps the fit's
 * reason and leaves the ledger's question to the card (record/besideAsk.ts, as the card's own
 * answer to a refused press does); Classic's stage says the refusal whole.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
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

function mount(recordHere?: boolean) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={qc}>
      <Stage
        pid={PID}
        view={VIEW}
        focus={FOCUS}
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

  it("draws neither where the card records, and says why there is no preview without the readings the card asks", async () => {
    mount(false);
    await waitFor(() =>
      expect(screen.getByTestId("stage")).toHaveAttribute("data-scene", "refusal"),
    );
    const said = screen.getByTestId("refusal");
    expect(said).toHaveTextContent("the fit waits for the answer for each");
    expect(said).not.toHaveTextContent("Tell me about");
    for (const c of COLUMNS) expect(said).not.toHaveTextContent(asked(c));
    expect(screen.queryByTestId("stage-record")).toBeNull();
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
