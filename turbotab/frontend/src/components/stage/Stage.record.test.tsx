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
 * core/decisions.py `_models_read_settled_readings` raises it: one block confirmation, then two
 * alternatives per column (core/readings.py `ask_exits`), then the roles question; 9 + 1 exits, by
 * construction.
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
const REFUSAL: Refusal = {
  error: {
    code: "reading_unsettled",
    message: `\`bp_sys\`, \`bp_di\`, \`hdl\` and \`triglycerides\` hold whole numbers, which may be codes for categories (one indicator per level) or amounts (one slope); the fit waits for the answer for each.`,
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

  it("keeps its record button and each way forward on Classic's stage", async () => {
    mount();
    await waitFor(() =>
      expect(screen.getByTestId("stage")).toHaveAttribute("data-scene", "refusal"),
    );
    expect(screen.getByTestId("stage-record")).toHaveTextContent("Choose these 2 families");
    expect(screen.getAllByTestId("refusal-exit")).toHaveLength(WITH_DECISION);
    expect(WITH_DECISION).toBe(9);
  });

  it("draws neither where the card records, and still says why there is no preview", async () => {
    mount(false);
    await waitFor(() =>
      expect(screen.getByTestId("stage")).toHaveAttribute("data-scene", "refusal"),
    );
    expect(screen.getByTestId("refusal")).toHaveTextContent(
      "the fit waits for the answer for each",
    );
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
