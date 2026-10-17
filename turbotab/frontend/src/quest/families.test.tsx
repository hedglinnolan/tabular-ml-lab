/**
 * The families card in the shell (finding 12 of the 2026-10-17 walkthrough).
 *
 * The readings the fit waits on show once: on the card, in the ledger's ask card on the families
 * question. A families press refused for them says what it waits for, in the consumer's own words,
 * and leaves the readings to that card (no wall of confirmations again under the press). And a
 * reading confirmed there keeps the families chosen: the confirmation rebuilds the table, the
 * families line waits while the shelf is computed again, and the card is drawn anew when it opens,
 * with the choice as it was. A choice held against a record that has changed since is not restored.
 *
 * The view is a real server's captured NHANES walk under Predict where the families question is
 * open (src/mocks/fixtures), each fresh stage's artifact as captured; the quest log is the engine's
 * (fixtures/quest-contract.json) with Models' selection answered and the families line open. The ask
 * card and the refusal are the engine's shapes (core/ask.py, core/decisions.py
 * `_models_read_settled_readings`: the fit's reason, the ledger's question after it, one block
 * confirmation, two alternatives per column, then the roles question), set by construction.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, vi } from "vitest";
import { api, RefusalError } from "../api/client";
import type { TeachingEntry } from "../api/m1-types";
import type { AskCard } from "../api/m3-types";
import { keys } from "../api/queries";
import type { Decision, ProjectView, QuestLine, QuestLog, Refusal } from "../api/schema";
import { besideAsk } from "../components/record/besideAsk";
import { artifactAt, expandView, viewsOf, type M3Fixture } from "../mocks/m3";
import { CONTRACT } from "./fixtures/contract";
import { QuestScreen } from "./QuestScreen";

const FIXTURES = import.meta.glob<M3Fixture>("../mocks/fixtures/m3-nhanes-prediction.json", {
  eager: true,
  import: "default",
});
const JOURNEY = Object.values(FIXTURES)[0]!;
const SHARED = Object.values(
  import.meta.glob<{ teaching: TeachingEntry[] }>("../mocks/fixtures/m3-shared.json", {
    eager: true,
    import: "default",
  }),
)[0]!;
const PID = "quest-families";
const VIEWS = viewsOf(JOURNEY);
const AT = VIEWS.findIndex((v) =>
  v.interview.some((s) => s.key === "models" && s.status === "open"),
);
const VIEW = expandView(JOURNEY, VIEWS[AT]!, PID, "2026-10-17T00:00:00Z");
const SHELF = artifactAt(JOURNEY, "shelf", VIEW.stages.shelf!.key) as {
  families: { key: string; label: string; rank: number }[];
};

const COLUMNS = ["bp_sys", "bp_di", "hdl", "triglycerides"];
const readings = (pairs: [string, string][]) =>
  ({
    kind: "confirm_readings",
    items: pairs.map(([column, value]) => ({ reading: "code_or_count", column, value })),
  }) as Extract<Decision, { kind: "confirm_readings" }>;
const REASON =
  "`bp_sys`, `bp_di`, `hdl` and `triglycerides` hold whole numbers, which may be codes for categories (one indicator per level) or amounts (one slope); the fit waits for the answer for each.";
const ASK_TEXT = `Tell me about these columns: ${COLUMNS.map((c) => `\`${c}\`: an amount? (whole numbers from 20 to 230)`).join("; ")}.`;
const ROLES_EXIT = { label: "Change the roles (the roles question)", decision: null };
const REFUSAL: Refusal = {
  error: {
    code: "reading_unsettled",
    message: `${REASON} ${ASK_TEXT}`,
    exits: [
      {
        label: "Confirm every best guess as shown",
        decision: readings(COLUMNS.map((c) => [c, "amount"])),
      },
      ...COLUMNS.flatMap((c) => [
        { label: `\`${c}\`: an amount`, decision: readings([[c, "amount"]]) },
        { label: `\`${c}\`: codes for categories`, decision: readings([[c, "code"]]) },
      ]),
      ROLES_EXIT,
    ],
  },
};
const ASK: AskCard = {
  question: "models",
  consumer: "The fit",
  text: ASK_TEXT,
  groups: COLUMNS.map((c) => ({
    kind: "code_or_count",
    columns: [c],
    guess: "amount",
    guess_words: "an amount",
    evidence: "whole numbers from 20 to 230",
  })),
  exits: REFUSAL.error.exits.filter((x) => x.decision) as AskCard["exits"],
  read_from_data: [],
} as AskCard;

/** The engine's log with Models' selection answered and the families line as `status` makes it. */
function logWith(status: QuestLine["status"]): QuestLog {
  const base = CONTRACT.logs.models_open.log;
  return {
    ...base,
    stages: base.stages.map((s) =>
      s.key === "models"
        ? {
            ...s,
            lines: s.lines.map((l) =>
              l.key === "set_selection"
                ? { ...l, status: "answered", decision_id: "r-selection" }
                : l.key === "models"
                  ? {
                      ...l,
                      status,
                      waiting_for: [],
                      computing: status === "waiting" ? ["shelf"] : [],
                    }
                  : l,
            ),
          }
        : s,
    ),
  };
}

function shell(view: ProjectView) {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false, staleTime: Infinity } },
  });
  qc.setQueryData(keys.teaching(), SHARED.teaching);
  qc.setQueryData(keys.quest(PID), logWith("open"));
  qc.setQueryData(keys.view(PID), view);
  for (const [name, status] of Object.entries(view.stages)) {
    if (!status || status.status !== "fresh") continue;
    qc.setQueryData(keys.stage(PID, name), {
      stage: name,
      key: status.key,
      fresh: true,
      status: status.status,
      artifact: artifactAt(JOURNEY, name, status.key),
    });
  }
  render(
    <QueryClientProvider client={qc}>
      <QuestScreen pid={PID} stream="open" />
    </QueryClientProvider>,
  );
  fireEvent.click(screen.getByTestId("stage-models"));
  return qc;
}

/** The two families the shelf ranks first, chosen on the card. */
function chooseTwo() {
  const two = [...SHELF.families].sort((a, b) => a.rank - b.rank).slice(0, 2);
  for (const f of two) fireEvent.click(screen.getByTestId(`option-${f.key}`));
  return two;
}

beforeEach(() => {
  vi.stubGlobal(
    "fetch",
    vi.fn(() => new Promise(() => {})),
  );
  // The views measure their box; jsdom has no layout to measure.
  vi.stubGlobal(
    "ResizeObserver",
    class {
      observe() {}
      unobserve() {}
      disconnect() {}
    },
  );
  vi.spyOn(api, "preview").mockReturnValue(new Promise(() => {}));
});
afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

/** The query cache tells its observers on a timer. */
const notified = () => new Promise((r) => setTimeout(r, 0));

describe("the families chosen, across a confirmation that recomputes the shelf", () => {
  it("are chosen still when the families line opens again", async () => {
    expect(AT).toBeGreaterThan(0);
    expect(VIEW.state.models).toBeNull();
    const qc = shell(VIEW);
    const two = chooseTwo();
    expect(screen.getByTestId("record-models")).toHaveTextContent("Choose these 2 families");
    // The confirmation rebuilds the table: the line waits while the shelf is computed again.
    await act(async () => {
      qc.setQueryData(keys.quest(PID), logWith("waiting"));
      await notified();
    });
    expect(screen.queryByTestId("record-models")).toBeNull();
    await act(async () => {
      qc.setQueryData(keys.quest(PID), logWith("open"));
      await notified();
    });
    const button = await screen.findByTestId("record-models");
    expect(button).toHaveTextContent("Choose these 2 families");
    expect(button).toBeEnabled();
    for (const f of two)
      expect(screen.getByTestId(`option-${f.key}`)).toHaveAttribute("data-chosen");
  });

  it("start from the record instead where the record changed meanwhile", async () => {
    const qc = shell(VIEW);
    chooseTwo();
    await act(async () => {
      qc.setQueryData(keys.quest(PID), logWith("waiting"));
      await notified();
    });
    const linear = SHELF.families.find((f) => f.key === "linear")!;
    await act(async () => {
      qc.setQueryData<ProjectView>(keys.view(PID), {
        ...VIEW,
        state: { ...VIEW.state, models: ["linear"] },
      });
      qc.setQueryData(keys.quest(PID), logWith("open"));
      await notified();
    });
    expect(await screen.findByTestId("record-models")).toHaveTextContent(
      `Choose the ${linear.label.toLowerCase()}`,
    );
    expect(document.querySelectorAll('[data-testid="options-models"] [data-chosen]')).toHaveLength(
      1,
    );
  });
});

describe("a families press refused for the readings the card asks", () => {
  const asking: ProjectView = {
    ...VIEW,
    interview: VIEW.interview.map((s) => (s.key === "models" ? { ...s, ask: ASK } : s)),
  };

  it("says what it waits for in the fit's words, and leaves the readings to the ask card", async () => {
    shell(asking);
    const ask = screen.getByTestId("ask-card");
    expect(within(ask).getAllByTestId("ask-confirm")).toHaveLength(COLUMNS.length);
    vi.spyOn(api, "decide").mockRejectedValue(new RefusalError(REFUSAL));
    chooseTwo();
    fireEvent.click(screen.getByTestId("record-models"));
    const note = await screen.findByTestId("refusal");
    expect(note).toHaveTextContent("the fit waits for the answer for each.");
    expect(note).not.toHaveTextContent("Tell me about");
    expect(
      within(note)
        .getAllByTestId("refusal-exit")
        .map((b) => b.textContent),
    ).toEqual([ROLES_EXIT.label]);
    // The readings' confirmations are on the card once: the ask card's.
    expect(screen.getAllByTestId("ask-confirm")).toHaveLength(COLUMNS.length);
  });

  it("is said whole where the refusal offers no confirmation of a reading", async () => {
    shell(asking);
    const other: Refusal = {
      error: {
        code: "models_task",
        message: "These families cannot fit a numeric outcome. Tell me about this column: `x`.",
        exits: [ROLES_EXIT],
      },
    };
    vi.spyOn(api, "decide").mockRejectedValue(new RefusalError(other));
    chooseTwo();
    fireEvent.click(screen.getByTestId("record-models"));
    const note = await screen.findByTestId("refusal");
    await waitFor(() => expect(note).toHaveTextContent("Tell me about this column: x."));
  });
});

describe("besideAsk", () => {
  it("keeps the refusal's own words up to the ledger's question, and its ways forward but the confirmations", () => {
    const out = besideAsk(REFUSAL);
    expect(out.error.message).toBe(REASON);
    expect(out.error.exits).toEqual([ROLES_EXIT]);
    expect(out.error.code).toBe(REFUSAL.error.code);
  });

  it("returns a refusal that offers no confirmation as it is", () => {
    const plain: Refusal = {
      error: { code: "c", message: `${REASON} ${ASK_TEXT}`, exits: [ROLES_EXIT] },
    };
    expect(besideAsk(plain)).toBe(plain);
  });
});
