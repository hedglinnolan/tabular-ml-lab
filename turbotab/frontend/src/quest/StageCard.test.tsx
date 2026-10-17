/**
 * The Results stage's card column (src/quest/contracts.ts; plans/W1_PART2_PLAN.md §2 "W1-E": "the
 * Results stage goes to W1-G's ResultsCards"). Once Results is reached, ResultsCards is the column,
 * mounted once in place of the stage's lines, whichever line is open and after the opening; the
 * stage's head, For the record and the primary action at the foot stay the shell's. Before Results
 * is reached, its lines wait as any stage's do, and ResultsCards is not mounted.
 *
 * W1-G's stubs (src/quest/results) are replaced here by probes that count their mounts. The view is
 * a real server's captured NHANES walk under Predict; the quest logs are the engine's
 * (fixtures/quest-contract.json), with the Results lines' statuses set for each moment.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, vi } from "vitest";
import { keys } from "../api/queries";
import type { ForTheRecord, ProjectView, QuestLine, QuestLog } from "../api/schema";
import { expandView, viewsOf, type M3Fixture } from "../mocks/m3";
import { CONTRACT } from "./fixtures/contract";
import { QuestScreen } from "./QuestScreen";

vi.mock("./results", () => ({
  ResultsCards: ({ pid }: { pid: string }) => <p data-testid="probe-results-cards">{pid}</p>,
  ResultsTapestry: ({ pid }: { pid: string }) => <p data-testid="probe-results-tapestry">{pid}</p>,
}));

const FIXTURES = import.meta.glob<M3Fixture>("../mocks/fixtures/m3-nhanes-prediction.json", { eager: true, import: "default" });
const JOURNEY = Object.values(FIXTURES)[0]!;
const PID = "quest-results";
const VIEW = ((): ProjectView => {
  const views = viewsOf(JOURNEY);
  return expandView(JOURNEY, views[views.length - 1]!, PID, "2026-10-17T00:00:00Z");
})();
const { pressed } = CONTRACT.fit_locks;
const REACHED = CONTRACT.logs.results_reached.log;
const NOT_REACHED = CONTRACT.logs.models_open.log;
const RESULTS_KEYS = ["set_updating", "open_seal", "set_explain"];

/** The engine's log with Fit pressed and the Results lines as `set` makes them. */
function logWith(base: QuestLog, set: (l: QuestLine) => QuestLine = (l) => l): QuestLog {
  return {
    ...base,
    fit: base === REACHED ? pressed : base.fit,
    stages: base.stages.map((s) => (s.key === "results" ? { ...s, lines: s.lines.map(set) } : s)),
  };
}

/** For the record, with one Results line. */
const RECORD: ForTheRecord = {
  version: 1,
  stages: [
    {
      key: "results",
      name: "Results",
      lines: [{ kind: "noted", key: "k", id: null, text: "A line for the record.", decision_id: null }],
    },
  ],
};

function shell(log: QuestLog) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  qc.setQueryData(keys.quest(PID), log);
  qc.setQueryData(keys.view(PID), VIEW);
  qc.setQueryData(keys.record(PID), RECORD);
  const { unmount } = render(
    <QueryClientProvider client={qc}>
      <QuestScreen pid={PID} stream="open" />
    </QueryClientProvider>,
  );
  fireEvent.click(screen.getByTestId("stage-results"));
  return { card: screen.getByTestId("card-results"), unmount };
}

// Whatever is not seeded stays pending: nothing here reaches a server.
beforeEach(() => {
  vi.stubGlobal("fetch", vi.fn(() => new Promise(() => {})));
});
afterEach(() => {
  vi.unstubAllGlobals();
});

describe("the Results stage's card column", () => {
  it("is ResultsCards, once, on entering Results with the shrinkage question open", () => {
    const log = logWith(REACHED);
    const results = log.stages.find((s) => s.key === "results")!;
    expect(results.reached).toBe(true);
    expect(results.lines.find((l) => l.status === "open" && l.counted)?.key).toBe("set_updating");
    const { card } = shell(log);
    expect(screen.getAllByTestId("probe-results-cards")).toHaveLength(1);
    expect(within(card).getByTestId("results-column")).toContainElement(screen.getByTestId("probe-results-cards"));
    // The column draws the stage's lines; the shell draws none of them beside it.
    for (const key of RESULTS_KEYS) expect(within(card).queryByTestId(`line-${key}`)).toBeNull();
    // The head, For the record and the foot stay the shell's.
    expect(within(card).getByRole("heading", { level: 2 })).toHaveTextContent(results.name);
    expect(within(card).getByTestId("for-the-record")).toHaveTextContent("For the record · 1");
    expect(within(card).getByRole("button", { name: "Continue" })).toBeDisabled();
    // The tapestry's half is mounted once too, in the tapestry.
    expect(screen.getAllByTestId("probe-results-tapestry")).toHaveLength(1);
    expect(screen.getByTestId("tapestry-results")).toContainElement(screen.getByTestId("probe-results-tapestry"));
  });

  it("is ResultsCards, once, after the opening, whatever line is still open", () => {
    const opened = logWith(REACHED, (l) =>
      l.key === "set_explain" ? l : { ...l, status: "answered", decision_id: `r-${l.key}` },
    );
    const first = shell(opened);
    expect(screen.getAllByTestId("probe-results-cards")).toHaveLength(1);
    for (const key of RESULTS_KEYS) expect(within(first.card).queryByTestId(`line-${key}`)).toBeNull();
    first.unmount();

    const settled = logWith(REACHED, (l) => ({ ...l, status: "answered", decision_id: `r-${l.key}` }));
    shell(settled);
    expect(screen.getAllByTestId("probe-results-cards")).toHaveLength(1);
  });

  it("is not mounted before Results is reached: the stage's lines wait there, as any stage's do", () => {
    const log = logWith(NOT_REACHED);
    expect(log.stages.find((s) => s.key === "results")!.reached).toBe(false);
    const { card } = shell(log);
    expect(screen.queryByTestId("probe-results-cards")).toBeNull();
    expect(within(card).queryByTestId("results-column")).toBeNull();
    for (const key of RESULTS_KEYS) expect(within(card).getByTestId(`line-${key}`)).toHaveAttribute("data-status", "waiting");
  });
});
