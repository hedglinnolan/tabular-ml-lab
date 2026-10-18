/**
 * The Columns panel of "Your data now" (the walkthrough of 2026-10-17, finding 1; W1-R R1 P2): it
 * says how many columns proposed as predictors wait for their own role confirmation and are not
 * predictors yet (W1-T: a flag is never one, so it is not counted), counted from the quest log's
 * served role lines (core/quest.py `_role_lines`, here the engine's own lines
 * in src/quest/cards/roles/fixture.json), and says nothing of it where none waits. Where every
 * column waits, its empty sentence names those confirmations instead of the roles question.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { afterEach, beforeEach, vi } from "vitest";
import type { CohortArtifact } from "../../api/m1-stage-types";
import { keys } from "../../api/queries";
import type { ProjectView, QuestLine, QuestLog } from "../../api/schema";
import { expandView, viewsOf, type M3Fixture } from "../../mocks/m3";
import ROLES from "../../quest/cards/roles/fixture.json";
import { CONTRACT } from "../../quest/fixtures/contract";
import { StageFocusProvider } from "../../state/focus";
import { ColumnsCard, type LiveData } from "./LiveScenes";

const FIXTURES = import.meta.glob<M3Fixture>("../../mocks/fixtures/m3-nhanes-prediction.json", { eager: true, import: "default" });
const JOURNEY = Object.values(FIXTURES)[0]!;
const PID = "live-columns";
const VIEW: ProjectView = (() => {
  const views = viewsOf(JOURNEY);
  return expandView(JOURNEY, views[views.length - 1]!, PID, "2026-10-17T00:00:00Z");
})();
const LINES = ROLES.lines as unknown as QuestLine[];
/** The engine's role lines still open: each waits for its own confirmation. */
const OPEN = LINES.filter((l) => l.status === "open");

/** The contract's Predict log with `lines` added to Your data. */
function logWith(lines: QuestLine[]): QuestLog {
  const log = structuredClone(CONTRACT.logs.models_open.log) as QuestLog;
  log.stages.find((s) => s.key === "data")!.lines.push(...lines);
  return log;
}

/** The live data with the cohort's predictors as `predictors`, before any design. */
function data(predictors: string[]): LiveData {
  const cohort = { predictors, steps: [] } as unknown as CohortArtifact;
  return {
    pid: PID,
    cohort: { artifact: cohort, veil: "fresh" },
    splitVeil: "fresh",
    designVeil: "fresh",
    fit: { artifact: null, veil: "fresh" },
    shelf: null,
    design: null,
    split: null,
    substitution: { artifact: null, veil: "fresh" },
  };
}

function mount(log: QuestLog, predictors: string[]) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  qc.setQueryData(keys.quest(PID), log);
  render(
    <QueryClientProvider client={qc}>
      <StageFocusProvider>
        <ColumnsCard view={VIEW} data={data(predictors)} full={false} />
      </StageFocusProvider>
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  vi.stubGlobal("fetch", vi.fn(() => new Promise(() => {})));
  // The lineage measures its box; jsdom has no layout to measure.
  vi.stubGlobal(
    "ResizeObserver",
    class {
      observe() {}
      unobserve() {}
      disconnect() {}
    },
  );
});
afterEach(() => {
  vi.unstubAllGlobals();
});

/** The engine's open line of `bp_sys`, a column proposed as a predictor, for another column. */
function predictorLine(column: string): QuestLine {
  const bp = OPEN.find((l) => l.items[0]!.column === "bp_sys")!;
  const swap = (text: string) => text.replaceAll("bp_sys", column);
  return {
    ...bp,
    key: swap(bp.key),
    name: swap(bp.name),
    reason: bp.reason ? swap(bp.reason) : null,
    items: bp.items.map((i) => ({ ...i, column, words: swap(i.words), options: i.options.map((o) => ({ ...o, label: swap(o.label), decision: { ...o.decision, column } })) })),
  };
}

describe("the Columns panel and the roles that wait (finding 1)", () => {
  it("counts the waiting columns proposed as predictors, never a flag or a column proposed as left out (W1-T)", () => {
    // The engine's open lines: bp_sys proposed in a predictor role, a flag and a column proposed as
    // left out, which no confirmation makes a predictor.
    expect(OPEN.map((l) => [l.items[0]!.column, l.items[0]!.value])).toEqual([
      ["bp_sys", "covariate"],
      ["imputed_waist", "flag"],
      ["cycle_begin_year", "excluded"],
    ]);
    mount(logWith(LINES), ["age", "bmi"]);
    expect(screen.getByTestId("columns-waiting")).toHaveTextContent(/^1 column waits for its own confirmation and is not a predictor yet\.$/);
  });

  it("says it of several columns in the plural", () => {
    mount(logWith([...LINES, predictorLine("hdl")]), ["age", "bmi"]);
    expect(screen.getByTestId("columns-waiting")).toHaveTextContent(/^2 columns wait for their own confirmation and are not predictors yet\.$/);
  });

  it("says nothing of it where no role line is open", () => {
    mount(logWith(LINES.filter((l) => l.status !== "open")), ["age", "bmi"]);
    expect(screen.queryByTestId("columns-waiting")).toBeNull();
  });

  it("says nothing of it where the only open role lines are a flag's and an exclusion's", () => {
    mount(logWith(LINES.filter((l) => l.items[0]!.column !== "bp_sys")), ["age", "bmi"]);
    expect(screen.queryByTestId("columns-waiting")).toBeNull();
  });

  it("names the confirmations that wait, not the roles question, where no column is a predictor yet, saying yet once", () => {
    mount(logWith([...LINES, predictorLine("hdl")]), []);
    const said = screen.getByTestId("columns-waiting");
    expect(said).toHaveTextContent(/^No column is a predictor yet: 2 columns wait for their own confirmation\.$/);
    expect(said.textContent!.match(/\byet\b/g)).toHaveLength(1);
    expect(document.body).not.toHaveTextContent("the roles question");
  });

  it("names the roles question where no column is a predictor and none waits", () => {
    mount(logWith([]), []);
    expect(document.body).toHaveTextContent("No column is a predictor yet: the roles question decides which columns enter the model.");
    expect(screen.queryByTestId("columns-waiting")).toBeNull();
  });
});
