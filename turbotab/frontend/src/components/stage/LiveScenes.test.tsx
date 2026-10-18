/**
 * The Columns panel of "Your data now" (the walkthrough of 2026-10-17, finding 1; W1-R R1 P2): it
 * says how many columns proposed as predictors wait for their own role confirmation and are not
 * predictors yet (W1-T: a flag is never one, so it is not counted), counted from the quest log's
 * served role lines (core/quest.py `_role_lines`, here the engine's own lines
 * in src/quest/cards/roles/fixture.json), each column of a group's line among them (ruling R-g),
 * and says nothing of it where none waits. Where every column waits, its empty sentence names those
 * confirmations instead of the roles question. The count comes with the stage's data, from the
 * quest log the stage's parent reads (U9): the card asks the server nothing.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, vi } from "vitest";
import type { CohortArtifact } from "../../api/m1-stage-types";
import type { ProjectView, QuestLine, QuestLog } from "../../api/schema";
import { expandView, viewsOf, type M3Fixture } from "../../mocks/m3";
import ROLES from "../../quest/cards/roles/fixture.json";
import { CONTRACT } from "../../quest/fixtures/contract";
import { StageFocusProvider } from "../../state/focus";
import { servedGroups, waitingPredictorColumns } from "../../quest/cards/roles/roles";
import { ColumnsCard, inGroupWords, predictorsLineage, type LiveData } from "./LiveScenes";

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

/** The live data with the cohort's predictors as `predictors`, before any design, and the waiting
 *  count the stage's parent passes (`waiting`). */
function data(predictors: string[], waiting: number | null): LiveData {
  const cohort = { predictors, steps: [] } as unknown as CohortArtifact;
  return {
    waitingPredictors: waiting,
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

/** The card as the quest shell's Tapestry draws it: the waiting count from the quest log it reads. */
function mount(log: QuestLog | null, predictors: string[]) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  render(
    <QueryClientProvider client={qc}>
      <StageFocusProvider>
        <ColumnsCard view={VIEW} data={data(predictors, log ? waitingPredictorColumns(log) : null)} full={false} />
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

/**
 * The re-walk's N3: under Predict the lineage named its groups "13 covariate columns" and "7 exposure
 * columns", Estimate's words. It names them as the roles card does, in the predictor groups the server
 * serves for the project (core/readings.py `served_groups`; here the engine's own, from the roles
 * fixture, a dietary project's with no fill), each count read from the roles by construction; under
 * Estimate, with no groups served, the words are as before.
 */
describe("the lineage's groups in the roles card's words under Predict (N3)", () => {
  const SERVED = (ROLES as unknown as { under: { prediction: { kinds: { dietary: { groups: { screen: PredictorGroupsServed } } } } } })
    .under.prediction.kinds.dietary.groups.screen;
  type PredictorGroupsServed = { fill: Record<string, { one: string; head: [string, string]; note: string }>; no_fill: Record<string, { one: string; head: [string, string]; note: string }> };
  const roles: Record<string, string> = {
    ...Object.fromEntries(Array.from({ length: 8 }, (_, i) => [`c${i}`, "covariate"])),
    ...Object.fromEntries(Array.from({ length: 5 }, (_, i) => [`n${i}`, "exposure"])),
    kcal: "energy",
  };
  const predictors = Object.keys(roles);
  const view = { ...VIEW, state: { ...VIEW.state, roles } } as ProjectView;
  const head = (role: string) => SERVED.no_fill[role]!.head.map((w) => w.toLowerCase());

  /** The card drawn, its lineage given a box to draw in (jsdom measures none), as Lineage.test does. */
  function draw(groups: ReturnType<typeof servedGroups>) {
    const observed: ResizeObserverCallback[] = [];
    vi.stubGlobal(
      "ResizeObserver",
      class {
        constructor(private cb: ResizeObserverCallback) {}
        observe() {
          observed.push(this.cb);
        }
        unobserve() {}
        disconnect() {}
      },
    );
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
    render(
      <QueryClientProvider client={qc}>
        <StageFocusProvider>
          <ColumnsCard view={view} data={{ ...data(predictors, null), predictorGroups: groups }} full={false} />
        </StageFocusProvider>
      </QueryClientProvider>,
    );
    act(() => {
      for (const cb of observed) cb([{ contentRect: { width: 900, height: 380 } } as ResizeObserverEntry], {} as ResizeObserver);
    });
  }

  it("names each group as the roles card does, never as a covariate or an exposure", () => {
    expect(predictors.length).toBeGreaterThan(12); // collapsed into one node per role
    draw(servedGroups({ ...SERVED, energy_bearing: [] }, roles));
    const text = document.body.textContent ?? "";
    expect(text).toContain(`8 ${head("covariate")[1]}`);
    expect(text).toContain(`5 ${head("exposure")[1]}`);
    expect(head("energy")[0]).toBe(head("energy")[1]); // "total energy", one word for one or many
    expect(text).toContain(`${head("energy")[0]}: 1 column`);
    expect(text).not.toMatch(/covariate column|exposure column|energy column/);
  });

  it("keeps Estimate's words where no groups are served", () => {
    draw(null);
    const text = document.body.textContent ?? "";
    expect(text).toContain("8 covariate columns");
    expect(text).toContain("5 exposure columns");
  });

  it("names the design's collapsed groups the same way", () => {
    const engine = predictorsLineage(predictors, roles); // the engine's collapse has the same group nodes
    const named = inGroupWords(engine, servedGroups({ ...SERVED, energy_bearing: [] }, roles));
    const labels = named.nodes.filter((n) => n.lane === "raw").map((n) => n.label);
    expect(labels).toEqual([`8 ${head("covariate")[1]}`, `5 ${head("exposure")[1]}`, `${head("energy")[0]}: 1 column`]);
    expect(inGroupWords(engine, null)).toBe(engine);
  });
});

describe("the Columns panel and the roles that wait (finding 1)", () => {
  it("counts the waiting columns proposed as predictors, never a flag or a column proposed as left out (W1-T)", () => {
    // The engine's open lines: bp_sys proposed in a predictor role, a flag and a column proposed as
    // left out, which no confirmation makes a predictor.
    expect(OPEN.map((l) => [l.items[0]!.column, l.items[0]!.value])).toEqual([
      ["bp_sys", "covariate"],
      ["imputed_waist", "flag"],
      ["cycle_begin_year", "excluded"],
      ["hdl", "covariate"],
    ]);
    const lone = LINES.filter((l) => l.key !== "confirm_roles:hdl");
    mount(logWith(lone), ["age", "bmi"]);
    expect(screen.getByTestId("columns-waiting")).toHaveTextContent(/^1 column waits for its own confirmation and is not a predictor yet\.$/);
  });

  it("says it of several columns in the plural, each column of a group's line counted", () => {
    // bp_sys on its own line, and the two lab values of one group's line (ruling R-g)
    const group = OPEN.find((l) => l.key === "confirm_roles:hdl")!;
    const n = 1 + group.items.filter((i) => i.kind === "role").length;
    mount(logWith(LINES), ["age", "bmi"]);
    expect(screen.getByTestId("columns-waiting")).toHaveTextContent(new RegExp(`^${n} columns wait for their own confirmation and are not predictors yet\\.$`));
  });

  it("asks the server nothing: a card drawn with no quest log says nothing of them (U9)", () => {
    mount(null, ["age", "bmi"]);
    expect(screen.queryByTestId("columns-waiting")).toBeNull();
    expect(fetch).not.toHaveBeenCalled();
  });

  it("says nothing of it where no role line is open", () => {
    mount(logWith(LINES.filter((l) => l.status !== "open")), ["age", "bmi"]);
    expect(screen.queryByTestId("columns-waiting")).toBeNull();
  });

  it("says nothing of it where the only open role lines are a flag's and an exclusion's", () => {
    mount(logWith(LINES.filter((l) => l.items[0]!.column !== "bp_sys" && l.key !== "confirm_roles:hdl")), ["age", "bmi"]);
    expect(screen.queryByTestId("columns-waiting")).toBeNull();
  });

  it("names the confirmations that wait, not the roles question, where no column is a predictor yet, saying yet once", () => {
    const lone = LINES.filter((l) => l.key !== "confirm_roles:hdl");
    mount(logWith([...lone, predictorLine("ldl")]), []);
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
