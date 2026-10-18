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
import { act, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, vi } from "vitest";
import { api } from "../../api/client";
import type { CohortArtifact, PreviewResult } from "../../api/m1-stage-types";
import type { Decision, ProjectView, QuestLine, QuestLog } from "../../api/schema";
import { expandView, viewsOf, type M3Fixture } from "../../mocks/m3";
import ROLES from "../../quest/cards/roles/fixture.json";
import { CONTRACT } from "../../quest/fixtures/contract";
import { StageFocusProvider } from "../../state/focus";
import { servedGroups, waitingPredictorColumns } from "../../quest/cards/roles/roles";
import { keys } from "../../api/queries";
import { ColumnsCard, inGroupWords, predictorsLineage, viewInGroupWords, type LiveData } from "./LiveScenes";
import { Stage } from "./Stage";

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

  /** The card drawn under `purpose` (the journey's own, Predict, unless named), its lineage given a
   *  box to draw in (jsdom measures none), as Lineage.test does. */
  function draw(groups: ReturnType<typeof servedGroups>, purpose?: string) {
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
          <ColumnsCard
            view={purpose ? ({ ...view, state: { ...view.state, purpose } } as ProjectView) : view}
            data={{ ...data(predictors, null), predictorGroups: groups }}
            full={false}
          />
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

  it("keeps Estimate's words under Estimate, where no groups are served", () => {
    expect(view.state.purpose).toBe("prediction"); // the journey's goal, set aside here
    draw(null, "inference");
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

  /**
   * W1-W round 1, P1: a folded group's count is its lane's columns (core/models/lineage.py
   * `collapse`: `count = len(ids)` per lane), so in the model matrix it counts matrix columns, not
   * characteristics. Built as `collapse` folds a wide table (above COLLAPSE_AT = 40 raw columns, the
   * touched columns past SHOWN = 12 by role and signature): 14 imputed nutrients (12 kept, 2
   * folded), 3 categorical characteristics one-hot to 4 columns each (3 raw, 3 adjusted, 3 × 4 = 12
   * matrix columns) and 30 kept characteristics; the nodes and their ids are the ones `collapse`
   * returns for that table (checked by running it). Under Predict the raw lane names the
   * characteristics, each lane after it the group's columns; under Estimate the engine's words
   * stand, which count columns in every lane.
   */
  it("names a folded group past the raw lane by its columns, never as that many characteristics (P1)", () => {
    const group = (lane: "raw" | "adjusted" | "matrix", role: "exposure" | "covariate", g: number, count: number) => ({
      id: `${lane}:group:${role}:${g}`,
      column: null,
      lane,
      role,
      label: `${count} ${role} column${count === 1 ? "" : "s"}`,
      formula: null,
      group: role,
      count,
    });
    const nodes = [
      group("raw", "exposure", 0, 2), group("adjusted", "exposure", 0, 2), group("matrix", "exposure", 0, 2),
      group("raw", "covariate", 1, 3), group("adjusted", "covariate", 1, 3), group("matrix", "covariate", 1, 3 * 4),
      group("raw", "covariate", 2, 30), group("adjusted", "covariate", 2, 30), group("matrix", "covariate", 2, 30),
    ];
    const engine = { nodes, links: [], collapsed: true } as unknown as Parameters<typeof inGroupWords>[0];
    const named = inGroupWords(engine, servedGroups({ ...SERVED, energy_bearing: [] }, roles));
    const label = (id: string) => named.nodes.find((n) => n.id === id)!.label;
    const [, many] = head("covariate");
    expect(label("raw:group:covariate:1")).toBe(`3 ${many}`);
    expect(label("adjusted:group:covariate:1")).toBe(`${many}: 3 columns`);
    expect(label("matrix:group:covariate:1")).toBe(`${many}: 12 columns`);
    expect(label("matrix:group:covariate:2")).toBe(`${many}: 30 columns`);
    expect(label("matrix:group:exposure:0")).toBe(`${head("exposure")[1]}: 2 columns`);
    // No lane past the raw one counts its columns as characteristics or nutrients.
    for (const n of named.nodes.filter((x) => x.lane !== "raw"))
      expect(n.label).not.toMatch(new RegExp(`^\\d+ (${many}|${head("exposure")[1]})$`));
    // Estimate: the engine's words, which count columns in every lane, unchanged.
    expect(inGroupWords(engine, null)).toBe(engine);
    expect(engine.nodes.find((n) => n.id === "matrix:group:covariate:1")!.label).toBe("12 covariate columns");
  });

  /**
   * W1-W round 1, P2: the groups reached the Columns card only through the quest shell's Tapestry,
   * so the Classic screen (`<Stage pid view focus onFocus/>`, no groups) said "8 covariate columns"
   * under Predict beside a roles card in the served group words. The stage reads them itself, from
   * the goal and the roles stage it shares with the roles card. Drawn here as the Classic screen
   * mounts it, the roles stage's result and the cohort's put in the query cache by construction: the
   * served groups under Predict; while the roles stage's result is not at hand, the predictors' own
   * word, never Estimate's role words; under Estimate the engine's words as before.
   */
  describe("through the stage, as every screen mounts it (P2)", () => {
    function stageWith(purpose: string, rolesResult: boolean) {
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
      const fresh = { status: "fresh", key: "k" };
      const v = {
        ...view,
        state: { ...view.state, purpose },
        decisions: [],
        interview: [],
        stages: { cohort: fresh, roles: fresh },
      } as unknown as ProjectView;
      const qc = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
      qc.setQueryData(keys.stage(PID, "cohort"), { key: "k", fresh: true, artifact: { predictors, steps: [] } });
      if (rolesResult)
        qc.setQueryData(keys.stage(PID, "roles"), {
          key: "k",
          fresh: true,
          artifact: { columns: [], repeats: [], categorical: [], predictor_groups: { ...SERVED, energy_bearing: [] } },
        });
      render(
        <QueryClientProvider client={qc}>
          <StageFocusProvider>
            <Stage pid={PID} view={v} focus={{ kind: "live" }} onFocus={() => {}} />
          </StageFocusProvider>
        </QueryClientProvider>,
      );
      act(() => {
        for (const cb of observed) cb([{ contentRect: { width: 900, height: 380 } } as ResizeObserverEntry], {} as ResizeObserver);
      });
      return document.querySelector('[data-card="lineage"]')?.textContent ?? "";
    }

    it("names the groups as the roles card does under Predict, with no groups passed", () => {
      const text = stageWith("prediction", true);
      expect(text).toContain(`8 ${head("covariate")[1]}`);
      expect(text).toContain(`5 ${head("exposure")[1]}`);
      expect(text).not.toMatch(/covariate|exposure/);
    });

    it("never says Estimate's role words under Predict while the roles stage's result is not at hand", () => {
      const text = stageWith("prediction", false);
      expect(text).toContain("8 predictors");
      expect(text).toContain("5 predictors");
      expect(text).not.toMatch(/covariate|exposure/);
    });

    it("keeps the engine's words under Estimate", () => {
      const text = stageWith("inference", true);
      expect(text).toContain("8 covariate columns");
      expect(text).toContain("5 exposure columns");
    });
  });

  /**
   * W1-W round 2, P2: a roles option focused on the stage draws the server's preview of it, whose
   * lineage (core/row_previews.py `_roles_lineage`, through core/consequences.py `lineage_of`) names
   * its folded groups "{count} {group} columns": under Predict the real server said "12 covariate
   * columns" and "7 exposure columns" beside the roles card's groups. The preview here is shaped as
   * that server's was (scratchpad roles-preview.json of the W1-W verifier): the roles above with `c0`
   * focused on its own, then left out of the models; a group node per role and lane, labelled as
   * `lineage_of` labels it. Under Predict both sides of the preview name the groups as the live
   * Columns card does; under Estimate the server's words stand.
   */
  describe("a preview's lineage on the stage (P2)", () => {
    const group = (lane: "raw" | "matrix", role: "covariate" | "exposure" | "energy", count: number) => ({
      id: `${lane}:group:${role}:kept`,
      column: null,
      lane,
      role,
      label: `${count} ${role} ${count === 1 ? "column" : "columns"}`,
      formula: null,
      group: role,
      count,
    });
    const own = (lane: "raw" | "matrix", role: string) => ({
      id: `${lane}:c0`, column: "c0", lane, role, label: "c0", formula: null, group: null, count: 1,
    });
    const groups = (lane: "raw" | "matrix") => [group(lane, "covariate", 7), group(lane, "exposure", 5), group(lane, "energy", 1)];
    const before = { nodes: [...groups("raw"), own("raw", "covariate"), ...groups("matrix"), own("matrix", "covariate")], links: [], collapsed: true };
    const after = { nodes: [...groups("raw"), own("raw", "excluded"), ...groups("matrix")], links: [], collapsed: true };
    const LEFT_OUT = { kind: "set_roles", roles: { ...roles, c0: "excluded" } } as unknown as Decision;
    const PREVIEW = {
      kind: "set_roles",
      views: [{ kind: "lineage", title: "Which columns enter the model", caption: "`c0` leaves it.", emphasis: ["c0"], coach: [], before, after, story: [] }],
      basis: "The roles as recorded.",
      note: null,
      caution: null,
    } as unknown as PreviewResult;

    afterEach(() => {
      vi.restoreAllMocks();
    });

    /** The lineage the stage draws at its first state ("Your data now") and once it has played to
     *  the choice ("With this choice"), under `purpose`. */
    async function previewed(purpose: string): Promise<[string, string]> {
      vi.spyOn(api, "preview").mockResolvedValue(PREVIEW);
      // The lineage measures its box (jsdom measures none): each one observed is given one.
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
      const measure = () =>
        act(() => {
          for (const cb of observed) cb([{ contentRect: { width: 900, height: 380 } } as ResizeObserverEntry], {} as ResizeObserver);
        });
      const fresh = { status: "fresh", key: "k" };
      const v = { ...view, state: { ...view.state, purpose }, decisions: [], interview: [], stages: { cohort: fresh, roles: fresh } } as unknown as ProjectView;
      const qc = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
      qc.setQueryData(keys.stage(PID, "cohort"), { key: "k", fresh: true, artifact: { predictors, steps: [] } });
      qc.setQueryData(keys.stage(PID, "roles"), {
        key: "k",
        fresh: true,
        artifact: { columns: [], repeats: [], categorical: [], predictor_groups: { ...SERVED, energy_bearing: [] } },
      });
      render(
        <QueryClientProvider client={qc}>
          <StageFocusProvider>
            <Stage pid={PID} view={v} focus={{ kind: "option", decision: LEFT_OUT, label: "Leave c0 out" }} onFocus={() => {}} />
          </StageFocusProvider>
        </QueryClientProvider>,
      );
      await waitFor(() => expect(screen.getByTestId("stage")).toHaveAttribute("data-scene", "preview"));
      const drawn = () => document.querySelector("[data-state]");
      await waitFor(() => expect(drawn()).toHaveAttribute("data-state", "0"));
      measure();
      const now = drawn()!.textContent ?? "";
      await waitFor(() => expect(drawn()).not.toHaveAttribute("data-state", "0"));
      measure();
      return [now, drawn()!.textContent ?? ""];
    }

    it("names the groups as the roles card does under Predict, on both sides", async () => {
      const [now, chosen] = await previewed("prediction");
      for (const text of [now, chosen]) {
        expect(text).toContain(`7 ${head("covariate")[1]}`);
        expect(text).toContain(`5 ${head("exposure")[1]}`);
        expect(text).toContain(`${head("covariate")[1]}: 7 columns`); // the model matrix's lane
        expect(text).not.toMatch(/covariate|exposure|energy column/);
      }
      // A step between the two sides (a preview's story frame) is named the same way.
      const served = servedGroups({ ...SERVED, energy_bearing: [] }, roles);
      const framed = { ...PREVIEW.views[0]!, story: [{ label: "a step", lineage: before }] } as typeof PREVIEW.views[number];
      const named = viewInGroupWords(framed, served, true);
      const frame = named.kind === "lineage" ? named.story[0]!.lineage.nodes.map((n) => n.label) : [];
      expect(frame).toContain(`7 ${head("covariate")[1]}`);
      expect(frame.join(" ")).not.toMatch(/covariate|exposure|energy column/);
      expect(viewInGroupWords(framed, served, false)).toBe(framed);
    });

    it("keeps the server's words under Estimate", async () => {
      const [now, chosen] = await previewed("inference");
      for (const text of [now, chosen]) {
        expect(text).toContain("7 covariate columns");
        expect(text).toContain("5 exposure columns");
        expect(text).toContain("1 energy column");
      }
    });
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
