/**
 * The renderer contract (plans/W1_PART2_PLAN.md §2 "W1-E" acceptance; ruling 6): every (source, key)
 * the path fuzzer emits for the sample datasets under every goal (fixtures/quest-contract.json,
 * written by turbotab/core/tests/test_quest_renderer_contract.py) has a renderer in the card
 * dispatcher, and renders something, never a blank. A declaration counts as rendered only through
 * W1-F's DECLARATION_CARDS or the generic fallback (LineGeneric, GenericAsk's content in the line's
 * own words); the kinds that fall to it today are listed here, since W1-F's cards replace them.
 *
 * Each line renders on a real server's captured view (src/mocks/fixtures, the M3 captures), with
 * the stages it reads served from the capture, as the generic question's test does.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { afterEach, beforeEach, vi } from "vitest";
import { QUESTION_KEYS, type TeachingEntry } from "../../api/m1-types";
import { keys } from "../../api/queries";
import type { ProjectView, QuestLine } from "../../api/schema";
import { artifactAt, expandView, viewsOf, type M3Fixture } from "../../mocks/m3";
import { StageFocusProvider } from "../../state/focus";
import type { DeclarationCards, LineCardProps } from "../contracts";
import { CONTRACT, type ContractPair } from "../fixtures/contract";
import { CardFor } from "./CardFor";
import { DECLARATION_CARDS } from "./predict";
import { rendererOf, type Renderer } from "./renderers";
import { said } from "../words";

const FIXTURES = import.meta.glob<M3Fixture>("../../mocks/fixtures/m3-*.json", { eager: true, import: "default" });
const SHARED = Object.entries(FIXTURES).find(([p]) => p.endsWith("m3-shared.json"))![1] as unknown as {
  teaching: TeachingEntry[];
};
const JOURNEYS = Object.entries(FIXTURES)
  .filter(([p]) => !p.endsWith("m3-shared.json"))
  .map(([p, f]) => ({ name: /m3-(.+)\.json$/.exec(p)![1]!, f, views: viewsOf(f) }));
const PID = "quest-test";
const AT = "2026-10-17T00:00:00Z";

/** A captured view where the Router lists `key` as a step it asks; else the Predict journey's last. */
function viewFor(key: string): { view: ProjectView; f: M3Fixture } {
  for (const { f, views } of JOURNEYS) {
    for (let i = views.length - 1; i >= 0; i--) {
      const step = views[i]!.interview.find((s) => s.key === key);
      if (step && step.status !== "not_applicable") return { view: expandView(f, views[i]!, PID, AT), f };
    }
  }
  const j = JOURNEYS.find((x) => x.name === "nhanes-prediction")!;
  return { view: expandView(j.f, j.views[j.views.length - 1]!, PID, AT), f: j.f };
}

function lineOf(pair: Pick<ContractPair, "source" | "key">, label: QuestLine["label"] = "Decide"): QuestLine {
  const place = CONTRACT.questions[pair.key] ?? CONTRACT.declarations[pair.key];
  return {
    id: pair.key,
    key: pair.key,
    source: pair.source as QuestLine["source"],
    label,
    name: place?.name ?? `the finding ${pair.key}`,
    status: "open",
    counted: true,
    order: place?.order ?? 0,
    decision_id: null,
    reason: null,
    waiting_for: [],
    computing: [],
    reopened_by: null,
    would_change: null,
    changes_nothing: null,
    items: [],
    changed_since: null,
  };
}

function renderLine(line: QuestLine, stage: string, cards?: DeclarationCards) {
  const { view, f } = viewFor(line.key);
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  qc.setQueryData(keys.teaching(), SHARED.teaching);
  qc.setQueryData(keys.view(PID), view);
  qc.setQueryData(keys.quest(PID), CONTRACT.logs.results_reached.log);
  const columns = f.endpoints.columns;
  if (columns?.status === 200) qc.setQueryData(keys.columns(PID), columns.body);
  for (const [name, status] of Object.entries(view.stages)) {
    if (!status || status.status === "idle" || status.status === "blocked") continue;
    qc.setQueryData(keys.stage(PID, name), {
      stage: name,
      key: status.key,
      fresh: status.status === "fresh",
      status: status.status,
      artifact: artifactAt(f, name, status.key),
    });
  }
  return render(
    <QueryClientProvider client={qc}>
      <StageFocusProvider>
        <CardFor pid={PID} line={line} stage={stage} {...(cards ? { cards } : {})} />
      </StageFocusProvider>
    </QueryClientProvider>,
  );
}

// Whatever the capture does not hold stays pending: nothing here reaches a server.
beforeEach(() => {
  vi.stubGlobal("fetch", vi.fn(() => new Promise(() => {})));
});
afterEach(() => {
  vi.unstubAllGlobals();
});

const CASES = CONTRACT.pairs.flatMap((p) => p.stages.map((stage) => ({ ...p, stage })));
const ALLOWED: Record<string, readonly Renderer[]> = {
  question: ["ask", "results"],
  declaration: ["declaration", "generic"],
  finding: ["finding"],
  reading: ["reading"],
};

describe("every (source, key) the engine emits has a renderer", () => {
  it("walks both sample datasets under every goal", () => {
    const walked = new Set(CONTRACT.pairs.flatMap((p) => p.seen));
    for (const d of CONTRACT.datasets) for (const g of CONTRACT.goals) expect(walked.has(`${d}/${g}`), `${d}/${g}`).toBe(true);
    expect(CASES.length).toBeGreaterThan(40);
  });

  it.each(CASES)("$source $key in $stage", ({ source, key, stage }) => {
    const kind = rendererOf({ source: source as QuestLine["source"], key }, stage, DECLARATION_CARDS);
    expect(ALLOWED[source], `a source the shell does not know: ${source}`).toBeDefined();
    expect(ALLOWED[source]).toContain(kind);
    // A Router question is today's card, except the opening in Results, which is W1-G's.
    if (source === "question") expect(kind).toBe(stage === "results" ? "results" : "ask");
  });

  it("lists the declaration kinds that fall to the generic fallback today (W1-F's cards replace them)", () => {
    const generic = [
      ...new Set(
        CONTRACT.pairs
          .filter((p) => p.source === "declaration" && rendererOf({ source: p.source as QuestLine["source"], key: p.key }, p.stages[0]!, DECLARATION_CARDS) === "generic")
          .map((p) => p.key),
      ),
    ].sort();
    const declared = CONTRACT.pairs.filter((p) => p.source === "declaration").map((p) => p.key);
    // Every declaration is one or the other, never neither.
    expect(generic.length + declared.filter((k) => DECLARATION_CARDS[k]).length).toBe(declared.length);
    // On the base, DECLARATION_CARDS is W1-F's stub: every declaration falls to the generic fallback.
    if (Object.keys(DECLARATION_CARDS).length === 0) expect(generic).toEqual([...declared].sort());
  });
});

describe("the dispatcher never renders a blank", () => {
  it.each(CASES)("$source $key in $stage renders something", ({ source, key, stage, labels }) => {
    const line = lineOf({ source, key }, labels[0] as QuestLine["label"]);
    const { container } = renderLine(line, stage);
    expect(container.textContent!.trim().length, `${source} ${key}`).toBeGreaterThan(0);
    const kind = rendererOf(line, stage, DECLARATION_CARDS);
    if (kind === "ask") expect(screen.getByTestId(`ask-${key}`).textContent!.trim().length).toBeGreaterThan(0);
    if (kind === "generic") expect(screen.getByTestId(`generic-${key}`)).toBeInTheDocument();
    if (kind === "finding") expect(screen.getByTestId(`finding-card-${key}`)).toBeInTheDocument();
    if (kind === "results") expect(screen.getByTestId("results-cards")).toHaveAccessibleName(/opening the seal/i);
  });
});

describe("the dispatcher's fallbacks, both ways", () => {
  const Probe = ({ line }: LineCardProps) => <p data-testid="probe">{line.key}</p>;

  it("draws a declaration with its own card through that card, and without one through the generic fallback", () => {
    const line = lineOf({ source: "declaration", key: "set_selection" });
    const withCard = renderLine(line, "models", { set_selection: Probe });
    expect(screen.getByTestId("probe")).toHaveTextContent("set_selection");
    expect(screen.queryByTestId("generic-set_selection")).toBeNull();
    withCard.unmount();
    renderLine(line, "models", {});
    expect(screen.queryByTestId("probe")).toBeNull();
    expect(screen.getByTestId("generic-set_selection")).toHaveTextContent("The predictor-selection question");
  });

  it("names a fallback card by the line's own name, never by its decision kind (FOUNDATION §2)", () => {
    const unknown: QuestLine = { ...lineOf({ source: "question", key: "lens" }), source: "exhibit" as QuestLine["source"], key: "table_one", name: "the first table" };
    const lines = [
      ...Object.keys(CONTRACT.declarations).flatMap((key) =>
        (["open", "answered"] as const).map((status) => ({ ...lineOf({ source: "declaration", key }), status })),
      ),
      unknown,
    ];
    expect(lines.length).toBeGreaterThan(20);
    for (const line of lines) {
      const stage = line.source === "declaration" ? CONTRACT.declarations[line.key]!.stage : "results";
      const { container, unmount } = renderLine(line, stage, {});
      expect(rendererOf(line, stage, {})).toBe("generic");
      const text = container.textContent!.toLowerCase();
      expect(text, `${line.key} (${line.status})`).not.toContain(line.key.replace(/_/g, " "));
      expect(text, `${line.key} (${line.status})`).not.toContain(line.key);
      expect(screen.getByTestId(`generic-${line.key}`)).toHaveTextContent(said(line.name));
      unmount();
    }
  });

  it("draws a Router question as today's card, and a line of a source it does not know by its own name", () => {
    expect(QUESTION_KEYS).toContain("lens");
    const known = renderLine(lineOf({ source: "question", key: "lens" }), "data");
    expect(screen.getByTestId("question-lens")).toBeInTheDocument();
    expect(screen.queryByTestId("generic-lens")).toBeNull();
    known.unmount();
    const unknown: QuestLine = { ...lineOf({ source: "question", key: "lens" }), source: "exhibit" as QuestLine["source"], key: "table_one", name: "the first table" };
    expect(rendererOf(unknown, "results", DECLARATION_CARDS)).toBe("generic");
    renderLine(unknown, "results");
    expect(screen.getByTestId("generic-table_one")).toHaveTextContent("The first table");
  });

  it("draws the ledger's ask line and a reading line as rows of their items", () => {
    const ask: QuestLine = {
      ...lineOf({ source: "question", key: "ask:models" }),
      name: "Tell me about this column: kcal",
      items: [
        {
          kind: "unit",
          column: "kcal",
          value: "kcal",
          words: "kcal is energy in kilocalories",
          evidence: "Its median is 1,980.",
          options: [{ label: "It is in kilojoules", decision: { kind: "set_column_unit", column: "kcal", unit: "kJ" } }],
        },
      ],
    };
    expect(rendererOf(ask, "models", DECLARATION_CARDS)).toBe("reading");
    renderLine(ask, "models");
    expect(screen.getByTestId("reading-ask:models")).toHaveTextContent("kcal is energy in kilocalories");
    expect(screen.getByRole("button", { name: "It is in kilojoules" })).toBeEnabled();
  });
});
