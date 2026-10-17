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
import type { ForTheRecord, ProjectView, QuestLine, QuestLog, QuestStage } from "../api/schema";
import { expandView, viewsOf, type M3Fixture } from "../mocks/m3";
import { StageFocusProvider } from "../state/focus";
import { CONTRACT } from "./fixtures/contract";
import { QuestScreen } from "./QuestScreen";
import { StageCard, type OpenRequest } from "./StageCard";

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
    // The head, For the record and the foot stay the shell's; while the opening is open, the foot is
    // where ResultsCards draws it (FootSlot), and the shell draws no Continue beside it (P5).
    expect(within(card).getByRole("heading", { level: 2 })).toHaveTextContent(results.name);
    expect(within(card).getByTestId("for-the-record")).toHaveTextContent("For the record · 1");
    expect(within(card).getByTestId("primary-slot")).toBeInTheDocument();
    expect(within(card).queryByRole("button", { name: "Continue" })).toBeNull();
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

  it("keeps the shell's Continue once the opening is no longer open", () => {
    const opened = logWith(REACHED, (l) => ({ ...l, status: "answered", decision_id: `r-${l.key}` }));
    const { card } = shell(opened);
    expect(within(card).queryByTestId("primary-slot")).toBeNull();
    expect(within(card).getByRole("button", { name: "Continue" })).toBeEnabled();
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

// ── the design review's repairs (W1 part 2) ──────────────────────────────────
// The engine's log (the contract's, under Predict) with lines' statuses set by construction for
// each moment; expected words are read off the log or the captured view, never off the card.

/** The log with each stage's lines `set` by the stage's key, and its stages patched. */
function patched(base: QuestLog, set: Record<string, (l: QuestLine) => QuestLine>, stages: Record<string, Partial<QuestStage>> = {}): QuestLog {
  return {
    ...base,
    stages: base.stages.map((s) => ({ ...s, ...stages[s.key], lines: s.lines.map(set[s.key] ?? ((l) => l)) })),
  };
}
const stageIn = (log: QuestLog, key: string) => log.stages.find((s) => s.key === key)!;
const lineEl = (card: HTMLElement, key: string) => within(card).getAllByTestId(`line-${key}`)[0]!;

function cardOf(log: QuestLog, key: string, opts: { request?: OpenRequest | null; answers?: Map<string, string>; open?: (stage: string, line?: string) => void } = {}) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  qc.setQueryData(keys.quest(PID), log);
  qc.setQueryData(keys.view(PID), VIEW);
  qc.setQueryData(keys.record(PID), RECORD);
  const props = {
    pid: PID,
    log,
    stage: stageIn(log, key),
    answers: opts.answers ?? new Map<string, string>(),
    fit: { kind: "none" } as const,
    onOpenStage: opts.open ?? (() => {}),
    arrival: null,
    onOpenManuscript: () => {},
  };
  const ui = (request: OpenRequest | null) => (
    <QueryClientProvider client={qc}>
      <StageFocusProvider>
        <StageCard {...props} request={request} />
      </StageFocusProvider>
    </QueryClientProvider>
  );
  const r = render(ui(opts.request ?? null));
  return { card: screen.getByTestId(`card-${key}`), ask: (request: OpenRequest) => r.rerender(ui(request)) };
}

describe("a stage opened by a request (P3, P4)", () => {
  const ROLES = { key: "roles", stage: "data", name: CONTRACT.questions.roles!.name };
  // Your data's roles open; Who's in's counted lines wait on them.
  const log = patched(
    REACHED,
    {
      data: (l) => (l.key === "roles" ? { ...l, status: "open", decision_id: null } : l),
      whos_in: (l) => (l.label === "Decide" && l.counted ? { ...l, status: "waiting", decision_id: null, waiting_for: [ROLES] } : l),
    },
    { data: { progress: { answered: 1, required: 2, complete: false } } },
  );
  let scroll: ReturnType<typeof vi.fn>;
  beforeEach(() => {
    // Every line far below the window, as on a long stage: bringing one into view scrolls.
    vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockReturnValue({ top: 2165, bottom: 2400, left: 0, right: 400, width: 400, height: 235, x: 0, y: 2165, toJSON: () => ({}) });
    scroll = vi.fn();
    Element.prototype.scrollIntoView = scroll as unknown as Element["scrollIntoView"];
  });
  afterEach(() => {
    vi.restoreAllMocks();
    delete (Element.prototype as { scrollIntoView?: unknown }).scrollIntoView;
  });
  const render_ = () => {
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
    qc.setQueryData(keys.quest(PID), log);
    qc.setQueryData(keys.view(PID), VIEW);
    qc.setQueryData(keys.record(PID), RECORD);
    render(
      <QueryClientProvider client={qc}>
        <QuestScreen pid={PID} stream="open" />
      </QueryClientProvider>,
    );
  };
  const rolesLine = () => screen.getByTestId("card-data").querySelector<HTMLElement>('[data-line="question:roles"]')!;

  it("lands without moving the screen or the focus, where nothing asked for a line", () => {
    render_();
    expect(screen.getByTestId("card-data")).toBeInTheDocument();
    expect(scroll).not.toHaveBeenCalled();
    expect(document.activeElement).toBe(document.body);
  });

  it("brings the line a Waiting-for link names into view, expanded, and focuses it", () => {
    render_();
    fireEvent.click(screen.getByTestId("stage-whos_in"));
    fireEvent.click(within(screen.getAllByTestId(/^line-.*-waiting$/)[0]!).getByRole("button", { name: ROLES.name }));
    expect(rolesLine()).toHaveAttribute("data-expanded");
    expect(document.activeElement).toBe(rolesLine());
    expect(scroll).toHaveBeenLastCalledWith({ block: "start" });
    expect(scroll.mock.contexts.at(-1)).toBe(rolesLine());
  });

  it("goes there from the foot of a stage that waits on it, by its button or its link (P4)", () => {
    render_();
    fireEvent.click(screen.getByTestId("stage-whos_in"));
    const foot = screen.getByTestId("primary");
    expect(foot).toHaveTextContent("Go to Your data");
    expect(foot).toBeEnabled();
    expect(screen.getByTestId("primary-hint")).toHaveTextContent(`Waiting for ${ROLES.name}, in Your data.`);
    fireEvent.click(screen.getByTestId("primary-hint-link"));
    expect(document.activeElement).toBe(rolesLine());
    fireEvent.click(screen.getByTestId("stage-whos_in"));
    fireEvent.click(screen.getByTestId("primary"));
    expect(screen.getByTestId("card-data")).toBeInTheDocument();
  });
});

describe("a request in the stage already shown (P9)", () => {
  // Who's in as the engine serves it: the implausible-intake noticing open, its questions answered.
  const whos = stageIn(REACHED, "whos_in");
  const expanded = (card: HTMLElement) => card.querySelector("[data-expanded]")?.getAttribute("data-line");

  it("expands the line it names, and the next request's line, without leaving the stage", () => {
    expect(whos.lines.find((l) => l.status === "open" && l.counted)?.source).toBe("finding");
    const { card, ask } = cardOf(REACHED, "whos_in");
    expect(expanded(card)).toBe("finding:pack::dietary::implausible_intake#1");
    ask({ line: "exclusions", n: 1 });
    expect(expanded(card)).toBe("question:exclusions");
    ask({ line: "missing", n: 2 });
    expect(expanded(card)).toBe("question:missing");
    // No line named: the stage's own open line.
    ask({ line: null, n: 3 });
    expect(expanded(card)).toBe("finding:pack::dietary::implausible_intake#1");
  });
});

describe("a line no card can answer yet (P8)", () => {
  // Your data as the engine serves it: the join and the data dictionary open and uncounted, with no
  // card that records them; the lens and the roles answered.
  const data = stageIn(REACHED, "data");

  it("is listed after the lines that can be answered, as not available yet, with no Answer, and never opened", () => {
    expect(data.lines.filter((l) => l.status === "open" && l.label === "Decide").map((l) => l.key)).toEqual(["join_files", "import_codebook"]);
    const { card } = cardOf(REACHED, "data");
    for (const key of ["join_files", "import_codebook"]) {
      const line = lineEl(card, key);
      expect(within(line).queryByRole("button")).toBeNull();
      expect(line).toHaveTextContent("Not available yet");
      expect(line).not.toHaveAttribute("data-expanded");
    }
    const order = [...within(card).getByTestId("group-Decide").querySelectorAll("[data-line]")].map((x) => x.getAttribute("data-line"));
    expect(order).toEqual(["question:lens", "question:roles", "declaration:join_files", "declaration:import_codebook"]);
    expect(card.querySelector("[data-expanded]")).toBeNull();
  });

  it("leaves a line a card can answer its Change", () => {
    const { card } = cardOf(REACHED, "data");
    expect(within(lineEl(card, "lens")).getByRole("button", { name: "Change" })).toBeInTheDocument();
    expect(lineEl(card, "lens")).not.toHaveTextContent("Not available yet");
  });
});

describe("an answered noticing (P12)", () => {
  const KEY = "pack::dietary::implausible_intake#1";
  const ANSWER = "1,917 rows were excluded as implausible intakes.";
  const log = patched(REACHED, { whos_in: (l) => (l.key === KEY ? { ...l, status: "answered", decision_id: "r-imp" } : l) });
  const name = stageIn(REACHED, "whos_in").lines.find((l) => l.key === KEY)!.name;

  it("reads as its answer, and its open-tense name is gone", () => {
    const { card } = cardOf(log, "whos_in", { answers: new Map([["r-imp", ANSWER]]) });
    expect(lineEl(card, KEY)).toHaveTextContent(ANSWER);
    expect(lineEl(card, KEY)).not.toHaveTextContent(name);
  });

  it("keeps a question's name above its answer", () => {
    const exclusions = stageIn(REACHED, "whos_in").lines.find((l) => l.key === "exclusions")!;
    const { card } = cardOf(log, "whos_in", { answers: new Map([[exclusions.decision_id!, ANSWER]]) });
    expect(lineEl(card, "exclusions")).toHaveTextContent(exclusions.name.charAt(0).toUpperCase() + exclusions.name.slice(1));
    expect(lineEl(card, "exclusions")).toHaveTextContent(ANSWER);
  });
});

describe("First look (P14)", () => {
  it("lists each noticing with the stage that decides it, each a link there", () => {
    const open = vi.fn();
    const { card } = cardOf(REACHED, "first_look", { open });
    const findings = REACHED.stages.flatMap((s) => s.lines.filter((l) => l.source === "finding").map((l) => ({ s, l })));
    expect(findings.length).toBeGreaterThan(2);
    for (const { s, l } of findings) {
      const item = within(card).getByTestId(`noticed-${l.key}`);
      const where = l.label === "For the record" ? "For the record, in" : l.status === "answered" ? "Settled in" : "Decided in";
      expect(item).toHaveTextContent(`${where} ${s.name}`);
      fireEvent.click(within(item).getByRole("button", { name: s.name }));
      expect(open).toHaveBeenLastCalledWith(s.key, l.key);
    }
    expect(within(card).getByTestId("nothing-here")).toHaveTextContent("Nothing to decide here.");
  });
});

describe("a default the engine states no reason for (P10)", () => {
  it("is stated from the engine's value: the split's scheme", () => {
    const models = stageIn(REACHED, "models");
    expect(models.lines.find((l) => l.key === "set_validation")).toMatchObject({ label: "Confirm", status: "set_for_you", reason: null });
    const split = VIEW.state.split!;
    expect(split.validation).toBe("kfold");
    expect(split.nested_cv).toBe(false);
    const { card } = cardOf(REACHED, "models");
    expect(within(card).getByTestId("line-set_validation-reason")).toHaveTextContent(`${split.folds}-fold cross-validation.`);
  });

  it("leaves the engine's own reason where it serves one", () => {
    const form = stageIn(REACHED, "models").lines.find((l) => l.key === "form")!;
    expect(form.reason).toBeTruthy();
    const { card } = cardOf(REACHED, "models");
    expect(within(card).getByTestId("line-form-reason")).toHaveTextContent(form.reason!.replaceAll("`", ""));
  });
});

describe("Who's in's draw under Predict (P15)", () => {
  const SWEEP = { lines: 1, answered: false, id: "other:confirm-sweep:whos_in", heading: "Here is the other choice set for you", action: "Confirm it", confirmed_by: null, changed: [] };
  const at = (answered: boolean) =>
    patched(
      REACHED,
      {
        whos_in: (l) =>
          l.key === "grain"
            ? { ...l, label: "Confirm", counted: false, status: answered ? "answered" : "set_for_you" }
            : l.key === "split"
              ? { ...l, status: "open", decision_id: null }
              : l.label === "Decide"
                ? { ...l, status: "answered", decision_id: `r-${l.key}` }
                : l,
      },
      { whos_in: { sweep: { ...SWEEP, answered } } },
    );

  it("waits for the sweep: not opened, not offered, saying why, with the sweep at the foot", () => {
    const { card } = cardOf(at(false), "whos_in");
    const split = lineEl(card, "split");
    expect(split).not.toHaveAttribute("data-expanded");
    expect(within(split).queryByRole("button")).toBeNull();
    expect(within(card).getByTestId("line-split-held")).toBeInTheDocument();
    expect(within(card).getByTestId("primary")).toHaveTextContent(SWEEP.action);
  });

  it("is the open line once the sweep is confirmed", () => {
    const { card } = cardOf(at(true), "whos_in");
    expect(lineEl(card, "split")).toHaveAttribute("data-expanded");
    expect(within(card).queryByTestId("line-split-held")).toBeNull();
  });
});
