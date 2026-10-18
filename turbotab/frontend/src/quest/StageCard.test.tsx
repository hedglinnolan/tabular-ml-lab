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

function cardOf(
  log: QuestLog,
  key: string,
  opts: { request?: OpenRequest | null; answers?: Map<string, string>; open?: (stage: string, line?: string) => void; seed?: (qc: QueryClient) => void } = {},
) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  qc.setQueryData(keys.quest(PID), log);
  qc.setQueryData(keys.view(PID), VIEW);
  qc.setQueryData(keys.record(PID), RECORD);
  opts.seed?.(qc);
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

/**
 * Findings 6 and 7 through the shell: Who's in as the engine serves it under Predict (its questions
 * answered, one noticing open) holds Continue no longer; the foot says the noticing waits for the
 * triage in Results, and Continue opens the next open line a card can answer, Your question's
 * intended use (open in the engine's log), brought into view and expanded.
 */
describe("Continue past an open noticing, to the next open line (findings 6, 7)", () => {
  beforeEach(() => {
    vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockReturnValue({ top: 2165, bottom: 2400, left: 0, right: 400, width: 400, height: 235, x: 0, y: 2165, toJSON: () => ({}) });
    Element.prototype.scrollIntoView = vi.fn() as unknown as Element["scrollIntoView"];
  });
  afterEach(() => {
    vi.restoreAllMocks();
    delete (Element.prototype as { scrollIntoView?: unknown }).scrollIntoView;
  });

  it("is enabled beside the noticing, says it waits for the triage in Results, and opens the intended use", () => {
    const whos = stageIn(REACHED, "whos_in");
    expect(whos.lines.filter((l) => l.status === "open" && l.counted).map((l) => l.source)).toEqual(["finding"]);
    const intended = stageIn(REACHED, "question").lines.find((l) => l.key === "set_intended_use")!;
    expect(intended.status).toBe("open");
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
    qc.setQueryData(keys.quest(PID), REACHED);
    qc.setQueryData(keys.view(PID), VIEW);
    qc.setQueryData(keys.record(PID), RECORD);
    render(
      <QueryClientProvider client={qc}>
        <QuestScreen pid={PID} stream="open" />
      </QueryClientProvider>,
    );
    fireEvent.click(screen.getByTestId("stage-whos_in"));
    const foot = screen.getByTestId("primary");
    expect(foot).toHaveTextContent("Continue");
    expect(foot).toBeEnabled();
    expect(screen.getByTestId("primary-note")).toHaveTextContent(`1 noticing waits for the triage in ${stageIn(REACHED, "results").name}.`);
    expect(screen.getByTestId("primary-hint")).toHaveTextContent(`Next: ${intended.name}, in ${stageIn(REACHED, "question").name}.`);
    fireEvent.click(foot);
    const card = screen.getByTestId("card-question");
    const line = card.querySelector<HTMLElement>('[data-line="declaration:set_intended_use"]')!;
    expect(line).toHaveAttribute("data-expanded");
    expect(document.activeElement).toBe(line);
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

  it("is said, with every other such line that counts toward nothing, in one quiet line after the rest, never opened (finding 15)", () => {
    const unavailable = data.lines.filter((l) => l.status === "open" && l.label === "Decide");
    expect(unavailable.map((l) => l.key)).toEqual(["join_files", "import_codebook"]);
    expect(unavailable.every((l) => !l.counted)).toBe(true);
    const { card } = cardOf(REACHED, "data");
    for (const key of ["join_files", "import_codebook"]) expect(within(card).queryByTestId(`line-${key}`)).toBeNull();
    const quiet = within(card).getByTestId("lines-unavailable");
    expect(quiet).toHaveTextContent(`Not available in this version yet: ${unavailable[0]!.name} and ${unavailable[1]!.name}.`);
    expect(within(quiet).queryByRole("button")).toBeNull();
    const group = within(card).getByTestId("group-Decide");
    const order = [...group.querySelectorAll("[data-line]")].map((x) => x.getAttribute("data-line"));
    expect(order).toEqual(["question:lens", "question:roles"]);
    expect(group.querySelector("ul")!.lastElementChild).toBe(quiet);
    expect(card.querySelector("[data-expanded]")).toBeNull();
  });

  it("keeps a line of its own where it counts, said as not available yet, with no Answer", () => {
    const log = patched(REACHED, { data: (l) => (l.key === "join_files" ? { ...l, counted: true } : l) });
    const { card } = cardOf(log, "data");
    const line = lineEl(card, "join_files");
    expect(within(line).queryByRole("button")).toBeNull();
    expect(line).toHaveTextContent("Not available yet");
    expect(line).not.toHaveAttribute("data-expanded");
    expect(within(card).getByTestId("lines-unavailable")).not.toHaveTextContent(stageIn(log, "data").lines.find((l) => l.key === "join_files")!.name);
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

/**
 * A noticing settled by its question's answer (P2): the engine gives it that question's record, so
 * reading as its answer would print the question's sentence twice and drop what the noticing was
 * about. It names its columns (the findings artifact's) and the question instead, and the answer
 * is printed once, on the question's line. The noticing and its columns are set by construction.
 */
describe("a noticing settled by its question's answer (P2)", () => {
  const KEY = "pack::dietary::implausible_intake#1";
  const ANSWER = "1,917 rows were excluded as implausible intakes.";
  const exclusions = stageIn(REACHED, "whos_in").lines.find((l) => l.key === "exclusions")!;
  const log = patched(REACHED, { whos_in: (l) => (l.key === KEY ? { ...l, status: "answered", decision_id: exclusions.decision_id } : l) });
  const name = stageIn(REACHED, "whos_in").lines.find((l) => l.key === KEY)!.name;
  const SETTLED = `Kcal: settled by ${exclusions.name}`;
  const count = (el: HTMLElement, text: string) => (el.textContent ?? "").split(text).length - 1;
  const findings = { findings: [{ id: KEY, affected_columns: ["kcal"], title: "The days in `kcal` are not settled." }] };
  const seeded = (findingsServed: boolean) => {
    const view = { ...VIEW, stages: { ...VIEW.stages, findings: { ...VIEW.stages.findings!, status: "fresh" as const } } };
    return (qc: QueryClient) => {
      qc.setQueryData(keys.view(PID), view);
      if (findingsServed) qc.setQueryData(keys.stage(PID, "findings"), { stage: "findings", key: "k", fresh: true, status: "fresh", artifact: findings });
    };
  };

  it("names its columns and the question, and the answer is printed once, on the question's line", () => {
    const { card } = cardOf(log, "whos_in", { answers: new Map([[exclusions.decision_id!, ANSWER]]), seed: seeded(true) });
    expect(lineEl(card, KEY)).toHaveTextContent(SETTLED);
    expect(lineEl(card, KEY)).not.toHaveTextContent(ANSWER);
    expect(lineEl(card, KEY)).not.toHaveTextContent(name);
    expect(lineEl(card, "exclusions")).toHaveTextContent(ANSWER);
    expect(count(card, ANSWER)).toBe(1);
  });

  it("names itself by its own name while the findings are not read", () => {
    const { card } = cardOf(log, "whos_in", { answers: new Map([[exclusions.decision_id!, ANSWER]]), seed: seeded(false) });
    expect(lineEl(card, KEY)).toHaveTextContent(`${name}: settled by ${exclusions.name}`);
    expect(count(card, ANSWER)).toBe(1);
  });

  it("is listed the same way in First look", () => {
    const { card } = cardOf(log, "first_look", { answers: new Map([[exclusions.decision_id!, ANSWER]]), seed: seeded(true) });
    const item = within(card).getByTestId(`noticed-${KEY}`);
    expect(item).toHaveTextContent(SETTLED);
    expect(item).toHaveTextContent(`Settled in ${stageIn(REACHED, "whos_in").name}`);
    expect(count(card, ANSWER)).toBe(0);
  });
});

describe("First look (P14)", () => {
  it("lists each noticing with the stage that decides it, each a link there", () => {
    const open = vi.fn();
    const { card } = cardOf(REACHED, "first_look", { open });
    const findings = REACHED.stages.flatMap((s) => s.lines.filter((l) => l.source === "finding").map((l) => ({ s, l })));
    // Several noticings; under Predict the dietary energy noticing no longer fires (ruling 5, W1-P11).
    expect(findings.length).toBeGreaterThan(1);
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
    // The engine states the scheme itself since the re-walk's N6; a log that serves none (an older
    // capture) is stated from the split, in the same words.
    const unstated = patched(REACHED, { models: (l) => (l.key === "set_validation" ? { ...l, reason: null } : l) });
    expect(stageIn(unstated, "models").lines.find((l) => l.key === "set_validation")).toMatchObject({ label: "Confirm", status: "set_for_you", reason: null });
    const split = VIEW.state.split!;
    expect(split.validation).toBe("kfold");
    expect(split.nested_cv).toBe(false);
    const { card } = cardOf(unstated, "models");
    expect(within(card).getByTestId("line-set_validation-reason")).toHaveTextContent(`${split.folds}-fold cross-validation.`);
  });

  it("is the engine's own statement where it serves one (N6)", () => {
    const served = stageIn(REACHED, "models").lines.find((l) => l.key === "set_validation")!;
    expect(served.reason).toMatch(/^\d+-fold cross-validation\.$/);
    const { card } = cardOf(REACHED, "models");
    expect(within(card).getByTestId("line-set_validation-reason")).toHaveTextContent(served.reason!);
  });

  it("leaves the engine's own reason where it serves one", () => {
    const form = stageIn(REACHED, "models").lines.find((l) => l.key === "form")!;
    expect(form.reason).toBeTruthy();
    const { card } = cardOf(REACHED, "models");
    expect(within(card).getByTestId("line-form-reason")).toHaveTextContent(form.reason!.replaceAll("`", ""));
  });
});

/**
 * The re-walk's N6: "Confirm all" is one record, but each line it answers reads as its own default,
 * as stated (the line's reason, which the sweep recorded), never the record's one sentence for all
 * of them. Models' sweep is confirmed here by construction: each Confirm line answered by one record.
 */
describe("a line answered by its stage's Confirm all (N6)", () => {
  const RECORD_ID = "r-models-sweep";
  const COMBINED = "Confirmed the 3 choices set for you in Models, as stated: one; two; three.";
  const OWN: Record<string, string> = {
    form: "Under prediction each predictor enters as each family takes it.",
    set_validation: "5-fold cross-validation.",
    set_levers: "Predictor forms: none; variance filter: none; imbalance correction: none.",
  };
  const swept = patched(
    REACHED,
    { models: (l) => (l.label === "Confirm" ? { ...l, status: "answered", decision_id: RECORD_ID, reason: OWN[l.key] ?? l.reason } : l) },
    { models: { sweep: { ...stageIn(REACHED, "models").sweep!, answered: true, confirmed_by: RECORD_ID } } },
  );

  it("reads as its own value, not the record's sentence for all of them", () => {
    const confirms = stageIn(swept, "models").lines.filter((l) => l.label === "Confirm");
    expect(confirms.map((l) => l.key).sort()).toEqual(Object.keys(OWN).sort());
    const { card } = cardOf(swept, "models", { answers: new Map([[RECORD_ID, COMBINED]]) });
    for (const l of confirms) {
      expect(lineEl(card, l.key)).toHaveTextContent(OWN[l.key]!);
      expect(lineEl(card, l.key)).not.toHaveTextContent(COMBINED);
    }
  });

  it("keeps a record of its own: a line the person answered reads as that record's sentence", () => {
    const own = patched(swept, { models: (l) => (l.key === "set_validation" ? { ...l, decision_id: "r-validation" } : l) });
    const { card } = cardOf(own, "models", { answers: new Map([[RECORD_ID, COMBINED], ["r-validation", "The validation scheme was changed."]]) });
    expect(lineEl(card, "set_validation")).toHaveTextContent("The validation scheme was changed.");
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

/**
 * Finding 24 of the 2026-10-17 walkthrough: a card recorded on a line that is not the stage's open
 * line (an answer changed, a later line answered first) stayed expanded. A record on the expanded
 * line collapses it, and the stage's open line opens; a change elsewhere leaves it as it is.
 */
describe("a recorded card (finding 24)", () => {
  const expanded = (card: HTMLElement) => card.querySelector("[data-expanded]")?.getAttribute("data-line");
  /** Who's in in a StageCard, re-rendered with each log in turn. */
  function whosIn(first: QuestLog) {
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
    qc.setQueryData(keys.view(PID), VIEW);
    qc.setQueryData(keys.record(PID), RECORD);
    const ui = (log: QuestLog) => (
      <QueryClientProvider client={qc}>
        <StageFocusProvider>
          <StageCard pid={PID} log={log} stage={stageIn(log, "whos_in")} answers={new Map()} fit={{ kind: "none" }} onOpenStage={() => {}} arrival={null} onOpenManuscript={() => {}} />
        </StageFocusProvider>
      </QueryClientProvider>
    );
    const r = render(ui(first));
    return { card: screen.getByTestId("card-whos_in"), next: (log: QuestLog) => r.rerender(ui(log)) };
  }
  // Who's in as the engine serves it: the noticing is the open line, the eligibility answered.
  const NOTICING = "finding:pack::dietary::implausible_intake#1";
  const withExclusions = (decision: string) =>
    patched(REACHED, { whos_in: (l) => (l.key === "exclusions" ? { ...l, decision_id: decision } : l) });

  it("collapses once its line's answer is recorded, and the open line opens", () => {
    const before = stageIn(REACHED, "whos_in").lines.find((l) => l.key === "exclusions")!;
    expect(before.status).toBe("answered");
    const { card, next } = whosIn(REACHED);
    expect(expanded(card)).toBe(NOTICING);
    fireEvent.click(within(lineEl(card, "exclusions")).getByRole("button", { name: "Change" }));
    expect(expanded(card)).toBe("question:exclusions");
    next(withExclusions(`${before.decision_id}-changed`));
    expect(expanded(card)).toBe(NOTICING);
    expect(lineEl(card, "exclusions")).not.toHaveAttribute("data-expanded");
  });

  it("stays expanded while its own record has not changed", () => {
    const { card, next } = whosIn(REACHED);
    fireEvent.click(within(lineEl(card, "exclusions")).getByRole("button", { name: "Change" }));
    // Another line's record changes; this one's does not.
    next(patched(REACHED, { whos_in: (l) => (l.key === "missing" ? { ...l, decision_id: "r-missing-changed" } : l) }));
    expect(expanded(card)).toBe("question:exclusions");
  });
});

describe("an open line that only comes back from a recompute (the own-csv race)", () => {
  const expanded = (card: HTMLElement) => card.querySelector("[data-expanded]")?.getAttribute("data-line") ?? null;
  /** Who's in in a StageCard, re-rendered with each log in turn, as GET /quest serves them. */
  function whosIn(first: QuestLog) {
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
    qc.setQueryData(keys.view(PID), VIEW);
    qc.setQueryData(keys.record(PID), RECORD);
    const ui = (log: QuestLog) => (
      <QueryClientProvider client={qc}>
        <StageFocusProvider>
          <StageCard pid={PID} log={log} stage={stageIn(log, "whos_in")} answers={new Map()} fit={{ kind: "none" }} onOpenStage={() => {}} arrival={null} onOpenManuscript={() => {}} />
        </StageFocusProvider>
      </QueryClientProvider>
    );
    const r = render(ui(first));
    return { card: screen.getByTestId("card-whos_in"), next: (log: QuestLog) => r.rerender(ui(log)) };
  }
  const NOTICING = "finding:pack::dietary::implausible_intake#1";
  const GRAIN = "question:grain";
  const EXCLUSIONS = "question:exclusions";
  const WAITS = { key: "grain", stage: "whos_in", name: CONTRACT.questions.grain?.name ?? "grain" };
  /** Who's in by construction: the grain question and the exclusions question as `grain` and
   *  `exclusions` set them, the noticing open, the rest answered as the engine served them. */
  const at = (grain: (l: QuestLine) => QuestLine, exclusions: (l: QuestLine) => QuestLine) =>
    patched(REACHED, { whos_in: (l) => (l.key === "grain" ? grain(l) : l.key === "exclusions" ? exclusions(l) : l) });
  const open = (l: QuestLine): QuestLine => ({ ...l, status: "open", decision_id: null, waiting_for: [] });
  const waiting = (l: QuestLine): QuestLine => ({ ...l, status: "waiting", decision_id: null, waiting_for: [WAITS] });
  const answered = (id: string) => (l: QuestLine): QuestLine => ({ ...l, status: "answered", decision_id: id, waiting_for: [] });

  it("keeps the line expanded where the open line went waiting and came back with no answer", () => {
    // The grain is the open line, and expanded; its answer is recorded, and the exclusions wait
    // while their stage computes again: the noticing is the open line now, and expanded.
    const { card, next } = whosIn(at(open, open));
    expect(expanded(card)).toBe(GRAIN);
    next(at(answered("r-grain"), waiting));
    expect(expanded(card)).toBe(NOTICING);
    // The exclusions are open again with no answer of anyone's between: the noticing stays.
    next(at(answered("r-grain"), open));
    expect(expanded(card)).toBe(NOTICING);
    expect(lineEl(card, "exclusions")).not.toHaveAttribute("data-expanded");
  });

  it("keeps the line the person expanded through its own line's wait, with no answer", () => {
    const { card, next } = whosIn(at(answered("r-grain"), open));
    expect(expanded(card)).toBe(EXCLUSIONS);
    // Its line waits while it computes: the noticing, open, is not expanded in its place.
    next(at(answered("r-grain"), waiting));
    expect(expanded(card)).toBeNull();
    next(at(answered("r-grain"), open));
    expect(expanded(card)).toBe(EXCLUSIONS);
  });

  it("moves to the next open line once the open line is answered", () => {
    const { card, next } = whosIn(at(open, open));
    expect(expanded(card)).toBe(GRAIN);
    next(at(answered("r-grain"), open));
    expect(expanded(card)).toBe(EXCLUSIONS);
    next(at(answered("r-grain"), answered("r-exclusions")));
    expect(expanded(card)).toBe(NOTICING);
  });

  it("opens the open line where nothing was open, and so nothing expanded", () => {
    const { card, next } = whosIn(
      patched(REACHED, { whos_in: (l) => (l.source === "finding" ? waiting(l) : l) }),
    );
    expect(expanded(card)).toBeNull();
    next(REACHED);
    expect(expanded(card)).toBe(NOTICING);
  });
});

/** Fit's foot beside a noticing in Models under Predict (ruling R-a): it names Results' triage. */
describe("the fit's foot (ruling R-a)", () => {
  it("names the noticings that wait for the triage in Results", () => {
    const models = stageIn(NOT_REACHED, "models");
    const noticing = stageIn(NOT_REACHED, "whos_in").lines.find((l) => l.source === "finding")!;
    const answered = models.lines.map((l): QuestLine => (l.label === "Decide" ? { ...l, status: "answered", decision_id: `r-${l.key}`, waiting_for: [] } : l));
    const log: QuestLog = {
      ...NOT_REACHED,
      stages: NOT_REACHED.stages.map((s) =>
        s.key === "models" ? { ...s, sweep: { ...models.sweep!, answered: true }, lines: [...answered, { ...noticing, id: "finding:m", key: "m" }] } : s,
      ),
    };
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
    qc.setQueryData(keys.view(PID), VIEW);
    qc.setQueryData(keys.record(PID), RECORD);
    render(
      <QueryClientProvider client={qc}>
        <StageFocusProvider>
          <StageCard pid={PID} log={log} stage={stageIn(log, "models")} answers={new Map()} fit={{ kind: "press", held: false, estimate: null }} onOpenStage={() => {}} arrival={null} onOpenManuscript={() => {}} />
        </StageFocusProvider>
      </QueryClientProvider>,
    );
    expect(screen.getByTestId("primary-fit")).toHaveTextContent(`1 noticing waits for the triage in ${stageIn(log, "results").name}.`);
    expect(screen.queryByTestId("primary")).toBeNull();
  });
});

/**
 * The foot's refusal (BLUEPRINT §14 rule 2; ruling R-b): a refused press at the foot (the Confirm
 * sweep, Download) draws the server's ways forward as presses, but never one that confirms a
 * column's role together with other readings ("Confirm each of the 7 as shown"); a role confirmed
 * alone stays, and a refusal with no such block keeps every way forward, a block of other
 * readings included.
 */
describe("the foot's refusal, one role at a time (ruling R-b)", () => {
  // Models as the engine serves it with its Decides answered: the sweep is the foot's action.
  const log = patched(REACHED, { models: (l) => (l.label === "Decide" ? { ...l, status: "answered", decision_id: `r-${l.key}`, waiting_for: [] } : l) });
  const role = (column: string, value: string) => ({ reading: "role", column, value });
  const exit = (label: string, decision: Record<string, unknown> | null) => ({ label, decision });
  /** The sweep's press, answered with `exits` in a refusal. */
  const refusedWith = async (exits: ReturnType<typeof exit>[]) => {
    const body = { error: { code: "readings_unsettled", message: "The sweep waits for the readings it rests on.", exits } };
    vi.stubGlobal(
      "fetch",
      vi.fn(() => Promise.resolve(new Response(JSON.stringify(body), { status: 409, headers: { "Content-Type": "application/json" } }))),
    );
    // The foot is the sweep's press: every counted Decide answered, the sweep still to confirm.
    const models = stageIn(log, "models");
    expect(models.sweep).toMatchObject({ answered: false });
    expect(models.lines.filter((l) => l.label === "Decide" && l.counted).every((l) => l.status === "answered")).toBe(true);
    const { card } = cardOf(log, "models");
    expect(within(card).getByTestId("primary")).toHaveTextContent(models.sweep!.action);
    fireEvent.click(within(card).getByTestId("primary"));
    const refusal = await screen.findByTestId("primary-refusal");
    return within(refusal).queryAllByRole("button").map((b) => b.textContent);
  };

  it("draws no exit that confirms several roles in one press, and draws the one that confirms a role alone", async () => {
    const labels = await refusedWith([
      exit("Confirm each of the 2 as shown", { kind: "confirm_readings", items: [role("bp_sys", "covariate"), role("imputed_bp", "flag")] }),
      exit("Confirm `bp_sys` as a predictor", { kind: "confirm_readings", items: [role("bp_sys", "covariate")] }),
    ]);
    expect(labels).toEqual(["Confirm `bp_sys` as a predictor"]);
  });

  it("keeps every exit of a refusal that confirms no role in a block: a block of other readings, a role alone, a way with no decision", async () => {
    const codes = [
      { reading: "code_or_count", column: "kids", value: "amount" },
      { reading: "code_or_count", column: "visits", value: "amount" },
    ];
    const labels = await refusedWith([
      exit("Confirm the 2 as amounts", { kind: "confirm_readings", items: codes }),
      exit("Confirm `bp_sys` as a predictor", { kind: "confirm_readings", items: [role("bp_sys", "covariate")] }),
      exit("Choose the models", null),
    ]);
    expect(labels).toEqual(["Confirm the 2 as amounts", "Confirm `bp_sys` as a predictor", "Choose the models"]);
  });
});
