/**
 * The Estimate declaration cards (the front-end audit's G01 and G08): the Model 1 question, how many
 * tests are accounted for, and the regression calibration question, each through DECLARATION_CARDS.
 *
 * On the base none of the three has a card: each falls to the fallback, which records nothing, so
 * the shell lists it "Not available yet" and never opens it, and under Estimate the counted Model 1
 * question holds Models open and Fit is never offered. Here each records.
 *
 * The truth each card is held to is the server's, never the card's own constants: the Model 1 card's
 * from the proposals stage a real server served (the M3 captures in src/mocks/fixtures), and every
 * payload, list of options, bound and default from turbotab/server/openapi.json, the contract the
 * server's own test holds to its pydantic models (core/decisions.py).
 */
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { api, RefusalError } from "../../../api/client";
import type { ProposalsArtifact } from "../../../api/m1-types";
import { keys } from "../../../api/queries";
import { STAGE_LABEL } from "../../../components/JobChips";
import type { Decision, ProjectState, ProjectView, QuestLine, QuestStage, Refusal } from "../../../api/schema";
import { artifactAt, expandView, viewsOf, type M3Fixture } from "../../../mocks/m3";
import { StageFocusProvider } from "../../../state/focus";
import { openLine } from "../../log";
import { DECLARATION_CARDS } from "../predict";
import { recordable, rendererOf } from "../renderers";

const PID = "estimate-test";
const AT = "2026-11-03T00:00:00Z";

// ── the server's contract (openapi.json) ─────────────────────────────────────

interface JsonSchema {
  type?: string;
  enum?: unknown[];
  const?: unknown;
  minimum?: number;
  maximum?: number;
  minLength?: number;
  default?: unknown;
  items?: JsonSchema;
  anyOf?: JsonSchema[];
  properties?: Record<string, JsonSchema>;
  required?: string[];
  additionalProperties?: boolean;
}
const OPENAPI = resolve(dirname(fileURLToPath(import.meta.url)), "../../../../../server/openapi.json");
const SCHEMAS = (JSON.parse(readFileSync(OPENAPI, "utf-8")) as { components: { schemas: Record<string, JsonSchema> } })
  .components.schemas;
const KINDS = ["set_model_sequence", "set_multiplicity", "set_measurement_error"] as const;
type Kind = (typeof KINDS)[number];
/** The model a posted decision is parsed as (the request body's variant). */
const MODEL: Record<Kind, JsonSchema> = {
  set_model_sequence: SCHEMAS["SetModelSequence-Input"]!,
  set_multiplicity: SCHEMAS["SetMultiplicity-Input"]!,
  set_measurement_error: SCHEMAS["SetMeasurementError-Input"]!,
};
const field = (kind: Kind, name: string): JsonSchema => MODEL[kind].properties![name]!;

/** What keeps `value` from parsing as schema `s`: one line per problem, none when it parses. */
function problems(s: JsonSchema, value: unknown, at = "decision"): string[] {
  if (s.anyOf) return s.anyOf.some((x) => problems(x, value, at).length === 0) ? [] : [`${at}: no variant takes it`];
  const out: string[] = [];
  if (s.const !== undefined && value !== s.const) out.push(`${at}: not ${String(s.const)}`);
  if (s.enum && !s.enum.includes(value)) out.push(`${at}: ${String(value)} is not one of ${s.enum.join(", ")}`);
  switch (s.type) {
    case "null":
      if (value !== null) out.push(`${at}: not null`);
      break;
    case "boolean":
      if (typeof value !== "boolean") out.push(`${at}: not a boolean`);
      break;
    case "string":
      if (typeof value !== "string") out.push(`${at}: not a string`);
      else if (s.minLength !== undefined && value.length < s.minLength) out.push(`${at}: too short`);
      break;
    case "integer":
    case "number":
      if (typeof value !== "number" || (s.type === "integer" && !Number.isInteger(value))) out.push(`${at}: not a ${s.type}`);
      else {
        if (s.minimum !== undefined && value < s.minimum) out.push(`${at}: below ${s.minimum}`);
        if (s.maximum !== undefined && value > s.maximum) out.push(`${at}: above ${s.maximum}`);
      }
      break;
    case "array":
      if (!Array.isArray(value)) out.push(`${at}: not a list`);
      else value.forEach((x, i) => out.push(...problems(s.items ?? {}, x, `${at}[${i}]`)));
      break;
    case "object": {
      if (typeof value !== "object" || value === null || Array.isArray(value)) {
        out.push(`${at}: not an object`);
        break;
      }
      const v = value as Record<string, unknown>;
      for (const k of s.required ?? []) if (!(k in v)) out.push(`${at}.${k}: missing`);
      for (const [k, x] of Object.entries(v)) {
        const p = s.properties?.[k];
        if (!p) {
          if (s.additionalProperties === false) out.push(`${at}.${k}: not a field of the model`);
        } else out.push(...problems(p, x, `${at}.${k}`));
      }
      break;
    }
  }
  return out;
}

/** The model's defaults for every field it defaults, from the contract. */
const defaultsOf = (kind: Kind): Record<string, unknown> =>
  Object.fromEntries(
    Object.entries(MODEL[kind].properties!)
      .filter(([k, p]) => k !== "kind" && p.default !== undefined)
      .map(([k, p]) => [k, p.default]),
  );

/** The cards' payloads module, loaded by the test that holds its constants to the contract (a glob,
 *  so that a tree without it fails that test alone). */
const PAYLOADS = import.meta.glob<typeof import("./payloads")>("./payloads.ts");

// ── the captured server (src/mocks/fixtures) ─────────────────────────────────

const FIXTURES = import.meta.glob<M3Fixture>(
  ["../../../mocks/fixtures/m3-nhanes-inference.json", "../../../mocks/fixtures/m3-survey-design.json"],
  { eager: true, import: "default" },
);
const journey = (name: string): M3Fixture => Object.entries(FIXTURES).find(([p]) => p.endsWith(`m3-${name}.json`))![1];

type Served = NonNullable<ProposalsArtifact["model_sequence"]>;

/** The last captured view whose proposals stage serves Model 1's declaration, and that artifact. */
function servedIn(name: string): { f: M3Fixture; view: ProjectView; proposals: ProposalsArtifact; served: Served } {
  const f = journey(name);
  const views = viewsOf(f);
  for (let i = views.length - 1; i >= 0; i--) {
    const key = views[i]!.stages.proposals?.key ?? null;
    const proposals = key ? (artifactAt(f, "proposals", key) as ProposalsArtifact | null) : null;
    if (proposals?.model_sequence) {
      return { f, view: expandView(f, views[i]!, PID, AT), proposals, served: proposals.model_sequence };
    }
  }
  throw new Error(`${name} serves no Model 1 declaration`);
}

/** A capture's view and proposals result once `declared` is recorded for the served exposure: the
 *  state holds the sequence (the quest line is answered while it holds, core/quest.py
 *  `_model_one_is_current`) and the fresh result serves the same columns as declared
 *  (core/estimand.py `model_sequence_card`). */
function declaredIn(
  at: { view: ProjectView; proposals: ProposalsArtifact; served: Served },
  declared: string[],
): { view: ProjectView; proposals: ProposalsArtifact } {
  const model_sequence = { exposure: at.served.decision.exposure as string, model_1: declared };
  return {
    view: { ...at.view, state: { ...at.view.state, model_sequence } } as ProjectView,
    proposals: { ...at.proposals, model_sequence: { ...at.served, declared } },
  };
}

/** The Model 1 columns checked now, in the card's order. */
const checkedNow = () =>
  Array.from(document.querySelectorAll<HTMLInputElement>("[data-testid^=pick-model_1-]"))
    .filter((p) => p.checked)
    .map((p) => p.dataset.testid!.slice("pick-model_1-".length));

// ── mounting ────────────────────────────────────────────────────────────────

const NAMES: Record<Kind, string> = {
  set_model_sequence: "the Model 1 question",
  set_multiplicity: "how many tests are accounted for",
  set_measurement_error: "the regression calibration question",
};
const LABELS: Record<Kind, QuestLine["label"]> = {
  set_model_sequence: "Decide",
  set_multiplicity: "Confirm",
  set_measurement_error: "Decide",
};

function lineOf(kind: string, over: Partial<QuestLine> = {}): QuestLine {
  const k = kind as Kind;
  return {
    id: `decision:${kind}`,
    key: kind,
    source: "declaration",
    label: LABELS[k] ?? "Decide",
    name: NAMES[k] ?? kind,
    status: LABELS[k] === "Confirm" ? "set_for_you" : "open",
    counted: LABELS[k] !== "Confirm",
    order: 1,
    decision_id: null,
    reason: null,
    waiting_for: [],
    computing: [],
    reopened_by: null,
    would_change: null,
    changes_nothing: null,
    items: [],
    changed_since: null,
    ...over,
  };
}

/** An Estimate project's state where no capture is mounted: two study factors on recalls. The
 *  multiplicity and calibration cards read only their own slots of it. */
function stateOf(over: Record<string, unknown> = {}): ProjectState {
  return {
    lens: ["dietary"],
    target: "glucose",
    task: "regression",
    purpose: "inference",
    roles: { sugar: "exposure", fiber: "exposure", age: "covariate", glucose: "outcome" },
    multiplicity: null,
    measurement_error: null,
    model_sequence: null,
    ...over,
  } as unknown as ProjectState;
}

/** How the proposals stage stands where it is not fresh: recomputing, failed, or stopped. */
type NotFresh =
  | { status: "running" }
  | { status: "error"; error: string }
  | { status: "stale"; cancelled: true };

/**
 * A stop as the engine serves it to the proposals stage. Only a heavy stage's job can be stopped
 * (the server cancels by job id, server/service.py `cancel_job`), and proposals is a light stage
 * with none, so its `cancelled` is always inherited from a stage it waits on (core/graph.py
 * `_status`: held, or any dep cancelled): here the working table's stop, which the column roles
 * inherit too, leaves proposals stale and cancelled.
 */
const STOPPED_UPSTREAM: Record<string, Record<string, unknown>> = {
  working: { stage: "working", status: "stale", key: "w", fresh: false, updated_at: null, error: null, cancelled: true },
  roles: { stage: "roles", status: "stale", key: "r", fresh: false, updated_at: null, error: null, cancelled: true },
};
const STOPPED: NotFresh = { status: "stale", cancelled: true };
const withStages = (view: ProjectView, stages: Record<string, Record<string, unknown>>): ProjectView =>
  ({ ...view, stages: { ...view.stages, ...stages } }) as unknown as ProjectView;

/**
 * Serve the proposals stage: fresh, its result at the key the view reports; or not fresh
 * (recomputing unless `over` says how it stands), the view reporting the key it computes while
 * the server serves the latest older result with `fresh: false` (core/graph.py `Engine.get`,
 * allow_stale).
 */
function serveProposals(qc: QueryClient, view: ProjectView, artifact: ProposalsArtifact | null, fresh = true, over?: NotFresh) {
  const status = over?.status ?? (fresh ? "fresh" : "running");
  qc.setQueryData(keys.view(PID), {
    ...view,
    stages: {
      ...view.stages,
      proposals: { stage: "proposals", status, key: "p", fresh, updated_at: null, error: null, cancelled: false, ...over },
    },
  } as unknown as ProjectView);
  qc.setQueryData(keys.stage(PID, "proposals"), { stage: "proposals", key: fresh ? "p" : "p0", fresh, status, artifact });
}

function mount(
  kind: Kind,
  opts: {
    view?: ProjectView;
    proposals?: ProposalsArtifact | null;
    /** Serve the proposals result as an older one while the stage recomputes. */
    stale?: boolean;
    /** Serve it as an older one while the stage stands so (failed, stopped). */
    status?: NotFresh;
    state?: Record<string, unknown>;
    line?: Partial<QuestLine>;
  } = {},
) {
  const view =
    opts.view ??
    ({
      state: stateOf(opts.state),
      decisions: [],
      interview: [],
      summary: null,
      stages: { ingest: { stage: "ingest", status: "fresh", key: "k", updated_at: null } },
    } as unknown as ProjectView);
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  qc.setQueryData(keys.view(PID), view);
  if (opts.proposals !== undefined) serveProposals(qc, view, opts.proposals, !opts.stale && !opts.status, opts.status);
  const withProposals = qc.getQueryData<ProjectView>(keys.view(PID))!;
  qc.setQueryData(keys.columns(PID), Object.keys(withProposals.state.roles ?? {}).map((name) => ({ name })));
  const decide = vi.spyOn(api, "decide").mockResolvedValue(withProposals);
  // The engine's preview of an option pointed at says nothing here: the words are its, not these tests'.
  vi.spyOn(api, "preview").mockImplementation(() => new Promise(() => {}));
  const Card = DECLARATION_CARDS[kind];
  if (!Card) throw new Error(`no card for ${kind}`);
  render(
    <QueryClientProvider client={qc}>
      <StageFocusProvider>
        <Card pid={PID} line={lineOf(kind, opts.line)} />
      </StageFocusProvider>
    </QueryClientProvider>,
  );
  return { decide, qc };
}

/** A press as a mouse makes it: a click chooses the option, and the card's own button records it. */
const press = (option: string) => {
  fireEvent.click(screen.getByTestId(`option-${option}`));
  fireEvent.click(screen.getByTestId("card-record"));
};
const offered = () => Array.from(document.querySelectorAll("[role=option]")).map((o) => o.getAttribute("data-key"));
const settle = () => new Promise((resolve) => setTimeout(resolve, 20));
const posted = async (decide: ReturnType<typeof vi.spyOn>, n = 1): Promise<Decision> => {
  await waitFor(() => expect(decide).toHaveBeenCalledTimes(n));
  return decide.mock.calls[n - 1]![1] as Decision;
};

afterEach(() => {
  vi.restoreAllMocks();
});

// ── the shell ───────────────────────────────────────────────────────────────

describe("the Estimate declarations the shell can answer (G01, G08)", () => {
  it("draws the Model 1 question, the multiplicity and the calibration question through a card that records", () => {
    for (const key of KINDS) {
      expect(rendererOf({ source: "declaration", key }, DECLARATION_CARDS), key).toBe("declaration");
      expect(recordable({ source: "declaration", key }, DECLARATION_CARDS), key).toBe(true);
    }
  });

  it("leaves the batch, usual-intake and scales questions to the fallback, as later packages' cards", () => {
    for (const key of ["set_batch", "set_usual_intake", "set_scales"]) {
      expect(rendererOf({ source: "declaration", key }, DECLARATION_CARDS), key).toBe("generic");
    }
  });

  it("opens Models on the Model 1 question, where without its card the stage opens nothing", () => {
    // The card column's own test of a line it can answer (StageCard.tsx `canAnswer`).
    const models = {
      key: "models",
      name: "Models",
      reached: true,
      progress: { answered: 0, required: 2, complete: false },
      sweep: null,
      lines: [lineOf("set_model_sequence"), lineOf("set_measurement_error"), lineOf("set_multiplicity")],
      reopened: [],
    } as unknown as QuestStage;
    expect(openLine(models, (l) => recordable(l, DECLARATION_CARDS))?.key).toBe("set_model_sequence");
    // Without the cards no line opens: the base's dead end, "Not available yet".
    expect(openLine(models, (l) => recordable(l, {}))).toBeNull();
  });
});

// ── the Model 1 question ────────────────────────────────────────────────────

describe("the Model 1 card, from the proposals stage's model_sequence", () => {
  it("says the engine's reason and lists the columns Model 1 may hold, in its order, none checked", () => {
    const { view, proposals, served } = servedIn("nhanes-inference");
    expect(served.allowed.length).toBeGreaterThan(served.guess.length);
    mount("set_model_sequence", { view, proposals });
    const card = screen.getByTestId("declaration-set_model_sequence");
    expect(within(card).getByRole("heading")).toHaveTextContent("The Model 1 question");
    expect(within(card).getByTestId("model-one-reason")).toHaveTextContent(served.reason);
    const picks = Array.from(document.querySelectorAll<HTMLInputElement>("[data-testid^=pick-model_1-]"));
    expect(picks.map((p) => p.dataset.testid)).toEqual(served.allowed.map((c) => `pick-model_1-${c}`));
    // A guess suggests beside an option; it never pre-selects a column.
    expect(picks.filter((p) => p.checked)).toHaveLength(0);
  });

  it("records the engine's one-tap decision from the suggested option", async () => {
    const { view, proposals, served } = servedIn("nhanes-inference");
    const { decide } = mount("set_model_sequence", { view, proposals });
    expect(offered()).toEqual(["model_1:guess", "model_1:checked"]);
    for (const c of served.guess) expect(screen.getByTestId("option-model_1:guess")).toHaveTextContent(c);
    press("model_1:guess");
    const sent = await posted(decide);
    expect(sent).toEqual(served.decision);
    expect(problems(MODEL.set_model_sequence, sent)).toEqual([]);
  });

  it("records the columns checked, in the engine's order, for the exposure the engine declares it for", async () => {
    const { view, proposals, served } = servedIn("nhanes-inference");
    const { decide } = mount("set_model_sequence", { view, proposals });
    // Two the guess leaves out, checked last first.
    const two = served.allowed.filter((c) => !served.guess.includes(c)).slice(0, 2);
    expect(two).toHaveLength(2);
    for (const c of [...two].reverse()) fireEvent.click(screen.getByTestId(`pick-model_1-${c}`));
    for (const c of two) expect(screen.getByTestId("option-model_1:checked")).toHaveTextContent(c);
    press("model_1:checked");
    const sent = await posted(decide);
    expect(sent).toEqual({ kind: "set_model_sequence", exposure: served.decision.exposure, model_1: two });
    expect(problems(MODEL.set_model_sequence, sent)).toEqual([]);
  });

  it("records Model 1 with no column where none is checked, as the model takes it", async () => {
    const { view, proposals, served } = servedIn("nhanes-inference");
    const { decide } = mount("set_model_sequence", { view, proposals });
    press("model_1:checked");
    const sent = await posted(decide);
    expect(sent).toEqual({ kind: "set_model_sequence", exposure: served.decision.exposure, model_1: [] });
    expect(problems(MODEL.set_model_sequence, sent)).toEqual([]);
  });

  it("suggests nothing where the engine's guess is empty, and offers the columns checked alone", () => {
    const { view, proposals, served } = servedIn("survey-design");
    expect(served.guess).toEqual([]);
    mount("set_model_sequence", { view, proposals });
    expect(offered()).toEqual(["model_1:checked"]);
  });

  it("checks the columns declared once the line is answered, and marks the option that records them", () => {
    const { view, proposals, served } = servedIn("nhanes-inference");
    const declared = served.allowed.filter((c) => !served.guess.includes(c)).slice(0, 1);
    const at = declaredIn({ view, proposals, served }, declared);
    mount("set_model_sequence", { ...at, line: { status: "answered", decision_id: "r1" } });
    expect(checkedNow()).toEqual(declared);
    expect(screen.getByTestId("option-model_1:checked")).toHaveAttribute("data-recorded");
    expect(screen.getByTestId("option-model_1:guess")).not.toHaveAttribute("data-recorded");
  });

  it("marks the suggested option recorded where the columns declared are the guess", () => {
    const { view, proposals, served } = servedIn("nhanes-inference");
    const at = declaredIn({ view, proposals, served }, [...served.guess].reverse());
    mount("set_model_sequence", { ...at, line: { status: "answered", decision_id: "r1" } });
    expect(screen.getByTestId("option-model_1:guess")).toHaveAttribute("data-recorded");
    expect(screen.getByTestId("option-model_1:checked")).not.toHaveAttribute("data-recorded");
  });

  it("checks nothing and marks nothing recorded on an open line, whatever sequence the state holds", () => {
    // An open line's sequence on record no longer holds (another exposure, or a column gone from
    // the primary set: core/estimand.py `current_model_sequence`), so it is no answer to show.
    const { view, proposals, served } = servedIn("nhanes-inference");
    const declared = served.allowed.filter((c) => !served.guess.includes(c)).slice(0, 2);
    const at = declaredIn({ view, proposals, served }, declared);
    mount("set_model_sequence", { view: at.view, proposals });
    expect(checkedNow()).toEqual([]);
    expect(document.querySelectorAll("[data-recorded]")).toHaveLength(0);
  });

  it("keeps the checks and the recorded mark on the columns declared while the proposals result catches up with the record", async () => {
    // The proposals stage reads model_sequence, so a record recomputes it; meanwhile the server
    // serves its older result, the one before the record (declared null), with fresh false.
    const { view, proposals, served } = servedIn("nhanes-inference");
    const declared = served.allowed.filter((c) => !served.guess.includes(c)).slice(0, 2);
    expect(declared).toHaveLength(2);
    const at = declaredIn({ view, proposals, served }, declared);
    const before = { ...proposals, model_sequence: { ...served, declared: null } };
    const { qc } = mount("set_model_sequence", { view: at.view, proposals: before, stale: true, line: { status: "answered", decision_id: "r1" } });
    expect(checkedNow()).toEqual(declared);
    act(() => serveProposals(qc, at.view, at.proposals));
    await waitFor(() => expect(screen.getByTestId("model-one-veil")).toHaveAttribute("data-veil", "fresh"));
    expect(checkedNow()).toEqual(declared);
    const recorded = screen.getByTestId("option-model_1:checked");
    expect(recorded).toHaveAttribute("data-recorded");
    for (const c of declared) expect(recorded).toHaveTextContent(c);
    expect(screen.getByTestId("option-model_1:guess")).not.toHaveAttribute("data-recorded");
  });

  it("veils the columns and the options, inert, while the proposals result is not fresh", () => {
    const { view, proposals } = servedIn("nhanes-inference");
    mount("set_model_sequence", { view, proposals, stale: true });
    const veil = screen.getByTestId("model-one-veil");
    expect(veil.getAttribute("data-veil")).not.toBe("fresh");
    for (const id of ["model-one-reason", "options-model-one", "card-record"]) {
      expect(screen.getByTestId(id).closest("[inert]"), id).not.toBeNull();
    }
    expect(veil).toContainElement(screen.getByTestId("option-model_1:guess"));
  });

  it("leaves them unveiled and live once the proposals result is fresh", async () => {
    const { view, proposals } = servedIn("nhanes-inference");
    const { qc } = mount("set_model_sequence", { view, proposals, stale: true });
    act(() => serveProposals(qc, view, proposals));
    await waitFor(() => expect(screen.getByTestId("model-one-veil")).toHaveAttribute("data-veil", "fresh"));
    for (const id of ["model-one-reason", "options-model-one", "card-record"]) {
      expect(screen.getByTestId(id).closest("[inert]"), id).toBeNull();
    }
  });

  it("says so, and offers nothing to record, where the proposals stage serves no Model 1 declaration", async () => {
    const { view, proposals } = servedIn("nhanes-inference");
    const { decide } = mount("set_model_sequence", { view, proposals: { ...proposals, model_sequence: null } });
    expect(screen.getByTestId("model-one-unserved")).toBeInTheDocument();
    expect(offered()).toEqual([]);
    await settle();
    expect(decide).not.toHaveBeenCalled();
  });

  // The line opens once the adjustment is answered, and only a result computed after that answer
  // serves Model 1's columns (core/stages/proposals.py): the older result the server falls back to
  // serves none, so a recompute that fails or is stopped leaves no columns to show.
  const unservedAfter = (status: NotFresh, upstream: Record<string, Record<string, unknown>> = {}) => {
    const { view, proposals } = servedIn("nhanes-inference");
    mount("set_model_sequence", { view: withStages(view, upstream), proposals: { ...proposals, model_sequence: null }, status });
    const card = screen.getByTestId("declaration-set_model_sequence");
    return { card, title: within(card).getByRole("heading").textContent! };
  };

  it("names the stage that did not finish, its reason and its retry where the proposals stage failed, never 'not computed yet'", async () => {
    const reason = "the recompute ran out of memory";
    const { card, title } = unservedAfter({ status: "error", error: reason });
    expect(card).not.toHaveTextContent("not computed yet");
    expect(screen.queryByTestId("model-one-unserved")).toBeNull();
    const failure = screen.getByTestId("model-one-failure");
    expect(failure).toHaveAttribute("data-stage", "proposals");
    expect(failure).toHaveAttribute("data-kind", "failed");
    expect(failure).toHaveTextContent(reason);
    expect(failure).toHaveTextContent(`${title} waits on it.`);
    expect(screen.queryByTestId("model-one-stopped")).toBeNull();
    expect(card).not.toHaveTextContent("You stopped");
    expect(offered()).toEqual([]);
    const run = vi.spyOn(api, "runStage").mockImplementation(() => new Promise(() => {}));
    fireEvent.click(within(failure).getByTestId("retry-proposals"));
    await waitFor(() => expect(run).toHaveBeenCalledWith(PID, "proposals"));
  });

  it("says the recompute was stopped, with a way to run it again, where a stage the proposals wait on was stopped", async () => {
    const { card, title } = unservedAfter(STOPPED, STOPPED_UPSTREAM);
    expect(card).not.toHaveTextContent("not computed yet");
    // The stop is the working table's: no computation the person could not stop is named.
    expect(card).not.toHaveTextContent(new RegExp(STAGE_LABEL.proposals!, "i"));
    expect(screen.queryByTestId("model-one-failure")).toBeNull();
    const stopped = screen.getByTestId("model-one-stopped");
    expect(stopped).toHaveTextContent(`You stopped the recompute before it finished. ${title} waits on it.`);
    expect(offered()).toEqual([]);
    // Recomputing the proposals reruns every stopped stage they wait on (core/graph.py `ensure`).
    const run = vi.spyOn(api, "runStage").mockImplementation(() => new Promise(() => {}));
    fireEvent.click(within(stopped).getByTestId("retry-proposals"));
    await waitFor(() => expect(run).toHaveBeenCalledWith(PID, "proposals"));
  });

  // The veiled columns and options come from an older result that does serve them: a recompute
  // that fails or is stopped does not restart by itself, so the veil offers its way forward.
  const veiledAfter = (status: NotFresh, upstream: Record<string, Record<string, unknown>> = {}) => {
    const { view, proposals } = servedIn("nhanes-inference");
    mount("set_model_sequence", { view: withStages(view, upstream), proposals, status });
    const veil = screen.getByTestId("model-one-veil");
    const retry = within(veil).queryByTestId("retry-proposals");
    // The way forward sits outside the inert body it would otherwise share.
    if (retry) expect(retry.closest("[inert]")).toBeNull();
    return { veil, retry };
  };

  it("offers to try again from the veil where the proposals recompute failed", async () => {
    const { veil, retry } = veiledAfter({ status: "error", error: "the recompute ran out of memory" });
    expect(veil).toHaveAttribute("data-veil", "failed");
    expect(retry).toHaveTextContent("Try again");
    expect(screen.getByTestId("options-model-one").closest("[inert]")).not.toBeNull();
    const run = vi.spyOn(api, "runStage").mockImplementation(() => new Promise(() => {}));
    fireEvent.click(retry!);
    await waitFor(() => expect(run).toHaveBeenCalledWith(PID, "proposals"));
  });

  it("offers to recompute from the veil where a stage the proposals wait on was stopped", async () => {
    const { veil, retry } = veiledAfter(STOPPED, STOPPED_UPSTREAM);
    expect(veil).toHaveAttribute("data-veil", "stopped");
    expect(retry).toHaveTextContent("Recompute");
    const run = vi.spyOn(api, "runStage").mockImplementation(() => new Promise(() => {}));
    fireEvent.click(retry!);
    await waitFor(() => expect(run).toHaveBeenCalledWith(PID, "proposals"));
  });

  it("offers no way forward from the veil while the proposals recompute, nor once it is fresh", async () => {
    expect(veiledAfter({ status: "running" }).retry).toBeNull();
    expect(screen.getByTestId("model-one-veil")).toHaveAttribute("data-veil", "recomputing");
    cleanup();
    const { view, proposals } = servedIn("nhanes-inference");
    mount("set_model_sequence", { view, proposals });
    expect(screen.getByTestId("model-one-veil")).toHaveAttribute("data-veil", "fresh");
    expect(screen.queryByTestId("retry-proposals")).toBeNull();
  });

  it("says the columns are not computed yet, and names no failure, while the proposals stage computes its first result", () => {
    const { view } = servedIn("nhanes-inference");
    mount("set_model_sequence", { view, proposals: null, stale: true });
    expect(screen.getByTestId("model-one-unserved")).toHaveTextContent("not computed yet");
    expect(screen.queryByTestId("model-one-failure")).toBeNull();
    expect(screen.queryByRole("alert")).toBeNull();
  });
});

// ── how many tests are accounted for ────────────────────────────────────────

describe("the multiplicity card", () => {
  it("offers exactly the model's methods, in its order", () => {
    mount("set_multiplicity");
    expect(offered()).toEqual(field("set_multiplicity", "method").enum!.map((m) => `method:${m}`));
  });

  it.each(field("set_multiplicity", "method").enum!.map((m) => [String(m)] as const))(
    "posts %s as the model takes it, unattested",
    async (method) => {
      const { decide } = mount("set_multiplicity");
      press(`method:${method}`);
      const sent = await posted(decide);
      expect(sent).toEqual({ kind: "set_multiplicity", method, acknowledged: defaultsOf("set_multiplicity").acknowledged });
      expect(problems(MODEL.set_multiplicity, sent)).toEqual([]);
    },
  );

  it("takes the engine's own way forward from its refusal: the answer kept, attested", async () => {
    const exit = { kind: "set_multiplicity", method: "none", acknowledged: true } as Decision;
    const refusal: Refusal = {
      error: {
        code: "family_without_multiplicity",
        message: "A family of study factors without multiplicity control is blocked until it is recorded as a limitation.",
        exits: [{ label: "Keep it, recorded as a limitation", decision: exit }],
      },
    };
    const { decide } = mount("set_multiplicity");
    decide.mockRejectedValueOnce(new RefusalError(refusal));
    press("method:none");
    const note = await screen.findByTestId("refusal");
    expect(note).toHaveTextContent(refusal.error.message);
    fireEvent.click(within(note).getByTestId("refusal-exit"));
    expect(await posted(decide, 2)).toEqual(exit);
    expect(problems(MODEL.set_multiplicity, exit)).toEqual([]);
  });

  it("marks the method recorded once the line is answered", () => {
    mount("set_multiplicity", {
      state: { multiplicity: { method: "stated_count", acknowledged: false } },
      line: { status: "answered", decision_id: "r2" },
    });
    expect(screen.getByTestId("option-method:stated_count")).toHaveAttribute("data-recorded");
    expect(document.querySelectorAll("[data-recorded]")).toHaveLength(1);
  });

  it("marks nothing recorded on a line set for the person", () => {
    mount("set_multiplicity", { state: { multiplicity: { method: "stated_count", acknowledged: false } } });
    expect(document.querySelectorAll("[data-recorded]")).toHaveLength(0);
  });
});

// ── the regression calibration question ─────────────────────────────────────

describe("the regression calibration card", () => {
  const nBoot = field("set_measurement_error", "n_boot");

  it("offers exactly the model's methods, in its order", () => {
    mount("set_measurement_error");
    expect(offered()).toEqual(field("set_measurement_error", "method").enum!.map((m) => `method:${m}`));
  });

  it("starts the bootstrap's refits at the model's default, within the model's bounds", () => {
    mount("set_measurement_error");
    const input = screen.getByTestId("field-n_boot") as HTMLInputElement;
    expect(input.value).toBe(String(nBoot.default));
    expect(input.getAttribute("min")).toBe(String(nBoot.minimum));
    expect(input.getAttribute("max")).toBe(String(nBoot.maximum));
  });

  it("posts regression calibration with the refits typed and every other field at the model's default", async () => {
    const typed = nBoot.minimum! + 1;
    const { decide } = mount("set_measurement_error");
    fireEvent.change(screen.getByTestId("field-n_boot"), { target: { value: String(typed) } });
    press("method:regression_calibration");
    const sent = await posted(decide);
    expect(sent).toEqual({
      kind: "set_measurement_error",
      method: "regression_calibration",
      ...defaultsOf("set_measurement_error"),
      exposures: [],
      n_boot: typed,
      adjustment: null,
    });
    expect(problems(MODEL.set_measurement_error, sent)).toEqual([]);
  });

  it("posts no calibration with the model's defaults, whatever refits are typed", async () => {
    const { decide } = mount("set_measurement_error");
    fireEvent.change(screen.getByTestId("field-n_boot"), { target: { value: String(nBoot.minimum! + 1) } });
    press("method:none");
    const sent = await posted(decide);
    expect(sent).toEqual({ kind: "set_measurement_error", method: "none", ...defaultsOf("set_measurement_error"), exposures: [], adjustment: null });
    expect(problems(MODEL.set_measurement_error, sent)).toEqual([]);
  });

  it("does not send refits outside the model's bounds, and names the field and its range at the option", async () => {
    const { decide } = mount("set_measurement_error");
    fireEvent.change(screen.getByTestId("field-n_boot"), { target: { value: String(nBoot.minimum! - 1) } });
    const option = screen.getByTestId("option-method:regression_calibration");
    press("method:regression_calibration");
    await settle();
    expect(decide).not.toHaveBeenCalled();
    const note = screen.getByTestId("blocked");
    expect(option.nextElementSibling).toContainElement(note);
    const label = screen.getByTestId("field-n_boot").closest("label")!.textContent!;
    expect(note).toHaveTextContent(label);
    expect(note).toHaveTextContent(nBoot.minimum!.toLocaleString("en-US"));
    expect(note).toHaveTextContent(nBoot.maximum!.toLocaleString("en-US"));
  });

  it("still sends no calibration while the refits typed are out of bounds, since it does not read them", async () => {
    const { decide } = mount("set_measurement_error");
    fireEvent.change(screen.getByTestId("field-n_boot"), { target: { value: String(nBoot.maximum! + 1) } });
    press("method:none");
    const sent = await posted(decide);
    expect(sent).toMatchObject({ method: "none", n_boot: nBoot.default });
    expect(screen.queryByTestId("blocked")).toBeNull();
  });

  it("starts at what is recorded once the line is answered, and marks it", () => {
    const recorded = { method: "regression_calibration", exposures: [], n_boot: nBoot.maximum, interval: "whole_chain_bootstrap", adjustment: ["age"] };
    mount("set_measurement_error", { state: { measurement_error: recorded }, line: { status: "answered", decision_id: "r3" } });
    expect((screen.getByTestId("field-n_boot") as HTMLInputElement).value).toBe(String(nBoot.maximum));
    expect(screen.getByTestId("option-method:regression_calibration")).toHaveAttribute("data-recorded");
    expect(screen.getByTestId("option-method:none")).not.toHaveAttribute("data-recorded");
  });

  it("starts at the model's default and marks nothing recorded on an open line, whatever calibration the state holds", () => {
    // A calibration declared under another adjustment set is asked again while the state keeps it
    // (core/quest.py `_calibration_is_current`), so it is no answer to show.
    expect(nBoot.maximum).not.toBe(nBoot.default);
    const held = { method: "regression_calibration", exposures: [], n_boot: nBoot.maximum, interval: "whole_chain_bootstrap", adjustment: ["age"] };
    mount("set_measurement_error", { state: { measurement_error: held } });
    expect(screen.getByTestId("declaration-set_measurement_error")).toHaveAttribute("data-status", "open");
    expect((screen.getByTestId("field-n_boot") as HTMLInputElement).value).toBe(String(nBoot.default));
    expect(document.querySelectorAll("[data-recorded]")).toHaveLength(0);
  });

  it("holds its bound, its defaults and both cards' lists to the models'", async () => {
    const load = PAYLOADS["./payloads.ts"];
    expect(load, "the cards' payloads module").toBeDefined();
    const { BOUNDS, DEFAULTS, CALIBRATION_METHODS, MULTIPLICITY_METHODS } = await load!();
    expect(BOUNDS.set_measurement_error.n_boot).toEqual({ ge: nBoot.minimum, le: nBoot.maximum });
    // The contract states the defaults it has a value for; the intakes' empty list is the model's
    // `default_factory` (every exposure the recalls measure), which it does not state.
    const { exposures, ...stated } = DEFAULTS.set_measurement_error;
    expect(stated).toEqual(defaultsOf("set_measurement_error"));
    expect(exposures).toEqual([]);
    expect([...CALIBRATION_METHODS]).toEqual(field("set_measurement_error", "method").enum);
    expect([...MULTIPLICITY_METHODS]).toEqual(field("set_multiplicity", "method").enum);
  });
});
