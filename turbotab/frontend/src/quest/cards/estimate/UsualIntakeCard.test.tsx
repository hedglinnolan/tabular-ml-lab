/**
 * The usual-intake card (the front-end audit's G08): one dietary component's usual-intake
 * distribution by the NCI method, or none, through DECLARATION_CARDS.
 *
 * On the base the usual-intake question has no card: it falls to the fallback, which records
 * nothing, so the shell lists the counted Decide "Not available yet" and never opens it, and under
 * the dietary lens with repeated recalls Models never completes. Here it records.
 *
 * The truth the card is held to is the server's, never the card's own constants: the offer as a real
 * server served it (the dietary-recalls capture in src/mocks/fixtures, whose usual-intake stage
 * offers two components of a long table), or built in the served shape where no capture holds the
 * case (a wide table's recall days, core/stages/usual_intake.py `build_offer`; an analysis the
 * stage refuses, its `refuse`; an answer on record as the state holds it, every field present);
 * every payload, list of models and default from turbotab/server/openapi.json, the contract the
 * server's own test holds to its pydantic models (core/decisions.py `SetUsualIntake`); and the
 * line's name and place from the quest contract the engine writes
 * (src/quest/fixtures/quest-contract.json).
 */
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { api, RefusalError } from "../../../api/client";
import type { UsualIntakeArtifact } from "../../../api/m3-types";
import { keys } from "../../../api/queries";
import type { Decision, ProjectView, QuestLine, Refusal } from "../../../api/schema";
import { artifactAt, expandView, viewsOf, type M3Fixture } from "../../../mocks/m3";
import { StageFocusProvider } from "../../../state/focus";
import { CONTRACT } from "../../fixtures/contract";
import { DECLARATION_CARDS } from "../predict";
import { recordable, rendererOf } from "../renderers";

const PID = "usual-intake-test";
const AT = "2026-11-03T00:00:00Z";
const KIND = "set_usual_intake";

// ── the server's contract (openapi.json) ─────────────────────────────────────

interface JsonSchema {
  type?: string;
  enum?: unknown[];
  const?: unknown;
  minimum?: number;
  maximum?: number;
  exclusiveMinimum?: number;
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
/** The model a posted decision is parsed as (the request body's variant). */
const MODEL = SCHEMAS["SetUsualIntake-Input"]!;
const MODELS = MODEL.properties!.model!.enum as string[];

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
        if (s.exclusiveMinimum !== undefined && value <= s.exclusiveMinimum) out.push(`${at}: not above ${s.exclusiveMinimum}`);
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

/** The model's defaults for every field the contract states one for. */
const STATED = Object.fromEntries(
  Object.entries(MODEL.properties!)
    .filter(([k, p]) => k !== "kind" && p.default !== undefined)
    .map(([k, p]) => [k, p.default]),
);
/** The fields the model does not require and the contract states no default for: an optional
 *  column (None in core/decisions.py) or a list of columns (an empty `default_factory`), which the
 *  card sends as the empty value of its type. */
const UNSTATED = Object.entries(MODEL.properties!)
  .filter(([k, p]) => !(MODEL.required ?? []).includes(k) && p.default === undefined)
  .map(([k, p]) => [k, p.type === "array" ? [] : null] as const);

/** An answer as the state holds it (`UsualIntakeSpec`, every field present): the fields given, the
 *  model's defaults for the rest. */
const specOf = (over: Record<string, unknown>) => ({ ...Object.fromEntries(UNSTATED), ...STATED, ...over });

/** An analysis as the stage serves it (`UsualIntakeAnalysis`): the fields given, the model's
 *  defaults for the rest (turbotab/server/openapi.json). */
type Analysis = UsualIntakeArtifact["analyses"][number];
const ANALYSIS = SCHEMAS["UsualIntakeAnalysis"]!;
const analysisOf = (over: Partial<Analysis> & Pick<Analysis, "nutrient" | "model" | "applies" | "methods">): Analysis =>
  ({
    ...Object.fromEntries(
      Object.entries(ANALYSIS.properties!)
        .filter(([, p]) => p.default !== undefined)
        .map(([k, p]) => [k, p.default]),
    ),
    ...over,
  }) as Analysis;

// ── the captured server (src/mocks/fixtures) ─────────────────────────────────

const FIXTURES = import.meta.glob<M3Fixture>("../../../mocks/fixtures/m3-dietary-recalls.json", {
  eager: true,
  import: "default",
});
const RECALLS = Object.values(FIXTURES)[0]!;
type Offer = UsualIntakeArtifact["offer"];
type Candidate = Offer["candidates"][number];

/** The last captured view whose usual-intake stage offers the distribution, and that artifact. */
function served(): { view: ProjectView; artifact: UsualIntakeArtifact; offer: Offer } {
  const views = viewsOf(RECALLS);
  for (let i = views.length - 1; i >= 0; i--) {
    const key = views[i]!.stages.usual_intake?.key ?? null;
    const artifact = key ? (artifactAt(RECALLS, "usual_intake", key) as UsualIntakeArtifact | null) : null;
    if (artifact?.offer.offered && artifact.offer.candidates.length) {
      return { view: expandView(RECALLS, views[i]!, PID, AT), artifact, offer: artifact.offer };
    }
  }
  throw new Error("the dietary-recalls capture offers no usual-intake distribution");
}

/** A wide table's offer, as the stage builds one (core/stages/usual_intake.py `build_offer`): each
 *  candidate is a group of recall-day columns, named by its first day, its days as proposed from
 *  their names (core/usual_intake.py `day_columns`, which reads `DR1TPROT` beside `DR2TPROT`). */
function wide(offer: Offer): Offer {
  const protein: Candidate = {
    column: "DR1TPROT",
    days: ["DR1TPROT", "DR2TPROT"],
    zero_share: 0,
    n_persons: 900,
    n_repeat: 720,
    suggested: "amount_only",
    why: null,
  };
  return { ...offer, format: "wide", recalls: {}, candidates: [protein] };
}

// ── mounting ────────────────────────────────────────────────────────────────

const PLACE = CONTRACT.declarations[KIND]!;

function lineOf(over: Partial<QuestLine> = {}): QuestLine {
  return {
    id: `decision:${KIND}`,
    key: KIND,
    source: "declaration",
    label: PLACE.label,
    name: PLACE.name,
    status: "open",
    counted: PLACE.counted,
    order: PLACE.order,
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

/** How the usual-intake stage stands where it is not fresh: recomputing, failed, or stopped. */
type NotFresh =
  | { status: "running" }
  | { status: "error"; error: string }
  | { status: "stale" | "idle"; cancelled: true };

/**
 * A stop as the engine serves it to the usual-intake stage, a heavy stage whose job the person can
 * stop (core/stages/__init__.py, `heavy=True`): held, the stage is stale where it has an older
 * result and idle where it has none, and cancelled either way (core/graph.py `_status`).
 */
const STOPS: NotFresh[] = [
  { status: "stale", cancelled: true },
  { status: "idle", cancelled: true },
];

/**
 * Serve the usual-intake stage: fresh, its result at the key the view reports; or not fresh
 * (recomputing unless `over` says how it stands), the view reporting the key it computes while
 * the server serves the latest older result with `fresh: false` (core/graph.py `Engine.get`).
 */
function serveStage(qc: QueryClient, view: ProjectView, artifact: UsualIntakeArtifact | null, fresh = true, over?: NotFresh) {
  const status = over?.status ?? (fresh ? "fresh" : "running");
  qc.setQueryData(keys.view(PID), {
    ...view,
    stages: {
      ...view.stages,
      usual_intake: { stage: "usual_intake", status, key: "u", fresh, updated_at: null, error: null, cancelled: false, ...over },
    },
  } as unknown as ProjectView);
  qc.setQueryData(keys.stage(PID, "usual_intake"), { stage: "usual_intake", key: fresh ? "u" : "u0", fresh, status, artifact });
}

function mount(
  opts: {
    artifact?: UsualIntakeArtifact | null;
    stale?: boolean;
    status?: NotFresh;
    /** The state's usual-intake answers, keyed by component: the fields given, the model's
     *  defaults for the rest. */
    answers?: Record<string, Record<string, unknown>>;
    /** The view's records (the line's own among them, by its id). */
    decisions?: { id: string; decision: Decision }[];
    line?: Partial<QuestLine>;
  } = {},
) {
  const at = served();
  const base = {
    ...at.view,
    state: {
      ...at.view.state,
      usual_intake: opts.answers ? Object.fromEntries(Object.entries(opts.answers).map(([k, v]) => [k, specOf(v)])) : null,
    },
    decisions: [...at.view.decisions, ...(opts.decisions ?? [])],
  } as ProjectView;
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  qc.setQueryData(keys.view(PID), base);
  serveStage(qc, base, opts.artifact === undefined ? at.artifact : opts.artifact, !opts.stale && !opts.status, opts.status);
  const view = qc.getQueryData<ProjectView>(keys.view(PID))!;
  qc.setQueryData(keys.columns(PID), Object.keys(view.state.roles ?? {}).map((name) => ({ name })));
  const decide = vi.spyOn(api, "decide").mockResolvedValue(view);
  // The engine's preview of an option pointed at says nothing here: the words are its, not these tests'.
  vi.spyOn(api, "preview").mockImplementation(() => new Promise(() => {}));
  const Card = DECLARATION_CARDS[KIND];
  if (!Card) throw new Error(`no card for ${KIND}`);
  render(
    <QueryClientProvider client={qc}>
      <StageFocusProvider>
        <Card pid={PID} line={lineOf(opts.line)} />
      </StageFocusProvider>
    </QueryClientProvider>,
  );
  return { decide, qc, ...at };
}

/** A press as a mouse makes it: a click chooses the option, and the card's own button records it. */
const press = (option: string) => {
  fireEvent.click(screen.getByTestId(`option-${option}`));
  fireEvent.click(screen.getByTestId("card-record"));
};
const choose = (component: string) => fireEvent.change(screen.getByTestId("field-nutrient"), { target: { value: component } });
const offered = () => Array.from(document.querySelectorAll("[role=option]")).map((o) => o.getAttribute("data-key"));
const settle = () => new Promise((resolve) => setTimeout(resolve, 20));
const posted = async (decide: ReturnType<typeof vi.spyOn>, n = 1): Promise<Decision> => {
  await waitFor(() => expect(decide).toHaveBeenCalledTimes(n));
  return decide.mock.calls[n - 1]![1] as Decision;
};
const componentOptions = () =>
  Array.from((screen.getByTestId("field-nutrient") as HTMLSelectElement).options).map((o) => [o.value, o.textContent]);

afterEach(() => {
  vi.restoreAllMocks();
});

// ── the shell ───────────────────────────────────────────────────────────────

describe("the usual-intake question the shell can answer (G08)", () => {
  it("is a counted Decide in Models, drawn through a card that records", () => {
    expect(PLACE).toMatchObject({ stage: "models", label: "Decide", counted: true });
    expect(rendererOf({ source: "declaration", key: KIND }, DECLARATION_CARDS)).toBe("declaration");
    expect(recordable({ source: "declaration", key: KIND }, DECLARATION_CARDS)).toBe(true);
  });
});

// ── the card ────────────────────────────────────────────────────────────────

describe("the usual-intake card, from the usual-intake stage's offer", () => {
  it("says what the distribution estimates, in the engine's words, and lists the offer's components in its order, none chosen", () => {
    const { offer } = mount();
    const card = screen.getByTestId(`declaration-${KIND}`);
    expect(within(card).getByRole("heading")).toHaveTextContent(new RegExp(`^${PLACE.name}$`, "i"));
    expect(within(card).getByTestId("usual-intake-estimand")).toHaveTextContent(offer.estimand);
    expect(offer.format).toBe("long");
    expect(offer.candidates.length).toBeGreaterThan(1);
    const options = componentOptions();
    // A component is the person's to choose: none is chosen for them.
    expect(options[0]![0]).toBe("");
    expect((screen.getByTestId("field-nutrient") as HTMLSelectElement).value).toBe("");
    // A long table's component is its column.
    expect(options.slice(1)).toEqual(offer.candidates.map((c) => [c.column, c.column]));
  });

  it("offers exactly the models the server's model takes", () => {
    mount();
    expect([...offered()].sort()).toEqual(MODELS.map((m) => `model:${m}`).sort());
  });

  it("records nothing, and says what is in the way, where a model is pressed before a component is chosen", async () => {
    const { decide } = mount();
    press("model:amount_only");
    await settle();
    expect(decide).not.toHaveBeenCalled();
    const note = screen.getByTestId("blocked");
    const label = screen.getByTestId("field-nutrient").closest("label")!.querySelector("span")!.textContent!;
    expect(note).toHaveTextContent(label);
    expect(screen.getByTestId("field-nutrient")).toHaveAttribute("aria-invalid", "true");
  });

  it("records the model for the component chosen: a long table's column, no day columns, and the model's defaults for all it does not ask", async () => {
    const { decide, offer } = mount();
    const component = offer.candidates[offer.candidates.length - 1]!;
    choose(component.column);
    press("model:amount_only");
    const sent = await posted(decide);
    expect(problems(MODEL, sent)).toEqual([]);
    expect(sent).toEqual({
      kind: KIND,
      nutrient: component.column,
      model: "amount_only",
      ...Object.fromEntries(UNSTATED),
      ...STATED,
    });
    expect(screen.queryByTestId("blocked")).toBeNull();
  });

  it("records a wide table's component with the recall days the offer proposes, which the component's name lists", async () => {
    const at = served();
    const offer = wide(at.offer);
    const { decide } = mount({ artifact: { ...at.artifact, offer } });
    const protein = offer.candidates[0]!;
    const [, label] = componentOptions().find(([value]) => value === protein.column)!;
    for (const day of protein.days) expect(label).toContain(day);
    choose(protein.column);
    press("model:two_part");
    const sent = await posted(decide);
    expect(problems(MODEL, sent)).toEqual([]);
    expect(sent).toMatchObject({ kind: KIND, nutrient: protein.column, model: "two_part", days: protein.days });
  });

  it("leaves usual intake out as the engine's own exit does: the component, no model, and no day columns", async () => {
    const at = served();
    const offer = wide(at.offer);
    const { decide } = mount({ artifact: { ...at.artifact, offer } });
    choose(offer.candidates[0]!.column);
    press("model:none");
    const sent = await posted(decide);
    expect(problems(MODEL, sent)).toEqual([]);
    // core/usual_intake.py `leave_out`: {"kind", "nutrient", "model": "none"}, every other field the model's default.
    expect(sent).toEqual({ kind: KIND, nutrient: offer.candidates[0]!.column, model: "none", ...Object.fromEntries(UNSTATED), ...STATED });
  });

  it("says the engine's reason for the model it ranks first once a component is chosen, and nothing before", () => {
    const { offer } = mount();
    const component = offer.candidates[0]!;
    expect(component.why).toBeTruthy();
    expect(screen.queryByTestId("usual-intake-why")).toBeNull();
    choose(component.column);
    expect(screen.getByTestId("usual-intake-why")).toHaveTextContent(component.why!);
  });

  it("chooses the component of the line's record once the line is answered, and marks its model", () => {
    const { offer } = served();
    const component = offer.candidates[offer.candidates.length - 1]!;
    const decision = { kind: KIND, nutrient: component.column, model: "two_part" } as unknown as Decision;
    mount({
      answers: { [component.column]: { model: "two_part", days: [] } },
      decisions: [{ id: "r9", decision }],
      line: { status: "answered", decision_id: "r9" },
    });
    expect((screen.getByTestId("field-nutrient") as HTMLSelectElement).value).toBe(component.column);
    expect(screen.getByTestId("option-model:two_part")).toHaveAttribute("data-recorded");
    expect(screen.getByTestId("option-model:amount_only")).not.toHaveAttribute("data-recorded");
  });

  it("marks nothing recorded for a component with no answer, on an answered line", () => {
    const { offer } = served();
    const [first, second] = offer.candidates;
    const decision = { kind: KIND, nutrient: first!.column, model: "amount_only" } as unknown as Decision;
    mount({
      answers: { [first!.column]: { model: "amount_only", days: [] } },
      decisions: [{ id: "r9", decision }],
      line: { status: "answered", decision_id: "r9" },
    });
    expect(screen.getByTestId("option-model:amount_only")).toHaveAttribute("data-recorded");
    choose(second!.column);
    expect(document.querySelectorAll("[data-recorded]")).toHaveLength(0);
  });

  it("keeps a component on record that the offer no longer lists, with its record marked", async () => {
    const { offer } = served();
    expect(offer.candidates.map((c) => c.column)).not.toContain("fiber_g");
    const decision = { kind: KIND, nutrient: "fiber_g", model: "amount_only" } as unknown as Decision;
    const { decide } = mount({
      answers: { fiber_g: { model: "amount_only", days: [] } },
      decisions: [{ id: "r9", decision }],
      line: { status: "answered", decision_id: "r9" },
    });
    expect(componentOptions().map(([value]) => value)).toContain("fiber_g");
    expect(screen.getByTestId("option-model:amount_only")).toHaveAttribute("data-recorded");
    press("model:none");
    expect(await posted(decide)).toMatchObject({ nutrient: "fiber_g", model: "none" });
  });

  // A wide table's answer on record that the offer no longer lists (its days renamed, or no longer
  // numeric): the label the engine keys it by is the first of its days, as the offer names one.
  const ENERGY = { nutrient: "DR1TKCAL", days: ["DR1TKCAL", "DR2TKCAL"] };
  const onRecord = (answer: { nutrient: string; days: string[] }, model = "amount_only") => {
    const at = served();
    const offer = wide(at.offer);
    const decision = { kind: KIND, nutrient: answer.nutrient, model, days: answer.days } as unknown as Decision;
    const mounted = mount({
      artifact: { ...at.artifact, offer },
      answers: { [answer.nutrient]: { model, days: answer.days } },
      decisions: [{ id: "r9", decision }],
      line: { status: "answered", decision_id: "r9" },
    });
    return { ...mounted, offer };
  };

  it.each(["amount_only", "two_part"])("records a wide component on record that the offer no longer lists with its days on record (%s)", async (model) => {
    const { decide, offer } = onRecord(ENERGY);
    expect(offer.candidates.map((c) => c.column)).not.toContain(ENERGY.nutrient);
    expect((screen.getByTestId("field-nutrient") as HTMLSelectElement).value).toBe(ENERGY.nutrient);
    press(`model:${model}`);
    const sent = await posted(decide);
    expect(problems(MODEL, sent)).toEqual([]);
    expect(sent).toMatchObject({ kind: KIND, nutrient: ENERGY.nutrient, model, days: ENERGY.days });
  });

  it("leaves a wide component on record that the offer no longer lists out with no day columns, as the engine's own exit does", async () => {
    const { decide } = onRecord(ENERGY);
    press("model:none");
    const sent = await posted(decide);
    expect(problems(MODEL, sent)).toEqual([]);
    expect(sent).toMatchObject({ kind: KIND, nutrient: ENERGY.nutrient, model: "none", days: [] });
  });

  it("names a component on record by its own label beside its days where the label is not its first day, so it never reads as the offer's own", () => {
    // The engine keys a wide component by any label (core/decisions.py `SetUsualIntake`), as its own
    // NHANES test records protein: DRxTPROT, with the days the offer names DR1TPROT by.
    const protein = wide(served().offer).candidates[0]!;
    const label = "DRxTPROT";
    expect(label).not.toBe(protein.column);
    onRecord({ nutrient: label, days: [...protein.days] });
    const options = componentOptions().slice(1);
    expect(options.map(([value]) => value)).toEqual([protein.column, label]);
    const names = options.map(([, name]) => name);
    expect(new Set(names).size).toBe(names.length);
    const [, named] = options.find(([value]) => value === label)!;
    expect(named).toContain(label);
    for (const day of protein.days) expect(named).toContain(day);
  });

  it("names the offer's component as it does with no record where the answer on record is keyed by its own column", () => {
    const offer = wide(served().offer);
    const protein = offer.candidates[0]!;
    mount({ artifact: { ...served().artifact, offer } });
    const unanswered = componentOptions();
    // The offer names the component by its first day, so its days alone name it.
    expect(unanswered[1]).toEqual([protein.column, protein.days.join(", ")]);
    cleanup();
    vi.restoreAllMocks();
    onRecord({ nutrient: protein.column, days: [...protein.days] });
    expect(componentOptions()).toEqual(unanswered);
    expect(componentOptions()).toHaveLength(2);
  });

  it("says why the distribution is not offered where the offer is not made, and nothing of it where it is", () => {
    const at = served();
    expect(at.offer.reason).toBeNull();
    mount();
    expect(screen.queryByTestId("usual-intake-unoffered")).toBeNull();
  });

  it("says the offer's reason where the distribution is no longer offered and an answer is on record", () => {
    // The line stays once an answer is recorded (core/quest.py `_usual_intake_applies`); the
    // stage's reason is the engine's (core/usual_intake.py, the repeats read as time points).
    const at = served();
    const reason = viewsOf(RECALLS)
      .map((v) => artifactAt(RECALLS, "usual_intake", v.stages.usual_intake?.key ?? null) as UsualIntakeArtifact | null)
      .find((a) => a && !a.offer.offered && a.offer.reason)!.offer.reason!;
    const offer = { ...at.offer, offered: false, reason, candidates: [] };
    const { offer: before } = at;
    const decision = { kind: KIND, nutrient: before.candidates[0]!.column, model: "amount_only" } as unknown as Decision;
    mount({
      artifact: { ...at.artifact, offer },
      answers: { [before.candidates[0]!.column]: { model: "amount_only", days: [] } },
      decisions: [{ id: "r9", decision }],
      line: { status: "answered", decision_id: "r9" },
    });
    expect(screen.getByTestId("usual-intake-unoffered")).toHaveTextContent(reason);
  });

  it("shows no distribution the stage holds, and no refusal, for an answer whose analysis applies", () => {
    // After the lock the stage serves each answer's distribution (core/estimand.py `withhold` keeps
    // only the offer and the refused analyses before it); the card asks the question and never
    // shows one.
    const at = served();
    const nutrient = at.offer.candidates[0]!.column;
    const methods = "The usual-intake distribution of protein was estimated by the NCI method.";
    const estimate = { value: 81.25, se: 1.5, ci_low: 78.31, ci_high: 84.19 };
    const analysis = analysisOf({
      nutrient,
      model: "amount_only",
      applies: true,
      methods,
      percentiles: { "50": estimate },
      mean: estimate,
      exits: [{ label: "Report the share below it as a plain share", decision: { kind: KIND, nutrient, model: "amount_only" } }],
    });
    mount({
      artifact: { ...at.artifact, analyses: [analysis] },
      answers: { [nutrient]: { model: "amount_only" } },
      decisions: [{ id: "r9", decision: { kind: KIND, nutrient, model: "amount_only" } as unknown as Decision }],
      line: { status: "answered", decision_id: "r9" },
    });
    const card = screen.getByTestId(`declaration-${KIND}`);
    expect(card).not.toHaveTextContent(methods);
    for (const n of Object.values(estimate)) expect(card).not.toHaveTextContent(String(n));
    expect(screen.queryByTestId("usual-intake-refused")).toBeNull();
    expect(screen.queryByTestId("usual-intake-refused-exit")).toBeNull();
  });

  it("veils the components and the models, inert, while the stage's result is not fresh", () => {
    mount({ stale: true });
    const veil = screen.getByTestId("usual-intake-veil");
    expect(veil.getAttribute("data-veil")).not.toBe("fresh");
    for (const id of ["usual-intake-estimand", "field-nutrient", "options-usual-intake", "card-record"]) {
      expect(screen.getByTestId(id).closest("[inert]"), id).not.toBeNull();
    }
  });

  it("leaves them live once the stage's result is fresh", async () => {
    const { qc, view, artifact } = mount({ stale: true });
    act(() => serveStage(qc, { ...view, state: { ...view.state, usual_intake: null } } as ProjectView, artifact));
    await waitFor(() => expect(screen.getByTestId("usual-intake-veil")).toHaveAttribute("data-veil", "fresh"));
    for (const id of ["usual-intake-estimand", "field-nutrient", "options-usual-intake", "card-record"]) {
      expect(screen.getByTestId(id).closest("[inert]"), id).toBeNull();
    }
  });

  it("names the stage that did not finish, its reason and its retry where the stage failed with no offer to show", async () => {
    const reason = "the recompute ran out of memory";
    mount({ artifact: null, status: { status: "error", error: reason } });
    const card = screen.getByTestId(`declaration-${KIND}`);
    expect(card).not.toHaveTextContent("not computed yet");
    const failure = screen.getByTestId("usual-intake-failure");
    expect(failure).toHaveAttribute("data-stage", "usual_intake");
    expect(failure).toHaveTextContent(reason);
    expect(screen.queryByTestId("usual-intake-stopped")).toBeNull();
    expect(card).not.toHaveTextContent("You stopped");
    expect(offered()).toEqual([]);
    const run = vi.spyOn(api, "runStage").mockImplementation(() => new Promise(() => {}));
    fireEvent.click(within(failure).getByTestId("retry-usual_intake"));
    await waitFor(() => expect(run).toHaveBeenCalledWith(PID, "usual_intake"));
  });

  it("says the components are not computed yet, and offers nothing, where the stage serves no offer and has not failed", async () => {
    const { decide } = mount({ artifact: null, status: { status: "running" } });
    expect(screen.getByTestId("usual-intake-unserved")).toBeInTheDocument();
    expect(screen.queryByTestId("usual-intake-failure")).toBeNull();
    expect(screen.queryByTestId("usual-intake-stopped")).toBeNull();
    expect(screen.queryByTestId("retry-usual_intake")).toBeNull();
    expect(offered()).toEqual([]);
    await settle();
    expect(decide).not.toHaveBeenCalled();
  });

  it.each(STOPS)("says the recompute was stopped, with a way to run it again, where the stage was stopped with no offer to show ($status)", async (stop) => {
    mount({ artifact: null, status: stop });
    const card = screen.getByTestId(`declaration-${KIND}`);
    const title = within(card).getByRole("heading").textContent!;
    expect(card).not.toHaveTextContent("not computed yet");
    expect(screen.queryByTestId("usual-intake-unserved")).toBeNull();
    expect(screen.queryByTestId("usual-intake-failure")).toBeNull();
    const stopped = screen.getByTestId("usual-intake-stopped");
    expect(stopped).toHaveTextContent(`You stopped the recompute before it finished. ${title} waits on it.`);
    expect(offered()).toEqual([]);
    const retry = within(stopped).getByTestId("retry-usual_intake");
    expect(retry).toHaveTextContent("Recompute");
    const run = vi.spyOn(api, "runStage").mockImplementation(() => new Promise(() => {}));
    fireEvent.click(retry);
    await waitFor(() => expect(run).toHaveBeenCalledWith(PID, "usual_intake"));
  });

  it("offers to recompute from the veil, outside its inert body, where the stage was stopped with an older offer on screen", async () => {
    mount({ status: { status: "stale", cancelled: true } });
    const veil = screen.getByTestId("usual-intake-veil");
    expect(veil).toHaveAttribute("data-veil", "stopped");
    expect(screen.queryByTestId("usual-intake-stopped")).toBeNull();
    expect(screen.getByTestId("options-usual-intake").closest("[inert]")).not.toBeNull();
    const retry = within(veil).getByTestId("retry-usual_intake");
    expect(retry).toHaveTextContent("Recompute");
    expect(retry.closest("[inert]")).toBeNull();
    const run = vi.spyOn(api, "runStage").mockImplementation(() => new Promise(() => {}));
    fireEvent.click(retry);
    await waitFor(() => expect(run).toHaveBeenCalledWith(PID, "usual_intake"));
  });

  it("offers no way forward from the veil while the stage recomputes, nor once it is fresh", () => {
    mount({ status: { status: "running" } });
    expect(screen.getByTestId("usual-intake-veil")).toHaveAttribute("data-veil", "recomputing");
    expect(screen.queryByTestId("retry-usual_intake")).toBeNull();
    cleanup();
    mount();
    expect(screen.getByTestId("usual-intake-veil")).toHaveAttribute("data-veil", "fresh");
    expect(screen.queryByTestId("retry-usual_intake")).toBeNull();
  });
});

// ── an answer on record whose analysis the stage refuses (A2) ───────────────

describe("the usual-intake card, where the stage refuses the analysis of an answer on record", () => {
  /**
   * The captured long table's energy is reported on every recall (its candidate's zero share is 0),
   * and the card offers the two-part model for it. Recorded, the stage refuses that analysis and
   * serves the refusal before the lock (core/estimand.py `withhold`), as it builds it
   * (core/stages/usual_intake.py `refuse` and `method_exits`; core/usual_intake.py `answer` and
   * `leave_out`; the words as core/methods/usual_intake.py raises `no_zero_day`).
   */
  function refusedEnergy() {
    const at = served();
    const energy = at.offer.candidates.find((c) => c.zero_share === 0)!;
    const nutrient = energy.column;
    const record = specOf({ model: "two_part" });
    const why =
      "No recall reports a day without it, so the probability part of the two-part model has nothing to estimate; " +
      "the amount-only model is the one for an intake reported every day.";
    const amountOnly = { kind: KIND, nutrient, ...record, model: "amount_only" } as unknown as Decision;
    const leaveOut = { kind: KIND, nutrient, model: "none" } as unknown as Decision;
    const analysis = analysisOf({
      nutrient,
      model: "two_part",
      applies: false,
      refused: why,
      format: "long",
      exits: [
        { label: "Use the amount-only model, for a component reported every day", decision: amountOnly },
        { label: "Leave usual intake out", decision: leaveOut },
      ],
      methods: `No usual-intake distribution was estimated for \`${nutrient}\`: ${why}`,
    });
    const mounted = mount({
      artifact: { ...at.artifact, analyses: [analysis] },
      answers: { [nutrient]: { model: "two_part" } },
      decisions: [{ id: "r9", decision: { kind: KIND, nutrient, model: "two_part" } as unknown as Decision }],
      line: { status: "answered", decision_id: "r9" },
    });
    return { ...mounted, analysis, nutrient, amountOnly, leaveOut };
  }

  it("says the refusal in the engine's words, naming the component, with each way forward", () => {
    const { analysis, nutrient } = refusedEnergy();
    expect(analysis.refused).toBeTruthy();
    const refused = screen.getByTestId("usual-intake-refused");
    expect(refused).toHaveAttribute("data-nutrient", nutrient);
    expect(refused).toHaveTextContent(analysis.refused!);
    expect(refused).toHaveTextContent(nutrient);
    const exits = within(refused).getAllByTestId("usual-intake-refused-exit");
    expect(exits.map((e) => e.textContent)).toEqual(analysis.exits.map((x) => x.label));
  });

  it.each([0, 1])("records the way forward pressed, as the engine serves it (exit %i)", async (i) => {
    const { decide, analysis } = refusedEnergy();
    fireEvent.click(within(screen.getByTestId("usual-intake-refused")).getAllByTestId("usual-intake-refused-exit")[i]!);
    const sent = await posted(decide);
    expect(sent).toEqual(analysis.exits[i]!.decision);
    expect(problems(MODEL, sent)).toEqual([]);
  });

  it("says a way forward that names no decision in words, and offers no press for it", () => {
    // core/stages/usual_intake.py `method_exits`, a column with no amounts: name it (a question).
    const at = served();
    const nutrient = at.offer.candidates[0]!.column;
    const ask = { label: "Name the column (or recall days) that hold the component's amounts", decision: null };
    const leaveOut = { label: "Leave usual intake out", decision: { kind: KIND, nutrient, model: "none" } };
    const why = `\`${nutrient}\` holds no amounts on the analysis rows.`;
    mount({
      artifact: {
        ...at.artifact,
        analyses: [
          analysisOf({
            nutrient,
            model: "amount_only",
            applies: false,
            refused: why,
            exits: [ask, leaveOut],
            methods: `No usual-intake distribution was estimated for \`${nutrient}\`: ${why}`,
          }),
        ],
      },
      answers: { [nutrient]: { model: "amount_only" } },
      decisions: [{ id: "r9", decision: { kind: KIND, nutrient, model: "amount_only" } as unknown as Decision }],
      line: { status: "answered", decision_id: "r9" },
    });
    const refused = screen.getByTestId("usual-intake-refused");
    expect(refused).toHaveTextContent(ask.label);
    expect(within(refused).getAllByTestId("usual-intake-refused-exit").map((e) => e.textContent)).toEqual([leaveOut.label]);
  });

  it("answers a way forward the engine refuses in its words, at the refusal", async () => {
    const { decide } = refusedEnergy();
    const refusal = {
      error: { code: "usual_intake_under_prediction", message: "Usual intake is not offered under prediction.", exits: [] },
    } as unknown as Refusal;
    decide.mockRejectedValueOnce(new RefusalError(refusal));
    const refused = screen.getByTestId("usual-intake-refused");
    fireEvent.click(within(refused).getAllByTestId("usual-intake-refused-exit")[0]!);
    await posted(decide);
    expect(await within(refused).findByTestId("refusal")).toHaveTextContent(refusal.error.message);
  });

  it("keeps the refusal veiled and inert while the stage recomputes", () => {
    const at = served();
    const nutrient = at.offer.candidates[0]!.column;
    const why = "No recall reports a day without it.";
    mount({
      stale: true,
      artifact: {
        ...at.artifact,
        analyses: [
          analysisOf({
            nutrient,
            model: "two_part",
            applies: false,
            refused: why,
            exits: [{ label: "Leave usual intake out", decision: { kind: KIND, nutrient, model: "none" } }],
            methods: `No usual-intake distribution was estimated for \`${nutrient}\`: ${why}`,
          }),
        ],
      },
      answers: { [nutrient]: { model: "two_part" } },
    });
    expect(screen.getByTestId("usual-intake-refused").closest("[inert]")).not.toBeNull();
  });
});

// ── a press on a component on record (A3) ───────────────────────────────────

describe("the usual-intake card, pressed on a component with an answer on record", () => {
  // A record made outside the card (the API, the headless engine), as the NHANES journey records
  // protein: its weekend days, an EAR cut-off answered as every participant's, more replicates.
  const PROTEIN = {
    model: "amount_only",
    days: ["DR1TPROT", "DR2TPROT"],
    weekend: ["DR1DAY", "DR2DAY"],
    cutoff: 46,
    cutoff_kind: "EAR",
    ear_for_all: true,
    n_boot: 500,
  };
  const recorded = (record: Record<string, unknown>) => {
    const at = served();
    const offer = wide(at.offer);
    const nutrient = offer.candidates[0]!.column;
    const mounted = mount({
      artifact: { ...at.artifact, offer },
      answers: { [nutrient]: record },
      decisions: [{ id: "r9", decision: { kind: KIND, nutrient, ...record } as unknown as Decision }],
      line: { status: "answered", decision_id: "r9" },
    });
    return { ...mounted, nutrient, offer };
  };

  it("keeps every field of the record the card does not ask, changing the model alone, as the engine changes one field of a record", async () => {
    // core/usual_intake.py `answer(nutrient, spec, model=...)`: the record with the model changed.
    const { decide, nutrient } = recorded(PROTEIN);
    press("model:two_part");
    const sent = await posted(decide);
    expect(problems(MODEL, sent)).toEqual([]);
    expect(sent).toEqual({ kind: KIND, nutrient, ...specOf(PROTEIN), model: "two_part" });
  });

  it("posts the days on record, which the component's name lists, where they are not the days the offer proposes", async () => {
    const days = ["DR1TPROT", "DR2TPROT", "DR3TPROT"];
    const { decide, nutrient, offer } = recorded({ ...PROTEIN, days });
    expect(offer.candidates[0]!.days).not.toEqual(days);
    const [, name] = componentOptions().find(([value]) => value === nutrient)!;
    expect(name).toBe(days.join(", "));
    press("model:two_part");
    expect(await posted(decide)).toMatchObject({ nutrient, days });
  });

  it("leaves usual intake out as the engine's own exit does, whatever the record holds", async () => {
    // core/usual_intake.py `leave_out`: {"kind", "nutrient", "model": "none"}, every other field the
    // model's default.
    const { decide, nutrient } = recorded(PROTEIN);
    press("model:none");
    const sent = await posted(decide);
    expect(sent).toEqual({ kind: KIND, nutrient, model: "none", ...Object.fromEntries(UNSTATED), ...STATED });
  });

  it("posts the model's defaults and the offer's days for a component with no answer, whatever another's record holds", async () => {
    const at = served();
    const [first, second] = at.offer.candidates;
    const { decide } = mount({
      answers: { [first!.column]: { ...PROTEIN, days: [] } },
      decisions: [{ id: "r9", decision: { kind: KIND, nutrient: first!.column, model: "amount_only" } as unknown as Decision }],
      line: { status: "answered", decision_id: "r9" },
    });
    choose(second!.column);
    press("model:amount_only");
    const sent = await posted(decide);
    expect(sent).toEqual({
      kind: KIND,
      nutrient: second!.column,
      model: "amount_only",
      ...Object.fromEntries(UNSTATED),
      ...STATED,
      days: second!.days,
    });
  });
});
