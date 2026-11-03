/**
 * The batch card (the front-end audit's G08): how a batch column is handled, through
 * DECLARATION_CARDS.
 *
 * On the base the batch question has no card: it falls to the fallback, which records nothing, so
 * the shell lists the counted Decide "Not available yet" and never opens it, and under an assay
 * lens with a column named as a batch, a run or a plate Models never completes. Here it records.
 *
 * The truth the card is held to is the server's, never the card's own constants: the project as a
 * real server served it (the metabolomics capture in src/mocks/fixtures, an assay table with a
 * `batch` column under prediction; under inference the same state with its purpose changed); every
 * payload and list of methods from turbotab/server/openapi.json, the contract the server's own test
 * holds to its pydantic models (core/decisions.py `SetBatch`); the line's name, place and where the
 * engine places it from the quest contract the engine writes (src/quest/fixtures/quest-contract.json);
 * and a refusal in the shape the server sends one (core/methods/batch.py `_batch_answer_fits`).
 */
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { api, RefusalError } from "../../../api/client";
import { keys } from "../../../api/queries";
import type { ColumnSummary, Decision, ProjectView, QuestLine, QuestStage, Refusal } from "../../../api/schema";
import { expandView, viewsOf, type M3Fixture } from "../../../mocks/m3";
import { StageFocusProvider } from "../../../state/focus";
import { CONTRACT } from "../../fixtures/contract";
import { openLine } from "../../log";
import { DECLARATION_CARDS } from "../predict";
import { recordable, rendererOf } from "../renderers";

const PID = "batch-test";
const AT = "2026-11-03T00:00:00Z";
const KIND = "set_batch";

// ── the server's contract (openapi.json) ─────────────────────────────────────

interface JsonSchema {
  type?: string;
  enum?: unknown[];
  const?: unknown;
  minLength?: number;
  default?: unknown;
  anyOf?: JsonSchema[];
  properties?: Record<string, JsonSchema>;
  required?: string[];
  additionalProperties?: boolean;
}
const OPENAPI = resolve(dirname(fileURLToPath(import.meta.url)), "../../../../../server/openapi.json");
const SCHEMAS = (JSON.parse(readFileSync(OPENAPI, "utf-8")) as { components: { schemas: Record<string, JsonSchema> } })
  .components.schemas;
/** The model a posted decision is parsed as (the request body's variant). */
const MODEL = SCHEMAS["SetBatch-Input"]!;
const METHODS = MODEL.properties!.method!.enum as string[];
const FIGURES_DEFAULT = MODEL.properties!.figures!.default;

/** What keeps `value` from parsing as schema `s`: one line per problem, none when it parses. */
function problems(s: JsonSchema, value: unknown, at = "decision"): string[] {
  if (s.anyOf) return s.anyOf.some((x) => problems(x, value, at).length === 0) ? [] : [`${at}: no variant takes it`];
  const out: string[] = [];
  if (s.const !== undefined && value !== s.const) out.push(`${at}: not ${String(s.const)}`);
  if (s.enum && !s.enum.includes(value)) out.push(`${at}: ${String(value)} is not one of ${s.enum.join(", ")}`);
  switch (s.type) {
    case "boolean":
      if (typeof value !== "boolean") out.push(`${at}: not a boolean`);
      break;
    case "string":
      if (typeof value !== "string") out.push(`${at}: not a string`);
      else if (s.minLength !== undefined && value.length < s.minLength) out.push(`${at}: too short`);
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

// ── the captured server (src/mocks/fixtures) ─────────────────────────────────

const FIXTURES = import.meta.glob<M3Fixture>("../../../mocks/fixtures/m3-metabolomics.json", {
  eager: true,
  import: "default",
});
const ASSAY = Object.values(FIXTURES)[0]!;
/** The capture's column summaries, in the table's order. The capture keeps the first 60 of the
 *  table's 400 (its `trim`), which the outcome is not among, so the outcome is served after them
 *  (the card reads a summary's name alone). */
const CAPTURED = ASSAY.endpoints.columns!.body as ColumnSummary[];
const SUMMARIES: ColumnSummary[] = [...CAPTURED, { name: "responder" } as ColumnSummary];
/** The capture's batch column: the one its acquisition inventory names as the batch. */
const BATCH = "batch";

/** The last captured view before the seal is opened: the Models stage, under prediction. */
function served(): ProjectView {
  const views = viewsOf(ASSAY);
  for (let i = views.length - 1; i >= 0; i--) {
    const v = views[i]!;
    if (v.state.target && v.state.purpose && !v.state.seal_opened) return expandView(ASSAY, v, PID, AT);
  }
  throw new Error("the metabolomics capture never reaches the models");
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

function mount(
  opts: {
    /** Fields of the captured state to change (the purpose, the record). */
    state?: Record<string, unknown>;
    line?: Partial<QuestLine>;
  } = {},
) {
  const at = served();
  const view = { ...at, state: { ...at.state, ...opts.state } } as ProjectView;
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  qc.setQueryData(keys.view(PID), view);
  qc.setQueryData(keys.columns(PID), SUMMARIES);
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
  return { decide, qc, view };
}

/** A press as a mouse makes it: a click chooses the option, and the card's own button records it. */
const press = (option: string) => {
  fireEvent.click(screen.getByTestId(`option-${option}`));
  fireEvent.click(screen.getByTestId("card-record"));
};
const choose = (column: string) => fireEvent.change(screen.getByTestId("field-column"), { target: { value: column } });
const offered = () => Array.from(document.querySelectorAll("[role=option]")).map((o) => o.getAttribute("data-key"));
const settle = () => new Promise((resolve) => setTimeout(resolve, 20));
const posted = async (decide: ReturnType<typeof vi.spyOn>, n = 1): Promise<Decision> => {
  await waitFor(() => expect(decide).toHaveBeenCalledTimes(n));
  return decide.mock.calls[n - 1]![1] as Decision;
};
const columnOptions = () => Array.from((screen.getByTestId("field-column") as HTMLSelectElement).options).map((o) => o.value);
/** A record of the batch answer, as the state holds one (core/decisions.py `BatchSpec`). */
const onRecord = (method: string, figures = false) => ({
  state: { batch: { column: BATCH, method, figures } },
  line: { status: "answered" as const, decision_id: "r9" },
});

afterEach(() => {
  vi.restoreAllMocks();
});

// ── the shell ───────────────────────────────────────────────────────────────

describe("the batch question the shell can answer (G08)", () => {
  it("is a counted Decide in Models, drawn through a card that records", () => {
    expect(PLACE).toMatchObject({ stage: "models", label: "Decide", counted: true });
    expect(rendererOf({ source: "declaration", key: KIND }, DECLARATION_CARDS)).toBe("declaration");
    expect(recordable({ source: "declaration", key: KIND }, DECLARATION_CARDS)).toBe(true);
  });

  it("opens Models on the batch question, where without its card the stage opens nothing", () => {
    const models = {
      key: "models",
      name: "Models",
      reached: true,
      progress: { answered: 0, required: 1, complete: false },
      sweep: null,
      lines: [lineOf()],
      reopened: [],
    } as unknown as QuestStage;
    expect(openLine(models, (l) => recordable(l, DECLARATION_CARDS))?.key).toBe(KIND);
    // Without the card no line opens: the base's dead end, "Not available yet".
    expect(openLine(models, (l) => recordable(l, {}))).toBeNull();
  });

  it("is placed by the engine on the assay walks under a goal alone, so no other walk mounts the card", () => {
    // core/quest.py `_batch_applies`: an assay lens, a goal, and a column named as a batch. The
    // fuzzer's NHANES table is no assay, and with no goal the line is not placed.
    const seen = CONTRACT.pairs.filter((p) => p.source === "declaration" && p.key === KIND).flatMap((p) => p.seen);
    expect([...seen].sort()).toEqual(["metabolomics/inference", "metabolomics/prediction"]);
    const lines = Object.values(CONTRACT.logs).flatMap((l) => l.log.stages.flatMap((s) => s.lines));
    expect(lines.length).toBeGreaterThan(0);
    expect(lines.some((l) => l.key === KIND)).toBe(false);
  });
});

// ── the card ────────────────────────────────────────────────────────────────

describe("the batch card", () => {
  it("lists every column but the outcome in the table's order, none chosen", () => {
    const { view } = mount();
    const card = screen.getByTestId(`declaration-${KIND}`);
    expect(within(card).getByRole("heading")).toHaveTextContent(new RegExp(`^${PLACE.name}$`, "i"));
    const options = columnOptions();
    // The column is the person's to choose: none is chosen for them.
    expect(options[0]).toBe("");
    expect((screen.getByTestId("field-column") as HTMLSelectElement).value).toBe("");
    const target = view.state.target!;
    expect(SUMMARIES.map((s) => s.name)).toContain(target);
    expect(options.slice(1)).toEqual(SUMMARIES.map((s) => s.name).filter((n) => n !== target));
    expect(options).toContain(BATCH);
  });

  it("offers exactly the methods the server's model takes", () => {
    mount();
    expect([...offered()].sort()).toEqual(METHODS.map((m) => `method:${m}`).sort());
  });

  it("records nothing, and says what is in the way, where a method is pressed before a column is chosen", async () => {
    const { decide } = mount();
    press("method:covariate");
    await settle();
    expect(decide).not.toHaveBeenCalled();
    const label = screen.getByTestId("field-column").closest("label")!.querySelector("span")!.textContent!;
    expect(screen.getByTestId("blocked")).toHaveTextContent(label);
    expect(screen.getByTestId("field-column")).toHaveAttribute("aria-invalid", "true");
  });

  it.each(["covariate", "reference_combat", "outcome_combat", "none", "not_a_batch"])(
    "records the method pressed for the column chosen, as the server's model takes it (%s)",
    async (method) => {
      const { decide } = mount();
      choose(BATCH);
      press(`method:${method}`);
      const sent = await posted(decide);
      expect(problems(MODEL, sent)).toEqual([]);
      expect(sent).toEqual({ kind: KIND, column: BATCH, method, figures: FIGURES_DEFAULT });
      expect(screen.queryByTestId("blocked")).toBeNull();
    },
  );

  it("marks nothing recorded on an open line", () => {
    mount();
    choose(BATCH);
    expect(document.querySelectorAll("[data-recorded]")).toHaveLength(0);
  });

  it("chooses the column of the record once the line is answered, and marks its method", () => {
    mount(onRecord("reference_combat"));
    expect((screen.getByTestId("field-column") as HTMLSelectElement).value).toBe(BATCH);
    expect(screen.getByTestId("option-method:reference_combat")).toHaveAttribute("data-recorded");
    expect(document.querySelectorAll("[data-recorded]")).toHaveLength(1);
  });

  it("keeps a column on record that the table's columns do not list choosable, with its method marked, and adds none on an open line", async () => {
    const gone = "plate_id";
    expect(SUMMARIES.map((x) => x.name)).not.toContain(gone);
    const { decide } = mount({ state: { batch: { column: gone, method: "covariate", figures: false } }, line: { status: "answered", decision_id: "r9" } });
    expect(columnOptions()).toContain(gone);
    expect((screen.getByTestId("field-column") as HTMLSelectElement).value).toBe(gone);
    expect(screen.getByTestId("option-method:covariate")).toHaveAttribute("data-recorded");
    press("method:none");
    expect(await posted(decide)).toEqual({ kind: KIND, column: gone, method: "none", figures: false });
    cleanup();
    vi.restoreAllMocks();
    mount({ state: { batch: { column: gone, method: "covariate", figures: false } } });
    expect(columnOptions()).not.toContain(gone);
  });

  it("marks nothing recorded for another column than the record's", () => {
    mount(onRecord("reference_combat"));
    choose("run_order");
    expect(document.querySelectorAll("[data-recorded]")).toHaveLength(0);
  });

  it("asks no figures-only answer under prediction, where the engine refuses one", async () => {
    const { decide, view } = mount();
    expect(view.state.purpose).toBe("prediction");
    expect(screen.queryByTestId("field-figures")).toBeNull();
    choose(BATCH);
    press("method:covariate");
    expect(await posted(decide)).toMatchObject({ method: "covariate", figures: false });
  });

  it("under inference asks whether ComBat serves the figures, and sends it with batch as a covariate", async () => {
    const { decide } = mount({ state: { purpose: "inference" } });
    const figures = screen.getByTestId("field-figures") as HTMLInputElement;
    expect(figures.checked).toBe(false);
    choose(BATCH);
    fireEvent.click(figures);
    press("method:covariate");
    const sent = await posted(decide);
    expect(problems(MODEL, sent)).toEqual([]);
    expect(sent).toEqual({ kind: KIND, column: BATCH, method: "covariate", figures: true });
  });

  it.each(["reference_combat", "outcome_combat", "none", "not_a_batch"])(
    "under inference sends no figures-only answer with any other method, which the engine reads with a covariate alone (%s)",
    async (method) => {
      const { decide } = mount({ state: { purpose: "inference" } });
      choose(BATCH);
      fireEvent.click(screen.getByTestId("field-figures"));
      press(`method:${method}`);
      expect(await posted(decide)).toEqual({ kind: KIND, column: BATCH, method, figures: false });
    },
  );

  it("starts the figures-only answer at the record's, and keeps it on a press of the method on record", async () => {
    const record = onRecord("covariate", true);
    const { decide } = mount({ ...record, state: { purpose: "inference", ...record.state } });
    expect((screen.getByTestId("field-figures") as HTMLInputElement).checked).toBe(true);
    expect(screen.getByTestId("option-method:covariate")).toHaveAttribute("data-recorded");
    press("method:covariate");
    expect(await posted(decide)).toEqual({ kind: KIND, column: BATCH, method: "covariate", figures: true });
  });

  it("answers a press the engine refuses in its own words at the option, and records the way forward pressed", async () => {
    // core/methods/batch.py `_batch_answer_fits`: ComBat with the outcome protected under
    // inference, refused for testing, its ways forward batch as a covariate with ComBat for figures.
    const forward = { kind: KIND, column: BATCH, method: "covariate", figures: true } as unknown as Decision;
    const refusal: Refusal = {
      error: {
        code: "outcome_combat_for_testing",
        message: "ComBat with the outcome protected inflates group differences when groups are spread unevenly over batches, so its values are refused for testing.",
        exits: [{ label: "Batch as a covariate; ComBat for figures only", decision: forward }],
      },
    } as unknown as Refusal;
    const { decide } = mount({ state: { purpose: "inference" } });
    decide.mockRejectedValueOnce(new RefusalError(refusal));
    choose(BATCH);
    press("method:outcome_combat");
    await posted(decide);
    const at = await screen.findByTestId("refusal");
    expect(at).toHaveTextContent(refusal.error.message);
    fireEvent.click(within(at).getByTestId("refusal-exit"));
    expect(await posted(decide, 2)).toEqual(forward);
  });
});
