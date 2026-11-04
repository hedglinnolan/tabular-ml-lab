/**
 * The missing-values card's multiple imputation (G03 of the front-end audit): under Estimate the
 * card offers it with its m and imputation-model choices, posting what the contract accepts, and
 * where the record names the columns whose blanks lie below a detection limit, the fills the server
 * offers for them. The expected values are the contract's own: SetMissing in
 * turbotab/server/openapi.json (m's default and bounds, the enums), read here, never retyped. The
 * served proposals and the refusal are the engine's: the captured journeys' proposals (methods,
 * imputation models and below-detection fills, each with its rung for the purpose), and the
 * imputation_too_wide refusal the engine serves for multiple imputation on test_imputation_width's
 * 150-row table (its words and its one exit, as `validate` raised them on this base).
 */
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { api, RefusalError } from "../../../api/client";
import type { ProposalsArtifact, TeachingEntry } from "../../../api/m1-types";
import { keys } from "../../../api/queries";
import type { Decision, ProjectState, ProjectView, Refusal } from "../../../api/schema";
import { artifactAt, expandView, servedAt, viewsOf, type M3Fixture } from "../../../mocks/m3";
import { StageFocusProvider } from "../../../state/focus";
import { AskFor } from "../AskFor";
import { MissingAsk } from "./ChoiceQuestions";

const HERE = dirname(fileURLToPath(import.meta.url));
const OPENAPI = JSON.parse(
  readFileSync(resolve(HERE, "../../../../../server/openapi.json"), "utf-8"),
) as { components: { schemas: Record<string, { properties: Record<string, unknown> }> } };
const FIELDS = OPENAPI.components.schemas.SetMissing!.properties as Record<
  string,
  { default?: unknown; minimum?: number; maximum?: number; anyOf?: { enum?: string[] }[] }
>;
/** The contract's m: its default and its bounds. */
const M = {
  default: FIELDS.m!.default as number,
  min: FIELDS.m!.minimum!,
  max: FIELDS.m!.maximum!,
};
/** A nullable enum's values, as the contract lists them. */
const enumOf = (field: string): string[] => FIELDS[field]!.anyOf!.find((a) => a.enum)!.enum!;

const FIXTURES = import.meta.glob<M3Fixture>("../../../mocks/fixtures/m3-*.json", {
  eager: true,
  import: "default",
});
const fixture = (journey: string) =>
  Object.entries(FIXTURES).find(([p]) => p.endsWith(`m3-${journey}.json`))![1];
const SHARED = fixture("shared") as unknown as { teaching: TeachingEntry[] };
const ENTRY = SHARED.teaching.find((e) => e.key === "missing")!;

/** A journey's first view with the missing-values question open, and its proposals as served. */
function missingOpen(journey: string) {
  const f = fixture(journey);
  const views = viewsOf(f);
  const i = views.findIndex((v) =>
    v.interview.some((s) => s.key === "missing" && s.status === "open"),
  );
  const v = views[i]!;
  const proposals = artifactAt(
    f,
    "proposals",
    v.stages.proposals?.key ?? null,
  ) as ProposalsArtifact;
  return { f, views, i, state: v.state, proposals };
}
// Estimate: the server's methods put multiple imputation first, recommended; Predict: refused.
const ESTIMATE = missingOpen("nhanes-inference");
const PREDICT = missingOpen("nhanes-prediction");
// Metabolomics (Predict), its missing answer as recorded: a fill in each fold, its non-detections
// censoring-aware, the censored columns the server named at its refusal.
const OMICS = missingOpen("metabolomics");
const OMICS_RECORD = OMICS.views.find((v) => (v.state.missing?.censored_columns ?? []).length > 0)!
  .state.missing!;

function renderCard({
  proposals,
  purpose,
  current = null,
}: {
  proposals: ProposalsArtifact;
  purpose: ProjectState["purpose"];
  current?: ProjectState["missing"];
}) {
  const record = vi.fn<(d: Decision, at?: string) => void>();
  render(
    <MissingAsk
      entry={ENTRY}
      pending={false}
      record={record}
      answerAt={null}
      shell={{
        qkey: "missing",
        now: true,
        arriving: false,
        onArrived: () => {},
        onOpenDrawer: () => {},
        reopened: false,
      }}
      proposals={proposals}
      columns={new Map()}
      purpose={purpose}
      current={current}
    />,
  );
  return record;
}

/** The one decision recorded, and the option it was recorded under. */
function recorded(record: ReturnType<typeof renderCard>): [Decision, string?] {
  expect(record).toHaveBeenCalledTimes(1);
  return record.mock.calls[0]!;
}

/** A multiple-imputation answer: every field the contract requires, at its contract default where
 *  the card leaves it. The compatible, clustered imputation is what the engine records for an
 *  answer that says nothing of either (decisions.py `_missing_records_what_its_copies_read`). */
const miSpec = (patch: Record<string, unknown> = {}) =>
  ({
    strategy: "multiple_imputation",
    drop_columns: [],
    categorical: "impute",
    indicators: false,
    m: M.default,
    below_detection: null,
    censored_columns: [],
    acknowledged: false,
    reason: null,
    imputation_model: "compatible",
    imputation_levels: "clustered",
    ...patch,
  }) as NonNullable<ProjectState["missing"]>;
const mi = (patch: Record<string, unknown> = {}): Decision =>
  ({ kind: "set_missing", ...miSpec(patch) }) as Decision;

afterEach(cleanup);

describe("multiple imputation, where the server offers it", () => {
  it("is offered under Estimate and posts the contract's default m", () => {
    expect(ESTIMATE.state.purpose).toBe("inference");
    const served = ESTIMATE.proposals.missing!.methods.find(
      (o) => o.key === "multiple_imputation",
    )!;
    expect(served.rung).toBe("recommended"); // the guard reads the server's offer, not the purpose
    const record = renderCard({ proposals: ESTIMATE.proposals, purpose: "inference" });
    fireEvent.click(screen.getByTestId("option-multiple_imputation"));
    expect(recorded(record)).toEqual([mi(), "multiple_imputation"]);
    expect(enumOf("imputation_model")).toContain("compatible");
    expect(enumOf("imputation_levels")).toContain("clustered");
  });

  it("is not offered under Predict, where the server refuses it", () => {
    expect(PREDICT.state.purpose).toBe("prediction");
    const served = PREDICT.proposals.missing!.methods.find((o) => o.key === "multiple_imputation")!;
    expect(served.rung).toBe("refused");
    renderCard({ proposals: PREDICT.proposals, purpose: "prediction" });
    expect(screen.queryByTestId("option-multiple_imputation")).toBeNull();
    expect(screen.queryByTestId("missing-m")).toBeNull();
    // the card's other options stand
    expect(screen.getByTestId("option-complete_case")).toBeTruthy();
    expect(screen.getByTestId("option-impute")).toBeTruthy();
  });

  it("is not offered where the served proposals list no methods", () => {
    const bare = {
      ...ESTIMATE.proposals,
      missing: { ...ESTIMATE.proposals.missing!, methods: [] },
    } as ProposalsArtifact;
    renderCard({ proposals: bare, purpose: "inference" });
    expect(screen.queryByTestId("option-multiple_imputation")).toBeNull();
  });
});

describe("m, within the contract's bounds", () => {
  const typeM = (value: string) =>
    fireEvent.change(screen.getByTestId("missing-m"), { target: { value } });

  it.each([
    ["the lowest", () => M.min],
    ["the highest", () => M.max],
    ["one between", () => Math.round((M.min + M.max) / 2)],
  ])("posts %s m the contract accepts", (_, at) => {
    const record = renderCard({ proposals: ESTIMATE.proposals, purpose: "inference" });
    typeM(String(at()));
    fireEvent.click(screen.getByTestId("option-multiple_imputation"));
    expect(recorded(record)).toEqual([mi({ m: at() }), "multiple_imputation"]);
  });

  it.each([
    ["below the lowest", () => String(M.min - 1)],
    ["above the highest", () => String(M.max + 1)],
    ["a fraction", () => `${M.min}.5`],
    ["blank", () => ""],
  ])("records nothing for an m %s, and says the bounds", (_, value) => {
    const record = renderCard({ proposals: ESTIMATE.proposals, purpose: "inference" });
    typeM(value());
    fireEvent.click(screen.getByTestId("option-multiple_imputation"));
    expect(record).not.toHaveBeenCalled();
    const said = screen.getByTestId("missing-m-bounds").textContent ?? "";
    expect(said).toContain(String(M.min));
    expect(said).toContain(String(M.max));
  });

  it("says when the fit draws more than m: where more than m% of the rows have a blank", () => {
    renderCard({ proposals: ESTIMATE.proposals, purpose: "inference" });
    typeM("37");
    const option = screen.getByTestId("option-multiple_imputation");
    expect(option.textContent).toContain("over 37% of the rows");
    expect(screen.queryByTestId("missing-m-bounds")).toBeNull();
  });

  // The fit draws the larger of m and the percentage of rows with a blank to impute (rounded up;
  // methods/imputation.py `m_rule`). A share of the rows is at most all of them, 100%, so from an
  // m of 100 the fit draws m whatever the blanks, and the card says no more.
  const ALL_ROWS = 100;
  it("says the fit may draw more only below an m of 100, the share of all the rows", () => {
    renderCard({ proposals: ESTIMATE.proposals, purpose: "inference" });
    const option = screen.getByTestId("option-multiple_imputation");
    typeM(String(ALL_ROWS - 1));
    expect(option.textContent).toContain(`over ${ALL_ROWS - 1}% of the rows`);
    for (const m of [ALL_ROWS, ALL_ROWS + 1, M.max]) {
      typeM(String(m));
      expect(option.textContent).not.toContain("% of the rows");
      expect(screen.queryByTestId("missing-m-bounds")).toBeNull(); // an m the contract accepts
    }
  });
});

describe("what multiple imputation is said to do", () => {
  // Under multiple imputation the cross-validated scores are the single fill's and the
  // coefficients are pooled over the imputations (models/pipeline.py `impute_detail`), and a
  // column below detection filled by half its minimum is not imputed (`imputation_count`).
  it("says the coefficients are pooled, and claims no imputation of every blank", () => {
    renderCard({ proposals: ESTIMATE.proposals, purpose: "inference" });
    const said = screen.getByTestId("option-multiple_imputation").textContent ?? "";
    expect(said).toContain("the coefficients are pooled");
    expect(said).not.toMatch(/each blank|results pooled/i);
  });
});

// ── a click on a control's words, inside an option ───────────────────────────────────────────
// A checkbox's or a field's words are its label: a click on them works the control (the label's
// activation) and records nothing, so a press of the option posts what the controls then show.

describe("a click on the words of an option's controls", () => {
  it("works multiple imputation's controls and records nothing", () => {
    const record = renderCard({ proposals: ESTIMATE.proposals, purpose: "inference" });
    const single = screen.getByTestId("missing-single-level") as HTMLInputElement;
    fireEvent.click(single.closest("label")!);
    expect(single.checked).toBe(true);
    fireEvent.click(screen.getByTestId("missing-m").closest("label")!);
    expect(record).not.toHaveBeenCalled();
    fireEvent.click(screen.getByTestId("option-multiple_imputation"));
    expect(recorded(record)).toEqual([
      mi({ imputation_levels: "single_level" }),
      "multiple_imputation",
    ]);
  });

  it("works the fill in each fold's boxes and records nothing", () => {
    const blanks = PREDICT.proposals.missing!.columns.map((b) => b.column);
    expect(blanks.length).toBeGreaterThanOrEqual(2);
    // one blank column that can be a level (categorical), one that cannot (a number)
    const columns = new Map([
      [blanks[0]!, { dtype: "categorical", n_unique: 3 }],
      [blanks[1]!, { dtype: "float", n_unique: 50 }],
    ]);
    const record = vi.fn<(d: Decision, at?: string) => void>();
    render(
      <MissingAsk
        entry={ENTRY}
        pending={false}
        record={record}
        answerAt={null}
        shell={{
          qkey: "missing",
          now: true,
          arriving: false,
          onArrived: () => {},
          onOpenDrawer: () => {},
          reopened: false,
        }}
        proposals={PREDICT.proposals}
        columns={columns}
        purpose="prediction"
        current={null}
      />,
    );
    const levels = screen.getByTestId("missing-levels") as HTMLInputElement;
    const indicators = screen.getByTestId("missing-indicators") as HTMLInputElement;
    fireEvent.click(levels.closest("label")!);
    fireEvent.click(indicators.closest("label")!);
    expect([levels.checked, indicators.checked]).toEqual([true, true]);
    expect(record).not.toHaveBeenCalled();
    fireEvent.click(screen.getByTestId("option-impute"));
    expect(recorded(record)).toEqual([
      expect.objectContaining({
        kind: "set_missing",
        strategy: "impute",
        categorical: "missing_category",
        indicators: true,
      }),
      "impute",
    ]);
  });
});

// ── a sub-answer the server does not recommend, said where it is chosen ──────────────────────
// The server serves each sub-answer with its rung and its sound sentence for the purpose
// (methods/missing.py `imputation_model_options`, `below_detection_options`; §11.3: an option is
// never offered silently). The card shows the sentence beside the control while a sub-answer the
// server does not recommend is chosen, and none for one it recommends.

describe("a sub-answer the server does not recommend, said where it is chosen", () => {
  const sub = (key: string) =>
    ESTIMATE.proposals.missing!.imputation_models.find((o) => o.key === key)!;
  const visible = (testId: string) => screen.getByTestId(testId).textContent ?? "";

  it("says the passive model's caution while it is chosen, and none for the compatible model", () => {
    expect(sub("passive").rung).not.toBe("recommended");
    expect(sub("compatible").rung).toBe("recommended");
    renderCard({ proposals: ESTIMATE.proposals, purpose: "inference" });
    const said = () => visible("option-multiple_imputation");
    expect(said()).not.toContain(sub("passive").sound);
    expect(said()).not.toContain(sub("compatible").sound);
    const model = screen.getByTestId("missing-imputation-model");
    fireEvent.change(model, { target: { value: "passive" } });
    expect(said()).toContain(sub("passive").sound);
    fireEvent.change(model, { target: { value: "compatible" } });
    expect(said()).not.toContain(sub("passive").sound);
    expect(said()).not.toContain(sub("compatible").sound);
  });

  it("says single-level's caution while it is ticked, and none for clustered imputation", () => {
    expect(sub("single_level").rung).not.toBe("recommended");
    renderCard({ proposals: ESTIMATE.proposals, purpose: "inference" });
    const said = () => visible("option-multiple_imputation");
    expect(said()).not.toContain(sub("single_level").sound);
    fireEvent.click(screen.getByTestId("missing-single-level"));
    expect(said()).toContain(sub("single_level").sound);
    fireEvent.click(screen.getByTestId("missing-single-level"));
    expect(said()).not.toContain(sub("single_level").sound);
  });

  it("says a recorded passive, single-level answer's cautions when the card opens", () => {
    const current = miSpec({ imputation_model: "passive", imputation_levels: "single_level" });
    renderCard({ proposals: ESTIMATE.proposals, purpose: "inference", current });
    const said = visible("option-multiple_imputation");
    expect(said).toContain(sub("passive").sound);
    expect(said).toContain(sub("single_level").sound);
  });

  it.each([
    ["the fill in each fold", "impute", () => OMICS, () => OMICS_RECORD],
    [
      "multiple imputation",
      "mi",
      () => ESTIMATE,
      () => ({ ...OMICS_RECORD, strategy: "multiple_imputation" }) as ProjectState["missing"],
    ],
  ])(
    "under %s, says half the minimum's caution where it is the fill, and none for the censoring-aware fill",
    (_, where, journey, current) => {
      const { proposals, state } = journey();
      const fills = proposals.missing!.below_detection;
      const fill = (key: string) => fills.find((o) => o.key === key)!;
      expect(fill("half_minimum").rung).not.toBe("recommended");
      expect(fill("censoring_aware").rung).toBe("recommended");
      renderCard({ proposals, purpose: state.purpose, current: current() });
      const option = where === "mi" ? "option-multiple_imputation" : "option-impute";
      const said = () => visible(option);
      expect(said()).not.toContain(fill("half_minimum").sound);
      expect(said()).not.toContain(fill("censoring_aware").sound);
      const select = screen.getByTestId(`missing-fill-${where}`);
      fireEvent.change(select, { target: { value: "half_minimum" } });
      expect(said()).toContain(fill("half_minimum").sound);
      fireEvent.change(select, { target: { value: "censoring_aware" } });
      expect(said()).not.toContain(fill("half_minimum").sound);
      expect(said()).not.toContain(fill("censoring_aware").sound);
    },
  );
});

describe("the imputation model and its levels, from the contract's enums", () => {
  it("offers each imputation model the server serves, every one a value of the contract", () => {
    renderCard({ proposals: ESTIMATE.proposals, purpose: "inference" });
    const select = screen.getByTestId("missing-imputation-model") as HTMLSelectElement;
    const values = [...select.options].map((o) => o.value);
    expect([...values].sort()).toEqual([...enumOf("imputation_model")].sort());
    // each labeled in the server's words
    const served = ESTIMATE.proposals.missing!.imputation_models;
    for (const o of select.options)
      expect(o.textContent).toBe(
        served.find((s) => s.decision.imputation_model === o.value)!.label,
      );
  });

  it("posts the passive model when it is chosen", () => {
    const record = renderCard({ proposals: ESTIMATE.proposals, purpose: "inference" });
    fireEvent.change(screen.getByTestId("missing-imputation-model"), {
      target: { value: "passive" },
    });
    fireEvent.click(screen.getByTestId("option-multiple_imputation"));
    expect(recorded(record)).toEqual([mi({ imputation_model: "passive" }), "multiple_imputation"]);
  });

  it("posts each of the contract's levels: clustered, or single-level when it is ticked", () => {
    const levels = enumOf("imputation_levels");
    const record = renderCard({ proposals: ESTIMATE.proposals, purpose: "inference" });
    fireEvent.click(screen.getByTestId("option-multiple_imputation"));
    fireEvent.click(screen.getByTestId("missing-single-level"));
    fireEvent.click(screen.getByTestId("option-multiple_imputation"));
    const posted = record.mock.calls.map(
      ([d]) => (d as { imputation_levels: string }).imputation_levels,
    );
    expect(posted).toEqual(["clustered", "single_level"]);
    expect([...posted].sort()).toEqual([...levels].sort());
  });

  it("reads an answer on record back: its m, its model and its levels", () => {
    const current = miSpec({
      m: 60,
      imputation_model: "passive",
      imputation_levels: "single_level",
    });
    const record = renderCard({ proposals: ESTIMATE.proposals, purpose: "inference", current });
    const option = screen.getByTestId("option-multiple_imputation");
    expect(option.getAttribute("aria-selected")).toBe("true");
    expect((screen.getByTestId("missing-m") as HTMLInputElement).value).toBe("60");
    expect((screen.getByTestId("missing-imputation-model") as HTMLSelectElement).value).toBe(
      "passive",
    );
    expect((screen.getByTestId("missing-single-level") as HTMLInputElement).checked).toBe(true);
    // a changed m is not the answer on record
    fireEvent.change(screen.getByTestId("missing-m"), { target: { value: "61" } });
    expect(option.getAttribute("aria-selected")).toBe("false");
    fireEvent.click(option);
    expect(recorded(record)).toEqual([
      mi({ m: 61, imputation_model: "passive", imputation_levels: "single_level" }),
      "multiple_imputation",
    ]);
  });
});

describe("values below a detection limit, where the record names their columns", () => {
  const censored = OMICS_RECORD.censored_columns;

  it("offers the fills the server offers, each a value of the contract, under the recorded fill", () => {
    expect(censored.length).toBeGreaterThan(0);
    expect(OMICS_RECORD.below_detection).toBe("censoring_aware");
    renderCard({ proposals: OMICS.proposals, purpose: "prediction", current: OMICS_RECORD });
    const select = screen.getByTestId("missing-fill-impute") as HTMLSelectElement;
    const served = OMICS.proposals.missing!.below_detection;
    const offered = served.filter((o) => o.rung !== "refused").map((o) => o.key);
    expect([...select.options].map((o) => o.value)).toEqual(offered);
    for (const v of offered) expect(enumOf("below_detection")).toContain(v);
    // the fill the server refuses is not offered
    expect(served.find((o) => o.key === "as_missing")!.rung).toBe("refused");
    expect([...select.options].map((o) => o.value)).not.toContain("as_missing");
    expect(select.value).toBe("censoring_aware");
    expect(screen.getByTestId("option-impute").getAttribute("aria-selected")).toBe("true");
  });

  it("keeps the recorded fill and its columns when the fill in each fold is pressed again", () => {
    const record = renderCard({
      proposals: OMICS.proposals,
      purpose: "prediction",
      current: OMICS_RECORD,
    });
    fireEvent.click(screen.getByTestId("option-impute"));
    const [d, at] = recorded(record);
    expect(at).toBe("impute");
    expect(d).toMatchObject({
      strategy: "impute",
      below_detection: "censoring_aware",
      censored_columns: censored,
    });
  });

  it("posts the other fill when it is chosen", () => {
    const record = renderCard({
      proposals: OMICS.proposals,
      purpose: "prediction",
      current: OMICS_RECORD,
    });
    fireEvent.change(screen.getByTestId("missing-fill-impute"), {
      target: { value: "half_minimum" },
    });
    expect(screen.getByTestId("option-impute").getAttribute("aria-selected")).toBe("false");
    fireEvent.click(screen.getByTestId("option-impute"));
    expect(recorded(record)[0]).toMatchObject({
      strategy: "impute",
      below_detection: "half_minimum",
      censored_columns: censored,
    });
  });

  it("goes with multiple imputation under Estimate", () => {
    const current = { ...OMICS_RECORD, strategy: "multiple_imputation" } as ProjectState["missing"];
    const record = renderCard({ proposals: ESTIMATE.proposals, purpose: "inference", current });
    fireEvent.change(screen.getByTestId("missing-fill-mi"), { target: { value: "half_minimum" } });
    fireEvent.click(screen.getByTestId("option-multiple_imputation"));
    expect(recorded(record)).toEqual([
      mi({ m: OMICS_RECORD.m, below_detection: "half_minimum", censored_columns: censored }),
      "multiple_imputation",
    ]);
  });

  it("shows and posts a fill recorded while the card is open (an exit taken at its refusal)", () => {
    const record = vi.fn<(d: Decision, at?: string) => void>();
    const card = (current: ProjectState["missing"]) => (
      <MissingAsk
        entry={ENTRY}
        pending={false}
        record={record}
        answerAt={null}
        shell={{
          qkey: "missing",
          now: true,
          arriving: false,
          onArrived: () => {},
          onOpenDrawer: () => {},
          reopened: false,
        }}
        proposals={OMICS.proposals}
        columns={new Map()}
        purpose="prediction"
        current={current}
      />
    );
    const { rerender } = render(card(null));
    expect(screen.queryByTestId("missing-fill-impute")).toBeNull();
    rerender(card(OMICS_RECORD));
    expect((screen.getByTestId("missing-fill-impute") as HTMLSelectElement).value).toBe(
      OMICS_RECORD.below_detection,
    );
    fireEvent.click(screen.getByTestId("option-impute"));
    expect(recorded(record)[0]).toMatchObject({
      below_detection: OMICS_RECORD.below_detection,
      censored_columns: censored,
    });
  });

  it("is not asked where the record names no censored column", () => {
    const current = PREDICT.views.find((v) => v.state.missing)?.state.missing ?? null;
    expect(current).not.toBeNull();
    expect(current!.censored_columns).toEqual([]);
    const record = renderCard({ proposals: PREDICT.proposals, purpose: "prediction", current });
    expect(screen.queryByTestId("missing-fill-impute")).toBeNull();
    fireEvent.click(screen.getByTestId("option-impute"));
    expect(recorded(record)[0]).toMatchObject({ below_detection: null, censored_columns: [] });
  });
});

// ── a refusal the server serves, said at the option ──────────────────────────────────────────

const PID = "m3-test";
/** The engine's imputation_too_wide refusal of multiple imputation on test_imputation_width's
 *  150-row table (smcfcs.py `_imputation_has_its_rows`), as `validate` raised it on this base. */
const TOO_WIDE: Refusal = {
  error: {
    code: "imputation_too_wide",
    message:
      "Multiple imputation cannot run on these data: The imputation model would hold 90 columns (categories as indicators): up to 80 it is drawn whatever its rows, and wider only where each incomplete column's model has the rows per parameter its rule asks for. The chained equations would impute `smoke` from a model of 89 predictor parameters (categories as indicators) fit on the 142 rows that record it, 61 of them its rarer value, 0.7 events per parameter, where a common rule for logistic coefficients asks for 10 (Peduzzi et al. 1996). The model of `x` falls short of its own rule too: 137 rows on 89 parameters, 1.5 per parameter where its rule asks for about 2.",
    exits: [
      {
        label: "Complete cases, with their assumption stated",
        decision: {
          kind: "set_missing",
          strategy: "complete_case",
          drop_columns: [],
          categorical: "impute",
          indicators: false,
          m: 20,
          below_detection: null,
          censored_columns: [],
          acknowledged: false,
          reason: null,
          imputation_model: "compatible",
          imputation_levels: "clustered",
        } as unknown as Decision,
      },
    ],
  },
};

/** The causal journey (Estimate) with the missing-values question open, as AskFor.test seeds it. */
function causalClient() {
  const { f, views, i } = missingOpen("causal");
  const view = expandView(f, views[i]!, PID, "2026-10-05T00:00:00Z") as ProjectView;
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false, staleTime: Infinity } },
  });
  qc.setQueryData(keys.teaching(), SHARED.teaching);
  if (view.state.purpose) qc.setQueryData(keys.teaching(view.state.purpose), SHARED.teaching);
  qc.setQueryData(keys.view(PID), view);
  const columns = f.endpoints.columns;
  if (columns?.status === 200) qc.setQueryData(keys.columns(PID), columns.body);
  for (const [stage, status] of Object.entries(view.stages)) {
    if (!status || status.status === "idle" || status.status === "blocked") continue;
    qc.setQueryData(keys.stage(PID, stage), {
      stage,
      key: status.key,
      fresh: status.status === "fresh",
      status: status.status,
      artifact: servedAt(f, stage, status.key, view),
    });
  }
  return { qc, view };
}

describe("a refusal of multiple imputation, served", () => {
  it("is said at the option in the server's words, and its way forward records its decision", async () => {
    const decide = vi.spyOn(api, "decide").mockRejectedValue(new RefusalError(TOO_WIDE));
    const { qc, view } = causalClient();
    expect(view.state.purpose).toBe("inference");
    const { container } = render(
      <QueryClientProvider client={qc}>
        <StageFocusProvider>
          <AskFor pid={PID} view={view} stepKey="missing" />
        </StageFocusProvider>
      </QueryClientProvider>,
    );
    const card = within(container.querySelector<HTMLElement>('[data-testid="question-missing"]')!);
    const option = card.getByTestId("option-multiple_imputation");
    fireEvent.click(option);
    await waitFor(() => expect(decide).toHaveBeenCalledTimes(1));
    expect(decide.mock.calls[0]![1]).toEqual(mi());
    const note = await card.findByTestId("refusal");
    // under the option pressed, in the server's words (its ticked names set as code)
    expect(option.nextElementSibling?.contains(note)).toBe(true);
    expect(note.textContent).toContain(TOO_WIDE.error.message.replace(/`/g, ""));
    const exits = within(note).getAllByTestId("refusal-exit");
    // RefusalNote says an exit that records a decision as "<label> instead"
    expect(exits.map((b) => b.textContent)).toEqual(
      TOO_WIDE.error.exits.map((x) => `${x.label} instead`),
    );
    fireEvent.click(exits[0]!);
    await waitFor(() => expect(decide).toHaveBeenCalledTimes(2));
    expect(decide.mock.calls[1]![1]).toEqual(TOO_WIDE.error.exits[0]!.decision);
    decide.mockRestore();
  });
});
