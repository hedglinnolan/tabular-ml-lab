/**
 * The survey card's design columns (G16 of the front-end audit; the owner's Q4, 2026-11-04: the
 * columns editable, the engine's proposal preselected). The weight, strata, PSU, cycle and four-year
 * weight are fields on the card, starting from the engine's proposal, and recording them posts
 * set_survey as the contract takes it (SetSurvey-Input in turbotab/server/openapi.json, read here,
 * never retyped). The served words are the engine's:
 *
 * - NHANES names: the survey-design journey as captured (fixtures/m3-survey-design.json, the
 *   I4 table of test_wp10_survey_design), its proposal and its views.
 * - Another survey's names: that I4 table (`informative_tables`, seed 2026) with its design columns
 *   renamed as NHIS names its weight and strata, `WTFA_A` and `PSTRAT`, which the names reader does
 *   not read, beside a PSU called `psu`, which it does; `SEQN` renamed `HHX`. Driven through the
 *   real server on this base as test_wp10_survey_design drives it (`open_project`, the roles
 *   `WTFA_A`, `PSTRAT` and `psu` set to design): the proposal it served offers only "These
 *   participants", and it accepted (200) the population answer naming those three columns.
 * - The I4 table itself, driven the same way: its refusal of the population answer with no PSU
 *   (partial_design, its words and its two exits), and its acceptance (200) of that refusal's
 *   attested exit.
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
import { SurveyAsk } from "./ChoiceQuestions";

const HERE = dirname(fileURLToPath(import.meta.url));
const OPENAPI = JSON.parse(
  readFileSync(resolve(HERE, "../../../../../server/openapi.json"), "utf-8"),
) as {
  components: {
    schemas: Record<
      string,
      { properties: Record<string, { default?: unknown }>; required: string[] }
    >;
  };
};
/** What POST /decisions takes for the survey answer. */
const CONTRACT = OPENAPI.components.schemas["SetSurvey-Input"]!;

const FIXTURES = import.meta.glob<M3Fixture>("../../../mocks/fixtures/m3-*.json", {
  eager: true,
  import: "default",
});
const fixture = (journey: string) =>
  Object.entries(FIXTURES).find(([p]) => p.endsWith(`m3-${journey}.json`))![1];
const SHARED = fixture("shared") as unknown as { teaching: TeachingEntry[] };

type Survey = NonNullable<ProposalsArtifact["survey"]>;

/** The survey-design journey's first view with the survey question open and no readings asked
 *  beside it (its roles confirmed), and its proposal as captured. */
function surveyOpen() {
  const f = fixture("survey-design");
  const views = viewsOf(f);
  const i = views.findIndex((v) =>
    v.interview.some((s) => s.key === "survey" && s.status === "open" && !s.ask),
  );
  const v = views[i]!;
  const proposals = artifactAt(
    f,
    "proposals",
    v.stages.proposals?.key ?? null,
  ) as ProposalsArtifact;
  return { f, view: v, survey: proposals.survey! };
}
const NHANES = surveyOpen();
/** The engine's first population option on the NHANES table, and its decision. */
const FIRST = NHANES.survey.options.find((o) => o.decision.estimand === "population")!;

/** The other survey's proposal, as the server served it (no weight read, so no population option). */
const OTHER: Survey = {
  weights: [],
  strata: [],
  psu: ["psu"],
  cycle: null,
  cycles: [],
  four_year: {},
  options: [
    {
      key: "sample",
      label: "These participants",
      consequence: "Unweighted; the methods state the comparison is this sample's.",
      decision: { kind: "set_survey", estimand: "sample" },
    },
  ],
};
/** The population answer the server accepted on the other survey's table (status 200). */
const OTHER_ACCEPTED = {
  kind: "set_survey",
  estimand: "population",
  weight: "WTFA_A",
  strata: "PSTRAT",
  psu: "psu",
  cycle: null,
  four_year_weight: null,
  acknowledged: false,
};
/** The I4 table's answer with no PSU, as posted, and the server's refusal of it (409). */
const I4_NO_PSU = {
  kind: "set_survey",
  estimand: "population",
  weight: "WTDRD1",
  strata: "SDMVSTRA",
  psu: null,
  cycle: null,
  four_year_weight: null,
  acknowledged: false,
};
const PARTIAL: Refusal = {
  error: {
    code: "partial_design",
    message:
      "No PSU is named. Weights correct the estimate, not its interval: the intervals would treat each row as a sampling unit of its own, and where people were sampled in clusters they are too narrow.",
    exits: [
      { label: "Name the PSU column", decision: null },
      {
        label: "My table has no such column; record that",
        decision: {
          kind: "set_survey",
          estimand: "population",
          weight: "WTDRD1",
          strata: "SDMVSTRA",
          psu: null,
          cycle: null,
          four_year_weight: null,
          acknowledged: true,
          goal: null,
        },
      },
    ],
  },
};

/** The design fields, in the card's order, by their accessible names. */
const FIELDS = [
  ["weight", "Survey weight column"],
  ["strata", "Strata column"],
  ["psu", "PSU column"],
  ["cycle", "Survey cycle column"],
  ["four_year_weight", "Four-year weight column"],
] as const;

function renderCard(survey: Survey | undefined, current: ProjectState["survey"] = null) {
  const record = vi.fn<(d: Decision, at?: string) => void>();
  render(
    <SurveyAsk
      entry={undefined}
      pending={false}
      record={record}
      answerAt={null}
      shell={{
        qkey: "survey",
        now: true,
        arriving: false,
        onArrived: () => {},
        onOpenDrawer: () => {},
        reopened: false,
      }}
      proposal={survey}
      current={current}
    />,
  );
  return record;
}

const own = () => within(screen.getByTestId("option-custom"));
const field = (name: string) => own().getByLabelText(name) as HTMLInputElement;
/** Each design field's value, by the contract's field name. */
const shown = () => Object.fromEntries(FIELDS.map(([k, name]) => [k, field(name).value]));
/** A decision's design columns as the fields show them: a column it does not name is blank. */
const asFields = (d: Record<string, unknown>) =>
  Object.fromEntries(FIELDS.map(([k]) => [k, typeof d[k] === "string" ? d[k] : ""]));
const type = (name: string, value: string) => fireEvent.change(field(name), { target: { value } });
const press = () => fireEvent.click(own().getByRole("button", { name: "Record this design" }));

/** The options an answer on record is marked on (aria-selected), by key. */
function recordedOptions(): string[] {
  return within(screen.getByTestId("options-survey"))
    .getAllByRole("option")
    .filter((o) => o.getAttribute("aria-selected") === "true")
    .map((o) => o.getAttribute("data-key") ?? "");
}

afterEach(cleanup);

describe("the design's columns start from the engine's proposal", () => {
  it("as the engine's first population option names them", () => {
    expect(FIRST.decision.weight).toBeTruthy(); // the capture really offers a population option
    renderCard(NHANES.survey);
    expect(shown()).toEqual(asFields(FIRST.decision));
  });

  it("as the names read them, with no weight, where the engine offers no population option", () => {
    expect(OTHER.options.every((o) => o.decision.estimand !== "population")).toBe(true);
    renderCard(OTHER);
    expect(shown()).toEqual(
      asFields({ strata: OTHER.strata[0], psu: OTHER.psu[0], cycle: OTHER.cycle }),
    );
  });

  it("as recorded, when the answer on record is a population answer", () => {
    const attested = PARTIAL.error.exits[1]!.decision as Record<string, unknown>;
    renderCard(NHANES.survey, attested as unknown as ProjectState["survey"]);
    expect(shown()).toEqual(asFields(attested));
  });
});

describe("a design of one's own posts set_survey as the contract takes it", () => {
  it("names the columns the engine did not read, as the server accepted them", () => {
    const record = renderCard(OTHER);
    type("Survey weight column", OTHER_ACCEPTED.weight);
    type("Strata column", OTHER_ACCEPTED.strata);
    press();
    expect(record).toHaveBeenCalledTimes(1);
    const [decision, at] = record.mock.calls[0]!;
    expect(decision).toEqual(OTHER_ACCEPTED);
    expect(at).toBe("custom");
    // every field is one the contract takes, the ones it requires among them, and the attestation
    // at the contract's default: the server asks for it where no strata or PSU is named
    const fields = Object.keys(decision);
    expect(fields.filter((k) => !(k in CONTRACT.properties))).toEqual([]);
    expect(CONTRACT.required.filter((k) => !fields.includes(k))).toEqual([]);
    expect((decision as { acknowledged: boolean }).acknowledged).toBe(
      CONTRACT.properties.acknowledged!.default,
    );
  });

  it("posts a field left blank as no column", () => {
    const record = renderCard(NHANES.survey);
    type("PSU column", "");
    press();
    expect(record.mock.calls[0]).toEqual([I4_NO_PSU, "custom"]);
  });

  it("records nothing without a weight, and leads to the weight's field", () => {
    const record = renderCard(OTHER);
    const button = own().getByRole("button", { name: "Record this design" });
    expect(button).toHaveProperty("disabled", true);
    fireEvent.click(screen.getByTestId("option-custom"));
    expect(record).not.toHaveBeenCalled();
    expect(document.activeElement).toBe(field("Survey weight column"));
    type("Survey weight column", OTHER_ACCEPTED.weight);
    expect(button).toHaveProperty("disabled", false);
  });

  it("leaves the engine's own options recording their own decisions", () => {
    const record = renderCard(NHANES.survey);
    fireEvent.click(screen.getByTestId(`option-${FIRST.key}`));
    expect(record.mock.calls[0]).toEqual([FIRST.decision, FIRST.key]);
  });
});

describe("the answer on record is marked on the option that records it", () => {
  it("on the engine's option where that option names every column on record", () => {
    renderCard(NHANES.survey, FIRST.decision as unknown as ProjectState["survey"]);
    expect(recordedOptions()).toEqual([FIRST.key]);
  });

  it("on one's own design where the record names other columns than the engine's option", () => {
    // the attested answer the server accepted: the engine's weight and strata, no PSU
    const attested = PARTIAL.error.exits[1]!.decision as unknown as ProjectState["survey"];
    expect(attested!.weight).toBe(FIRST.decision.weight);
    renderCard(NHANES.survey, attested);
    expect(recordedOptions()).toEqual(["custom"]);
  });

  it("on these participants for the sample answer", () => {
    renderCard(NHANES.survey, {
      estimand: "sample",
      weight: null,
      strata: null,
      psu: null,
      cycle: null,
      four_year_weight: null,
      acknowledged: false,
    });
    expect(recordedOptions()).toEqual(["sample"]);
  });
});

// ── the server's refusal, said at the option with its exits ──────────────────────────────────

const PID = "m3-test";

/** The survey-design journey's open survey question, seeded as AskFor.test seeds a view. */
function surveyClient() {
  const { f, view: v } = NHANES;
  const view = expandView(f, v, PID, "2026-10-05T00:00:00Z") as ProjectView;
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

describe("a design the server refuses", () => {
  it("is said at one's own design in the server's words, and its way forward records its decision", async () => {
    const decide = vi.spyOn(api, "decide").mockRejectedValue(new RefusalError(PARTIAL));
    const { qc, view } = surveyClient();
    const { container } = render(
      <QueryClientProvider client={qc}>
        <StageFocusProvider>
          <AskFor pid={PID} view={view} stepKey="survey" />
        </StageFocusProvider>
      </QueryClientProvider>,
    );
    const card = within(container.querySelector<HTMLElement>('[data-testid="question-survey"]')!);
    const option = card.getByTestId("option-custom");
    fireEvent.change(within(option).getByLabelText("PSU column"), { target: { value: "" } });
    fireEvent.click(within(option).getByRole("button", { name: "Record this design" }));
    await waitFor(() => expect(decide).toHaveBeenCalledTimes(1));
    expect(decide.mock.calls[0]![1]).toEqual(I4_NO_PSU);
    const note = await card.findByTestId("refusal");
    // under the option pressed, in the server's words
    expect(option.nextElementSibling?.contains(note)).toBe(true);
    expect(note.textContent).toContain(PARTIAL.error.message);
    const exits = within(note).getAllByTestId("refusal-exit");
    // RefusalNote says an exit that records a decision as "<label> instead"
    expect(exits.map((b) => b.textContent)).toEqual(
      PARTIAL.error.exits.map((x) => (x.decision ? `${x.label} instead` : x.label)),
    );
    fireEvent.click(exits[1]!);
    await waitFor(() => expect(decide).toHaveBeenCalledTimes(2));
    expect(decide.mock.calls[1]![1]).toEqual(PARTIAL.error.exits[1]!.decision);
    decide.mockRestore();
  });
});
