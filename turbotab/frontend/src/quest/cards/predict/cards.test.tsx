/**
 * The Predict declaration cards (W1-F): each option posts the payload core/decisions.py defines.
 *
 * payloads.fixture.json lists, for every option of every card, what the card must post when its
 * fields are filled in as the case says. This file presses each option and asserts the card posts
 * exactly that, and that the card lists exactly the options the fixture lists;
 * turbotab/core/tests/test_w1_declaration_cards.py holds the fixture to the server's models
 * (parse_decision, the Literal values, the bounds, the defaults). The words on a card are the
 * engine's: a line of the quest log, the preview the engine serves for the option pointed at, or
 * the refusal it answers with. A card with nothing served to say names the option and stops.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { api, ApiError, RefusalError } from "../../../api/client";
import { keys } from "../../../api/queries";
import type { PreviewResult } from "../../../api/m1-stage-types";
import type { Decision, ProjectState, ProjectView, QuestLine, Refusal } from "../../../api/schema";
import { StageFocusProvider } from "../../../state/focus";
import FIXTURE from "./payloads.fixture.json";
import { DECLARATION_CARDS } from "./index";

const PID = "nhanes-glucose";
const KINDS = [
  "set_intended_use",
  "set_selection",
  "set_validation",
  "set_levers",
  "set_updating",
  "set_explain",
] as const;
type Kind = (typeof KINDS)[number];
/** The six W1-F cards the fixture pins, and the sensitivity analyses' card the design repair added
 *  (its payloads are held to the model below, outside the six-model fixture). */
type CardKind = Kind | "set_sensitivity";

interface Case {
  card: Kind;
  press: string;
  note: string;
  fill?: Record<string, string>;
  toggle?: string[];
  recorded?: Record<string, unknown>;
  answered?: boolean;
  payload: Decision | null;
}
const CASES = FIXTURE.cases as unknown as Case[];

// The table the cards read: NHANES glucose under Predict. Subgroups and columns to draw come from
// it; the outcome is neither.
const COLUMNS = [
  "SEQN",
  "RIDAGEYR",
  "RIAGENDR",
  "RIDRETH1",
  "BMXBMI",
  "cycle_begin_year",
  "LBXGLU",
];
const ROLES = {
  SEQN: "identifier",
  RIDAGEYR: "covariate",
  RIAGENDR: "covariate",
  RIDRETH1: "covariate",
  BMXBMI: "exposure",
  cycle_begin_year: "cluster",
  LBXGLU: "covariate",
};
const DEFAULT_SPLIT = {
  holdout: 0.2,
  seed: 0,
  folds: 5,
  validation: "kfold",
  repeats: 10,
  n_boot: 500,
  cluster: null,
  nested_cv: false,
};

function stateWith(recorded: Record<string, unknown> = {}): ProjectState {
  return {
    lens: ["clinical"],
    target: "LBXGLU",
    task: "regression",
    purpose: "prediction",
    roles: ROLES,
    missing: null,
    split: DEFAULT_SPLIT,
    models: ["linear", "ridge"],
    intended_use: null,
    selection: null,
    levers: null,
    updating: null,
    explain: null,
    ...recorded,
  } as unknown as ProjectState;
}

function viewOf(state: ProjectState): ProjectView {
  return {
    state,
    decisions: [],
    interview: [],
    summary: null,
    stages: { ingest: { stage: "ingest", status: "fresh", key: "k", updated_at: null } },
  } as unknown as ProjectView;
}

const NAMES: Record<CardKind, string> = {
  set_intended_use: "the intended-use question",
  set_selection: "the predictor-selection question",
  set_validation: "the validation scheme",
  set_levers: "the rules repeated inside each fold",
  set_updating: "the shrinkage question",
  set_explain: "the explanations",
  set_sensitivity: "the sensitivity analyses",
};

function lineOf(kind: CardKind, over: Partial<QuestLine> = {}): QuestLine {
  return {
    id: `decision:${kind}`,
    key: kind,
    source: "declaration",
    label: "Decide",
    name: NAMES[kind],
    status: "open",
    counted: true,
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

interface Mounted {
  decide: ReturnType<typeof vi.spyOn>;
}

function mount(
  kind: CardKind,
  opts: { recorded?: Record<string, unknown>; answered?: boolean; line?: Partial<QuestLine> } = {},
): Mounted {
  const state = stateWith(opts.recorded);
  const view = viewOf(state);
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false, staleTime: Infinity } },
  });
  qc.setQueryData(keys.view(PID), view);
  qc.setQueryData(
    keys.columns(PID),
    COLUMNS.map((name) => ({ name })),
  );
  const decide = vi.spyOn(api, "decide").mockResolvedValue(view);
  const Card = DECLARATION_CARDS[kind];
  if (!Card) throw new Error(`no card for ${kind}`);
  const line = lineOf(kind, {
    status: opts.answered ? "answered" : "open",
    decision_id: opts.answered ? "r9" : null,
    ...opts.line,
  });
  render(
    <QueryClientProvider client={qc}>
      <StageFocusProvider>
        <Card pid={PID} line={line} />
      </StageFocusProvider>
    </QueryClientProvider>,
  );
  return { decide };
}

afterEach(() => {
  vi.restoreAllMocks();
});

const settle = () => new Promise((resolve) => setTimeout(resolve, 20));

describe("the Predict declaration cards", () => {
  it("are one per declaration kind, and no other", () => {
    expect(Object.keys(DECLARATION_CARDS).sort()).toEqual([...KINDS, "set_sensitivity"].sort());
  });

  it.each(CASES.map((c) => [`${c.card} · ${c.press}${c.note ? ` · ${c.note}` : ""}`, c] as const))(
    "%s",
    async (_name, c) => {
      const { decide } = mount(c.card, { recorded: c.recorded, answered: c.answered });
      for (const [testid, value] of Object.entries(c.fill ?? {})) {
        fireEvent.change(screen.getByTestId(testid), { target: { value } });
      }
      for (const testid of c.toggle ?? []) fireEvent.click(screen.getByTestId(testid));
      fireEvent.click(screen.getByTestId(`option-${c.press}`));
      if (c.payload === null) {
        // The model would refuse it (the Python test shows so): the card does not send it.
        await settle();
        expect(decide).not.toHaveBeenCalled();
        return;
      }
      await waitFor(() => expect(decide).toHaveBeenCalledTimes(1));
      expect(decide.mock.calls[0]).toEqual([PID, c.payload]);
      // A press that is sent has nothing to be told.
      expect(screen.queryByTestId("blocked")).toBeNull();
    },
  );

  it.each(
    CASES.filter((c) => c.payload === null).map(
      (c) => [`${c.card} · ${c.press}${c.note ? ` · ${c.note}` : ""}`, c] as const,
    ),
  )("%s: the press is answered at the option, in the field's own words", (_name, c) => {
    const { decide } = mount(c.card, { recorded: c.recorded, answered: c.answered });
    for (const [testid, value] of Object.entries(c.fill ?? {})) {
      fireEvent.change(screen.getByTestId(testid), { target: { value } });
    }
    const option = screen.getByTestId(`option-${c.press}`);
    fireEvent.click(option);
    expect(decide).not.toHaveBeenCalled();
    const note = screen.getByTestId("blocked");
    // right under the option pressed
    expect(option.nextElementSibling).toContainElement(note);
    // each field the case filled is named as the card labels it, with the model's bounds
    const fields = FIXTURE.bounds[c.card as keyof typeof FIXTURE.bounds] as
      Record<string, Record<string, number>> | undefined;
    for (const testid of Object.keys(c.fill ?? {})) {
      const bounds = fields?.[testid.replace("field-", "")];
      if (!bounds) continue;
      const label = screen.getByTestId(testid).closest("label")?.textContent ?? "";
      expect(label).not.toBe("");
      expect(note).toHaveTextContent(label);
      for (const n of Object.values(bounds)) {
        expect(note).toHaveTextContent(n.toLocaleString("en-US"));
      }
    }
  });
});

describe("what each card offers", () => {
  it.each(KINDS.map((k) => [k] as const))("%s lists exactly the fixture's options", (kind) => {
    mount(kind);
    const offered = Array.from(document.querySelectorAll("[role=option]")).map((o) =>
      o.getAttribute("data-key"),
    );
    const expected = Object.entries(FIXTURE.choices[kind]).flatMap(([field, values]) =>
      (values as string[]).map((v) => `${field}:${v}`),
    );
    expect(offered).toEqual(expected);
  });

  it.each(Object.entries(FIXTURE.selects))(
    "%s lists exactly the fixture's selects",
    (kind, selects) => {
      mount(kind as Kind);
      for (const [field, spec] of Object.entries(selects)) {
        const select = screen.getByTestId(`field-${field}`) as HTMLSelectElement;
        const values = Array.from(select.options).map((o) => o.value);
        expect(values.filter((v) => v !== "")).toEqual(spec.values);
        expect(values.includes("")).toBe(spec.blank);
      }
    },
  );

  it("holds its defaults and bounds to the fixture, which the Python test holds to the models", async () => {
    const { BOUNDS, DEFAULTS } = await import("./payloads");
    expect(BOUNDS).toEqual(FIXTURE.bounds);
    expect(DEFAULTS).toEqual(FIXTURE.defaults);
  });

  it.each(Object.entries(FIXTURE.bounds))(
    "%s takes its numbers within the models' bounds",
    (kind, fields) => {
      mount(kind as Kind);
      for (const [field, b] of Object.entries(fields) as [string, Record<string, number>][]) {
        const input = screen.getByTestId(`field-${field}`);
        expect(input.getAttribute("min")).toBe(String(b.ge ?? b.gt));
        if (b.le !== undefined || b.lt !== undefined) {
          expect(input.getAttribute("max")).toBe(String(b.le ?? b.lt));
        }
      }
    },
  );

  it("starts the validation fields at the model's defaults where no split is recorded", async () => {
    mount("set_validation", { recorded: { split: null } });
    expect((screen.getByTestId("field-folds") as HTMLInputElement).value).toBe("5");
    expect((screen.getByTestId("field-repeats") as HTMLInputElement).value).toBe("10");
    expect((screen.getByTestId("field-n_boot") as HTMLInputElement).value).toBe("500");
  });

  it("offers the subgroups the table has, and not its outcome or its identifier", () => {
    mount("set_intended_use");
    const picks = Array.from(document.querySelectorAll("[data-testid^=pick-subgroups-]")).map((e) =>
      e.getAttribute("data-testid"),
    );
    expect(picks).toEqual(
      ["RIDAGEYR", "RIAGENDR", "RIDRETH1", "BMXBMI", "cycle_begin_year"].map(
        (c) => `pick-subgroups-${c}`,
      ),
    );
  });

  it("offers the predictors the roles give the models, to draw curves for", () => {
    mount("set_explain");
    const picks = Array.from(document.querySelectorAll("[data-testid^=pick-exposures-]")).map((e) =>
      e.getAttribute("data-testid"),
    );
    // by role: the identifier, the grouping and the outcome are not predictors
    expect(picks).toEqual(
      ["RIDAGEYR", "RIAGENDR", "RIDRETH1", "BMXBMI"].map((c) => `pick-exposures-${c}`),
    );
  });

  it("offers the table's columns, the outcome apart, as the groups of internal–external validation", () => {
    mount("set_validation");
    const select = screen.getByTestId("field-cluster") as HTMLSelectElement;
    expect(Array.from(select.options).map((o) => o.value)).toEqual([
      "",
      "SEQN",
      "RIDAGEYR",
      "RIAGENDR",
      "RIDRETH1",
      "BMXBMI",
      "cycle_begin_year",
    ]);
  });
});

describe("what a card says of what is recorded", () => {
  it("marks the recorded option when the line is answered, and never when it is not", () => {
    mount("set_updating", {
      recorded: { updating: { method: "shrinkage" } },
      answered: true,
    });
    expect(screen.getByTestId("option-method:shrinkage")).toHaveAttribute("data-recorded");
    expect(screen.getByTestId("option-method:none")).not.toHaveAttribute("data-recorded");
  });

  it("marks nothing recorded on a line the engine set for the person", () => {
    // The validation scheme rides in the split's record until it is changed: set for you, not
    // answered, so "recorded" would say what the log does not.
    mount("set_validation", { line: { status: "set_for_you", label: "Confirm" } });
    expect(document.querySelectorAll("[data-recorded]")).toHaveLength(0);
  });

  it("marks the scheme recorded by set_validation", () => {
    mount("set_validation", {
      recorded: { split: { ...DEFAULT_SPLIT, validation: "bootstrap" } },
      answered: true,
    });
    expect(screen.getByTestId("option-validation:bootstrap")).toHaveAttribute("data-recorded");
  });

  it("marks one option in each of the levers", () => {
    mount("set_levers", {
      recorded: { levers: { forms: "rule", variance_filter: "top", keep: 500, imbalance: "none" } },
      answered: true,
    });
    const marked = Array.from(document.querySelectorAll("[data-recorded]")).map((o) =>
      o.getAttribute("data-key"),
    );
    expect(marked).toEqual(["forms:rule", "variance_filter:top", "imbalance:none"]);
  });

  it("says the engine's reason, what another choice would change, and why none would", () => {
    mount("set_levers", {
      line: {
        label: "Confirm",
        status: "set_for_you",
        reason: "Set for you: no rule repeats inside the folds.",
        would_change: "An in-fold rule changes what each model is fit on.",
        changes_nothing: "Nothing changes here.",
      },
    });
    expect(screen.getByTestId("line-reason")).toHaveTextContent(
      "Set for you: no rule repeats inside the folds.",
    );
    expect(screen.getByTestId("line-would-change")).toHaveTextContent(
      "An in-fold rule changes what each model is fit on.",
    );
    expect(screen.getByTestId("line-changes-nothing")).toHaveTextContent("Nothing changes here.");
  });

  it("says none of them when the engine served none", () => {
    mount("set_levers");
    expect(screen.queryByTestId("line-reason")).toBeNull();
    expect(screen.queryByTestId("line-would-change")).toBeNull();
    expect(screen.queryByTestId("line-changes-nothing")).toBeNull();
  });

  it("waits, naming what it waits for, and offers no option to press", () => {
    const { decide } = mount("set_explain", {
      line: {
        status: "waiting",
        waiting_for: [{ key: "models", stage: "models", name: "the models" }],
      },
    });
    expect(screen.getByTestId("waiting")).toHaveTextContent("Waiting for: the models");
    expect(document.querySelectorAll("[role=option]")).toHaveLength(0);
    expect(decide).not.toHaveBeenCalled();
  });
});

const PREVIEW: PreviewResult = {
  kind: "set_selection",
  views: [{ kind: "lineage", title: "Terms kept", caption: "9 of 14 terms stay on this fold." }],
  basis: "the training rows",
  note: null,
  caution: null,
} as unknown as PreviewResult;

const pointAt = (option: string) => fireEvent.focus(screen.getByTestId(`option-${option}`));

describe("the words on an option are the engine's", () => {
  it("shows the caption of the preview the engine serves for the option pointed at", async () => {
    vi.spyOn(api, "preview").mockResolvedValue(PREVIEW);
    mount("set_selection");
    pointAt("method:elastic_net");
    const option = screen.getByTestId("option-method:elastic_net");
    expect(await within(option).findByTestId("words")).toHaveTextContent(
      "9 of 14 terms stay on this fold.",
    );
    // only the option pointed at
    expect(within(screen.getByTestId("option-method:vip")).queryByTestId("words")).toBeNull();
  });

  it("previews the decision the option would record", async () => {
    const preview = vi.spyOn(api, "preview").mockResolvedValue(PREVIEW);
    mount("set_selection");
    fireEvent.change(screen.getByTestId("field-keep"), { target: { value: "25" } });
    pointAt("method:screening");
    await waitFor(() => expect(preview).toHaveBeenCalled());
    expect(preview.mock.calls[0]![1]).toMatchObject({
      kind: "set_selection",
      method: "screening",
      keep: 25,
    });
  });

  it("falls back on the note the engine gives when it can draw nothing", async () => {
    vi.spyOn(api, "preview").mockResolvedValue({
      ...PREVIEW,
      views: [],
      note: "Nothing is drawn before Fit is pressed.",
    });
    mount("set_updating");
    pointAt("method:shrinkage");
    expect(
      await within(screen.getByTestId("option-method:shrinkage")).findByTestId("words"),
    ).toHaveTextContent("Nothing is drawn before Fit is pressed.");
  });

  it("gives the refusal's reason where the engine would refuse the option", async () => {
    const refusal: Refusal = {
      error: {
        code: "decision_support_task",
        message: "A decision curve weighs a yes/no outcome.",
        exits: [],
      },
    };
    vi.spyOn(api, "preview").mockRejectedValue(new RefusalError(refusal));
    mount("set_intended_use");
    pointAt("use:decision_support");
    const words = await within(screen.getByTestId("option-use:decision_support")).findByTestId(
      "words",
    );
    expect(words).toHaveTextContent("A decision curve weighs a yes/no outcome.");
    expect(words).toHaveAttribute("data-refused");
  });

  it("names the option and stops where the engine serves nothing to say", async () => {
    const preview = vi
      .spyOn(api, "preview")
      .mockResolvedValue({ ...PREVIEW, views: [], note: null, caution: null });
    mount("set_explain");
    pointAt("curves:ale");
    await waitFor(() => expect(preview).toHaveBeenCalled());
    await settle();
    expect(document.querySelectorAll("[data-testid=words]")).toHaveLength(0);
    expect(screen.getByTestId("option-curves:ale")).toHaveTextContent("Accumulated local effects");
  });

  it("says nothing of an option before it is pointed at", async () => {
    const preview = vi.spyOn(api, "preview").mockResolvedValue(PREVIEW);
    mount("set_selection");
    await settle();
    expect(preview).not.toHaveBeenCalled();
    expect(document.querySelectorAll("[data-testid=words]")).toHaveLength(0);
  });
});

describe("a press the server does not record answers at the option", () => {
  it("shows the refusal and its exits, and an exit records its own decision", async () => {
    const exit: Decision = {
      kind: "set_levers",
      forms: "none",
      variance_filter: "top",
      keep: 1000,
      imbalance: "none",
    };
    const refusal: Refusal = {
      error: {
        code: "keep_how_many",
        message: "Keeping the most variable predictors needs how many to keep.",
        exits: [{ label: "Keep the 1,000 most variable", decision: exit }],
      },
    };
    const { decide } = mount("set_levers");
    decide.mockRejectedValueOnce(new RefusalError(refusal));
    fireEvent.click(screen.getByTestId("option-variance_filter:top"));
    const note = await screen.findByTestId("refusal");
    expect(note).toHaveTextContent("Keeping the most variable predictors needs how many to keep.");
    fireEvent.click(within(note).getByTestId("refusal-exit"));
    await waitFor(() => expect(decide).toHaveBeenCalledTimes(2));
    expect(decide.mock.calls[1]).toEqual([PID, exit]);
  });

  it("says the server did not record it where the failure is not a refusal", async () => {
    const { decide } = mount("set_updating");
    decide.mockRejectedValueOnce(new ApiError(500, "The server answered 500.", null));
    fireEvent.click(screen.getByTestId("option-method:none"));
    expect(await screen.findByTestId("failure")).toHaveTextContent("The server answered 500.");
  });
});

const type = (testid: string, value: string) =>
  fireEvent.change(screen.getByTestId(testid), { target: { value } });
const press = (option: string) => fireEvent.click(screen.getByTestId(`option-${option}`));

describe("a press the card will not send says why, at the option", () => {
  const folds = FIXTURE.bounds.set_validation.folds;
  const threshold = FIXTURE.bounds.set_intended_use.threshold;
  const keep = FIXTURE.bounds.set_selection.keep;

  it("names the field and its whole-number range, the numbers from the model's bounds", () => {
    const { decide } = mount("set_validation");
    type("field-folds", "11");
    press("validation:kfold");
    expect(decide).not.toHaveBeenCalled();
    expect(screen.getByTestId("blocked")).toHaveTextContent(
      `Folds: a whole number from ${folds.ge} to ${folds.le}`,
    );
  });

  it("names the range of a probability, open at both ends", () => {
    const { decide } = mount("set_intended_use");
    type("field-threshold", "1");
    press("use:decision_support");
    expect(decide).not.toHaveBeenCalled();
    expect(screen.getByTestId("blocked")).toHaveTextContent(
      `Decision threshold: a number above ${threshold.gt} and below ${threshold.lt}`,
    );
  });

  it("names the range of a count with no top", () => {
    mount("set_selection");
    type("field-keep", "0");
    press("method:screening");
    expect(screen.getByTestId("blocked")).toHaveTextContent(
      `Predictors to keep (screening): a whole number of at least ${keep.ge}`,
    );
  });

  it("writes a number over a thousand as the page does", () => {
    mount("set_validation");
    type("field-n_boot", "19");
    press("validation:bootstrap");
    expect(screen.getByTestId("blocked")).toHaveTextContent(
      `Resamples (bootstrap): a whole number from ${FIXTURE.bounds.set_validation.n_boot.ge} to 2,000`,
    );
  });

  it("names a blank number the model needs, and marks the field", () => {
    const { decide } = mount("set_validation");
    type("field-folds", "");
    expect(screen.getByTestId("field-folds")).toHaveAttribute("aria-invalid", "true");
    press("validation:kfold");
    expect(decide).not.toHaveBeenCalled();
    expect(screen.getByTestId("blocked")).toHaveTextContent(
      `Folds: a whole number from ${folds.ge} to ${folds.le}`,
    );
  });

  it.each([
    ["set_validation", "field-folds", "validation:kfold"],
    ["set_validation", "field-repeats", "validation:kfold"],
    ["set_validation", "field-n_boot", "validation:kfold"],
    ["set_explain", "field-reseeds", "curves:ale"],
  ] as const)("%s: a blank %s is marked, and a number in range is not", (kind, field, option) => {
    const { decide } = mount(kind);
    const input = screen.getByTestId(field);
    expect(input).not.toHaveAttribute("aria-invalid");
    type(field, "");
    expect(input).toHaveAttribute("aria-invalid", "true");
    press(option);
    expect(decide).not.toHaveBeenCalled();
    expect(screen.getByTestId("blocked")).toBeInTheDocument();
  });

  it("leaves a blank that the model takes as none unmarked, and sends the press", async () => {
    // The screening count is optional: the model's own default is none, and the engine answers
    // a screen with no count in its own words.
    const { decide } = mount("set_selection");
    expect(screen.getByTestId("field-keep")).toHaveValue(null);
    expect(screen.getByTestId("field-keep")).not.toHaveAttribute("aria-invalid");
    press("method:screening");
    await waitFor(() => expect(decide).toHaveBeenCalledTimes(1));
    expect(screen.queryByTestId("blocked")).toBeNull();
  });

  it("stays silent where the field is one the option does not read", async () => {
    // A count out of range matters to screening alone: elastic net reads none of the settings.
    const { decide } = mount("set_selection");
    type("field-keep", "0");
    press("method:elastic_net");
    await waitFor(() => expect(decide).toHaveBeenCalledTimes(1));
    expect(screen.queryByTestId("blocked")).toBeNull();
  });

  it("stays silent for risk estimation, which carries no threshold", async () => {
    const { decide } = mount("set_intended_use");
    type("field-threshold", "1");
    press("use:risk_estimation");
    await waitFor(() => expect(decide).toHaveBeenCalledTimes(1));
    expect(screen.queryByTestId("blocked")).toBeNull();
  });

  it("names every field in the way, in one note", () => {
    mount("set_validation");
    type("field-folds", "11");
    type("field-repeats", "1");
    press("validation:kfold");
    const note = screen.getByTestId("blocked");
    expect(note).toHaveTextContent("Folds: a whole number from");
    expect(note).toHaveTextContent("Repeats (repeated k-fold): a whole number from");
    expect(screen.getAllByTestId("blocked")).toHaveLength(1);
  });

  it("goes when the field is put right, and when another press is sent", async () => {
    const { decide } = mount("set_validation");
    type("field-folds", "11");
    press("validation:kfold");
    expect(screen.getByTestId("blocked")).toBeInTheDocument();
    type("field-folds", "5");
    expect(screen.queryByTestId("blocked")).toBeNull();
    press("validation:kfold");
    await waitFor(() => expect(decide).toHaveBeenCalledTimes(1));
    expect(screen.queryByTestId("blocked")).toBeNull();
  });

  it("is under the option pressed, and moves with the press", () => {
    mount("set_validation");
    type("field-folds", "11");
    press("validation:kfold");
    expect(screen.getByTestId("option-validation:kfold").nextElementSibling).toContainElement(
      screen.getByTestId("blocked"),
    );
    type("field-repeats", "1");
    type("field-folds", "5");
    press("validation:repeated_kfold");
    expect(screen.getAllByTestId("blocked")).toHaveLength(1);
    expect(
      screen.getByTestId("option-validation:repeated_kfold").nextElementSibling,
    ).toContainElement(screen.getByTestId("blocked"));
  });

  it("asks for the group column of internal–external validation, which the card sends with it", () => {
    const { decide } = mount("set_validation");
    press("validation:internal_external");
    expect(decide).not.toHaveBeenCalled();
    expect(screen.getByTestId("blocked")).toHaveTextContent(
      "Group column (internal–external): choose a column",
    );
    expect(screen.getByTestId("field-cluster")).toHaveAttribute("aria-invalid", "true");
    fireEvent.change(screen.getByTestId("field-cluster"), {
      target: { value: "cycle_begin_year" },
    });
    expect(screen.queryByTestId("blocked")).toBeNull();
  });

  it("names the count the most-variable filter keeps, only when that filter would be sent", async () => {
    const { decide } = mount("set_levers");
    type("field-keep", "0");
    press("variance_filter:top");
    expect(screen.getByTestId("blocked")).toHaveTextContent(
      `Predictors to keep (most variable): a whole number of at least ${keep.ge}`,
    );
    // another lever, with the filter still off, does not carry the count
    press("forms:rule");
    await waitFor(() => expect(decide).toHaveBeenCalledTimes(1));
    expect(screen.queryByTestId("blocked")).toBeNull();
  });
});

describe("what the explanations card says an empty column list draws", () => {
  const legend = () => screen.getByRole("group", { name: /^Curves for/ });

  it("is the declared study factor under inference, where the engine draws that", () => {
    mount("set_explain", { recorded: { purpose: "inference" } });
    expect(legend()).toHaveTextContent("Curves for (none chosen: the declared study factor)");
    expect(legend()).not.toHaveTextContent("matter most");
  });

  it("is the predictors that matter most to the model under prediction", () => {
    mount("set_explain");
    expect(legend()).toHaveTextContent(
      "Curves for (none chosen: the predictors that matter most to the model)",
    );
    expect(legend()).not.toHaveTextContent("declared");
  });

  it.each(["prediction", "inference"])(
    "keeps exposure a quiet term, never in the card's words (%s)",
    (purpose) => {
      mount("set_explain", { recorded: { purpose } });
      expect(legend().textContent ?? "").not.toMatch(/exposure/i);
    },
  );
});

// ── the design repair (W1 part 2) ────────────────────────────────────────────

describe("the sensitivity analyses' card (P1)", () => {
  // The model (core/decisions.py `SetSensitivity`, `SensitivityAnalysis`): `analyses`, each a label
  // and its own exclusion rules; an analysis with no rules keeps every row, and an empty list is the
  // answer "no sensitivity analysis". The every-row analysis is labeled as the engine's own cases
  // label it (core/tests/materiality_cases.py `EVERY_ROW`).
  const EVERY_ROW = { kind: "set_sensitivity", analyses: [{ label: "Every row", rules: [] }] };
  const NONE = { kind: "set_sensitivity", analyses: [] };

  it("posts the every-row analysis, and no analysis, as the model defines them", async () => {
    const { decide } = mount("set_sensitivity");
    expect(Array.from(document.querySelectorAll("[role=option]")).map((o) => o.getAttribute("data-key"))).toEqual([
      "analyses:every_row",
      "analyses:none",
    ]);
    fireEvent.click(screen.getByTestId("option-analyses:every_row"));
    await waitFor(() => expect(decide).toHaveBeenCalledTimes(1));
    expect(decide.mock.calls[0]).toEqual([PID, EVERY_ROW]);
    fireEvent.click(screen.getByTestId("option-analyses:none"));
    await waitFor(() => expect(decide).toHaveBeenCalledTimes(2));
    expect(decide.mock.calls[1]).toEqual([PID, NONE]);
  });

  it("marks the answer on record, and none for analyses set on rules of their own", () => {
    mount("set_sensitivity", { recorded: { sensitivity: EVERY_ROW.analyses }, answered: true });
    expect(screen.getByTestId("option-analyses:every_row")).toHaveAttribute("data-recorded");
    cleanup();
    mount("set_sensitivity", { recorded: { sensitivity: [] }, answered: true });
    expect(screen.getByTestId("option-analyses:none")).toHaveAttribute("data-recorded");
    cleanup();
    const own = [{ label: "800–4,000 kcal", rules: [{ column: "kcal", low: 800, high: 4000 }] }];
    mount("set_sensitivity", { recorded: { sensitivity: own }, answered: true });
    expect(document.querySelectorAll("[data-recorded]")).toHaveLength(0);
  });
});

describe("a default set for you (P10)", () => {
  const TAG = "set for you";

  it("marks the scheme in force as set for you while the line is a default, and not once answered", () => {
    mount("set_validation", { recorded: { split: { ...DEFAULT_SPLIT, validation: "bootstrap" } }, line: { label: "Confirm", status: "set_for_you" } });
    const tagged = Array.from(document.querySelectorAll("[role=option]")).filter((o) => o.textContent?.includes(TAG));
    expect(tagged.map((o) => o.getAttribute("data-key"))).toEqual(["validation:bootstrap"]);
    cleanup();
    mount("set_validation", { recorded: { split: { ...DEFAULT_SPLIT, validation: "bootstrap" } }, answered: true });
    expect(document.body.textContent).not.toContain(TAG);
  });

  it("marks each lever in force as set for you, the model's defaults where none is recorded", () => {
    mount("set_levers", { line: { label: "Confirm", status: "set_for_you" } });
    const tagged = Array.from(document.querySelectorAll("[role=option]")).filter((o) => o.textContent?.includes(TAG));
    const d = FIXTURE.defaults.set_levers;
    expect(tagged.map((o) => o.getAttribute("data-key"))).toEqual([`forms:${d.forms}`, `variance_filter:${d.variance_filter}`, `imbalance:${d.imbalance}`]);
  });

  it("states the scheme in force in the card's words, each number the split's", async () => {
    const { validationDefault } = await import("./ValidationCard");
    const split = (over: Record<string, unknown>) => stateWith({ split: { ...DEFAULT_SPLIT, ...over } });
    expect(validationDefault(split({}))).toBe(`${DEFAULT_SPLIT.folds}-fold cross-validation.`);
    expect(validationDefault(split({ validation: "repeated_kfold", folds: 8, repeats: 30 }))).toBe("8-fold cross-validation, repeated 30 times.");
    expect(validationDefault(split({ validation: "bootstrap", n_boot: 700 }))).toBe("Bootstrap optimism correction over 700 resamples.");
    expect(validationDefault(split({ validation: "internal_external", cluster: "cycle_begin_year" }))).toBe("Internal–external validation by cycle_begin_year.");
    expect(validationDefault(split({ nested_cv: true }))).toBe(`${DEFAULT_SPLIT.folds}-fold cross-validation, with a nested cross-validation interval.`);
    // No split recorded: the model's defaults (the fixture's, held to decisions.py).
    expect(validationDefault(stateWith({ split: null }))).toBe(`${FIXTURE.defaults.set_validation.folds}-fold cross-validation.`);
  });

  it("states the levers in force in the card's words", async () => {
    const { leversDefault } = await import("./LeversCard");
    expect(leversDefault(stateWith({ levers: null }))).toBe("Predictor forms: none; variance filter: none; imbalance correction: none.");
    expect(leversDefault(stateWith({ levers: { forms: "rule", variance_filter: "top", keep: 50, imbalance: "weights" } }))).toBe(
      "Predictor forms: spline by rule; variance filter: most variable (50 kept); imbalance correction: class weights.",
    );
  });
});
