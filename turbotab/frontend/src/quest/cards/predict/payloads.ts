/**
 * What each Predict declaration card posts, as pure functions of the card's fields.
 *
 * Every payload is the whole of the model in turbotab/core/decisions.py (SetIntendedUse,
 * SetSelection, SetValidation, SetLevers, SetUpdating, SetExplain): the type is the generated
 * contract, so a field the server adds or drops breaks the build here. The choices are exactly
 * the values the model accepts, and `Exactly` fails to compile if a list misses or invents one.
 * The defaults and the bounds are the model's; payloads.fixture.json pins them, and
 * turbotab/core/tests/test_w1_declaration_cards.py checks that file against the models.
 */
import type { Decision } from "../../../api/schema";

type Of<K extends Decision["kind"]> = Extract<Decision, { kind: K }>;
export type IntendedUse = Of<"set_intended_use">;
export type Selection = Of<"set_selection">;
export type Validation = Of<"set_validation">;
export type Levers = Of<"set_levers">;
export type Updating = Of<"set_updating">;
export type Explain = Of<"set_explain">;

type Exactly<L extends readonly unknown[], U> = [U] extends [L[number]]
  ? [L[number]] extends [U]
    ? L
    : never
  : never;

const uses = ["risk_estimation", "decision_support"] as const;
export const USES: Exactly<typeof uses, IntendedUse["use"]> = uses;
const fairness = ["subgroup_performance", "none"] as const;
export const FAIRNESS: Exactly<typeof fairness, IntendedUse["fairness"]> = fairness;

const methods = [
  "none",
  "elastic_net",
  "stability",
  "screening",
  "stepwise",
  "univariable",
  "vip",
] as const;
export const METHODS: Exactly<typeof methods, Selection["method"]> = methods;
const preSelected = ["no", "yes", "other_data", "unknown"] as const;
export const PRE_SELECTED: Exactly<
  typeof preSelected,
  NonNullable<Selection["pre_selected"]>
> = preSelected;

const validations = ["kfold", "repeated_kfold", "bootstrap", "internal_external"] as const;
export const VALIDATIONS: Exactly<typeof validations, Validation["validation"]> = validations;

const forms = ["none", "rule", "inner_cv"] as const;
export const FORMS: Exactly<typeof forms, Levers["forms"]> = forms;
const varianceFilters = ["none", "near_zero", "top"] as const;
export const VARIANCE_FILTERS: Exactly<typeof varianceFilters, Levers["variance_filter"]> =
  varianceFilters;
const imbalances = ["none", "weights", "undersample", "oversample"] as const;
export const IMBALANCES: Exactly<typeof imbalances, Levers["imbalance"]> = imbalances;

const updatingMethods = ["none", "shrinkage"] as const;
export const UPDATING_METHODS: Exactly<typeof updatingMethods, Updating["method"]> =
  updatingMethods;

const curves = ["ale", "partial_dependence"] as const;
export const CURVES: Exactly<typeof curves, Explain["curves"]> = curves;

/** A field's bounds: `gt` and `lt` exclusive, `ge` and `le` inclusive (pydantic's `Field`). */
export interface Bounds {
  ge?: number;
  le?: number;
  gt?: number;
  lt?: number;
}

/** The numeric fields' bounds, by decision kind and field (decisions.py). */
export const BOUNDS = {
  set_intended_use: {
    threshold_low: { gt: 0, lt: 1 },
    threshold_high: { gt: 0, lt: 1 },
    threshold: { gt: 0, lt: 1 },
  },
  set_selection: { keep: { ge: 1 }, threshold: { gt: 0, lt: 1 }, q: { ge: 1 } },
  set_validation: {
    folds: { ge: 2, le: 10 },
    repeats: { ge: 2, le: 50 },
    n_boot: { ge: 20, le: 2000 },
  },
  set_levers: { keep: { ge: 1 } },
  set_explain: { reseeds: { ge: 0, le: 20 } },
} as const satisfies Record<string, Record<string, Bounds>>;

/** The model's default for each field that has one (decisions.py). A field with none is absent. */
export const DEFAULTS = {
  set_intended_use: {
    threshold_low: null,
    threshold_high: null,
    threshold: null,
    subgroups: [] as string[],
    fairness: "subgroup_performance",
  },
  set_selection: {
    method: "none",
    where: "in_fold",
    keep: null,
    threshold: null,
    q: null,
    pre_selected: null,
    sensitivity: false,
  },
  set_validation: {
    folds: 5,
    validation: "kfold",
    repeats: 10,
    n_boot: 500,
    cluster: null,
    nested_cv: false,
    fold_order: null,
    cut_point: null,
    acknowledged: false,
  },
  set_levers: { forms: "none", variance_filter: "none", keep: null, imbalance: "none" },
  set_updating: { method: "none" },
  set_explain: { curves: "ale", exposures: [] as string[], reseeds: 5, as_effect: false },
} as const;

/** A number field as typed: blank is none, a number inside the bounds is a number, anything else
 *  is `invalid` (the card does not post it; the server would answer 422, which says nothing). A
 *  blank in a field the model needs a number for (`required`) is invalid as well. */
export type Typed = number | null | "invalid";

export function parseTyped(
  text: string,
  bounds: Bounds,
  integer: boolean,
  required: true,
): number | "invalid";
export function parseTyped(
  text: string,
  bounds: Bounds,
  integer: boolean,
  required?: boolean,
): Typed;
export function parseTyped(
  text: string,
  bounds: Bounds,
  integer: boolean,
  required = false,
): Typed {
  const t = text.trim();
  if (t === "") return required ? "invalid" : null;
  const n = Number(t);
  if (!Number.isFinite(n) || (integer && !Number.isInteger(n))) return "invalid";
  if (bounds.ge !== undefined && n < bounds.ge) return "invalid";
  if (bounds.le !== undefined && n > bounds.le) return "invalid";
  if (bounds.gt !== undefined && n <= bounds.gt) return "invalid";
  if (bounds.lt !== undefined && n >= bounds.lt) return "invalid";
  return n;
}

const shown = (n: number) => n.toLocaleString("en-US");

/** The numbers a field takes, in words, from its bounds: "a whole number from 2 to 10", "a number
 *  above 0 and below 1", "a whole number of at least 1". */
export function describeRange(b: Bounds, integer: boolean): string {
  const noun = integer ? "a whole number" : "a number";
  if (b.ge !== undefined && b.le !== undefined)
    return `${noun} from ${shown(b.ge)} to ${shown(b.le)}`;
  const edges = [
    b.ge !== undefined ? `of at least ${shown(b.ge)}` : null,
    b.gt !== undefined ? `above ${shown(b.gt)}` : null,
    b.le !== undefined ? `of at most ${shown(b.le)}` : null,
    b.lt !== undefined ? `below ${shown(b.lt)}` : null,
  ].filter((edge) => edge !== null);
  return [noun, edges.join(" and ")].join(" ").trim();
}

/** What stands in the way of sending a typed number: the field, then what it takes, or nothing. */
export const trouble = (name: string, typed: Typed, bounds: Bounds, integer: boolean): string[] =>
  typed === "invalid" ? [`${name}: ${describeRange(bounds, integer)}`] : [];

// ── the payloads ─────────────────────────────────────────────────────────────

export interface IntendedUseFields {
  threshold_low: number | null;
  threshold_high: number | null;
  threshold: number | null;
  subgroups: string[];
  fairness: IntendedUse["fairness"];
}

/** The thresholds belong to decision support alone: a risk estimate has none to declare. */
export function intendedUse(use: IntendedUse["use"], f: IntendedUseFields): IntendedUse {
  const decision = use === "decision_support";
  return {
    kind: "set_intended_use",
    use,
    threshold_low: decision ? f.threshold_low : null,
    threshold_high: decision ? f.threshold_high : null,
    threshold: decision ? f.threshold : null,
    subgroups: f.subgroups,
    fairness: f.fairness,
  };
}

export interface SelectionFields {
  keep: number | null;
  threshold: number | null;
  q: number | null;
  pre_selected: Selection["pre_selected"];
}

/** Which of a method's own settings the engine reads (models/variable_selection.py): the number
 *  screening keeps; the threshold of stability selection and of the univariable screen; the
 *  count stability selection controls. A setting typed for another method is not sent. */
export type Setting = "keep" | "threshold" | "q";
export const SETTINGS_OF: Record<Selection["method"], readonly Setting[]> = {
  none: [],
  elastic_net: [],
  stability: ["threshold", "q"],
  screening: ["keep"],
  stepwise: [],
  univariable: ["threshold"],
  vip: [],
};

export function selection(method: Selection["method"], f: SelectionFields): Selection {
  const reads = SETTINGS_OF[method];
  return {
    kind: "set_selection",
    method,
    where: "in_fold",
    keep: reads.includes("keep") ? f.keep : null,
    threshold: reads.includes("threshold") ? f.threshold : null,
    q: reads.includes("q") ? f.q : null,
    pre_selected: f.pre_selected,
    sensitivity: false,
  };
}

export interface ValidationFields {
  folds: number;
  repeats: number;
  n_boot: number;
  cluster: string | null;
  nested_cv: boolean;
}

/** A cluster column is read by internal–external validation alone, and it needs one. */
export function validation(kind: Validation["validation"], f: ValidationFields): Validation | null {
  const cluster = kind === "internal_external" ? f.cluster : null;
  if (kind === "internal_external" && !cluster) return null;
  return {
    kind: "set_validation",
    folds: f.folds,
    validation: kind,
    repeats: f.repeats,
    n_boot: f.n_boot,
    cluster,
    nested_cv: f.nested_cv,
    // The folds' order and the draw's cut-point are the seal's (SealAsk): null keeps them.
    fold_order: null,
    cut_point: null,
    acknowledged: false,
  };
}

/** The levers as they stand, or as the model defaults them: each option changes one of three and
 *  sends the others back as they are, as the engine's own offers do (stages/explore.py). */
export type LeverRules = Pick<Levers, "forms" | "variance_filter" | "imbalance"> & {
  keep: number | null;
};

export function levers(rules: LeverRules): Levers {
  return {
    kind: "set_levers",
    forms: rules.forms,
    variance_filter: rules.variance_filter,
    // How many to keep is read by the "most variable" filter alone.
    keep: rules.variance_filter === "top" ? rules.keep : null,
    imbalance: rules.imbalance,
  };
}

export const updating = (method: Updating["method"]): Updating => ({
  kind: "set_updating",
  method,
});

// ── the sensitivity analyses (set_sensitivity) ──
// Not among the six W1-F cards payloads.fixture.json pins (test_w1_declaration_cards.py holds that
// fixture to six models): this card came with the design repair, because a Predict project with an
// exclusion rule meets the line as a counted Decide (core/quest.py DECLARATIONS) and could not
// reach Fit without it. Its type is the generated model's, so a field the server changes breaks
// the build here; cards.test.tsx holds its two payloads to the model's own definition.

export type Sensitivity = Of<"set_sensitivity">;

/** The two answers the card offers: the every-row analysis beside the primary, or none. */
export const SENSITIVITIES = ["every_row", "none"] as const;
export type SensitivityChoice = (typeof SENSITIVITIES)[number];

/** The every-row analysis's label, as the engine's own cases name it (core/tests/materiality_cases.py
 *  `EVERY_ROW`): the methods sentence ticks it ("`Every row`, keeping every row", voice.py). */
export const EVERY_ROW = "Every row";

/**
 * The every-row analysis is the same model on every row, no exclusion rule applied (an analysis
 * with no rules keeps every row: decisions.py `SensitivityAnalysis`), as Banna et al. 2017 ask:
 * "analyses in the total sample without exclusion of participants should also be conducted and
 * reported". No analysis is the empty list, which the model reads as "no sensitivity analysis".
 */
export function sensitivity(choice: SensitivityChoice): Sensitivity {
  return {
    kind: "set_sensitivity",
    analyses: choice === "every_row" ? [{ label: EVERY_ROW, rules: [] }] : [],
  };
}

/** Which of the card's answers a recorded list is, or null for any other (analyses set elsewhere,
 *  on rules of their own). */
export function sensitivityOf(
  analyses: readonly { label: string; rules: readonly unknown[] }[] | null | undefined,
): SensitivityChoice | null {
  if (!analyses) return null;
  if (analyses.length === 0) return "none";
  return analyses.length === 1 && analyses[0]!.rules.length === 0 ? "every_row" : null;
}

export interface ExplainFields {
  exposures: string[];
  reseeds: number;
}

/** `as_effect` is never asked: the engine refuses an explanation reported as an effect under both
 *  purposes (decisions.py, ExplainSpec), so the card always says it is not one. */
export function explain(curves: Explain["curves"], f: ExplainFields): Explain {
  return {
    kind: "set_explain",
    curves,
    exposures: f.exposures,
    reseeds: f.reseeds,
    as_effect: false,
  };
}
